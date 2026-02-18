"""
Sample LIDC masks on a single GPU and evaluate 4x4 ambiguity metrics.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

sys.path.append(".")

from guided_diffusion import dist_util, logger
from guided_diffusion.lidcloader import LIDCDataset
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from metrics import (
    collective_insight,
    combined_sensitivity,
    diversity_agreement,
    generalized_energy_distance,
    max_dice,
)

seed = 10
th.manual_seed(seed)
if th.cuda.is_available():
    th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)


def parse_num_samples(num_samples_arg, dataset_size):
    value = str(num_samples_arg).strip().lower()
    if value == "all":
        return dataset_size
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("--num_samples must be a positive integer or 'all'.")
    return min(parsed, dataset_size)


def tensor_scalar(x):
    if isinstance(x, th.Tensor):
        return float(x.detach().float().mean().cpu().item())
    return float(x)


def infer_auto_threshold(sample):
    sample_min = float(sample.min().item())
    sample_max = float(sample.max().item())
    threshold = 0.5 if sample_min >= 0.0 and sample_max <= 1.0 else 0.0
    return threshold, sample_min, sample_max


def to_hard_mask(sample, threshold):
    return (sample.float() > float(threshold)).float()


def get_case_id(path_str):
    return Path(path_str).parent.name


def summarize_metrics(metric_values):
    summary = {}
    for metric_name, values in metric_values.items():
        arr = np.asarray(values, dtype=np.float64)
        summary[metric_name] = (arr.mean(), arr.std(ddof=0))
    return summary


def main():
    args = create_argparser().parse_args()

    logger.configure()
    logger.log("creating model and diffusion...")
    model, diffusion, _, _ = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    print(f"using checkpoint: {args.model_path}")
    print(
        f"device={dist_util.dev()} batch_size={args.batch_size} "
        f"num_ensemble={args.num_ensemble} sampler={'ddim' if args.use_ddim else 'ddpm'}"
    )

    dataset = LIDCDataset(args.data_dir, test_flag=True)
    dataloader = th.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )
    total_cases = parse_num_samples(args.num_samples, len(dataset))
    if total_cases == 0:
        raise ValueError(f"No test cases found in {args.data_dir}.")
    total_batches = (total_cases + args.batch_size - 1) // args.batch_size

    metric_values = {
        "GED": [],
        "D_max": [],
        "Sc": [],
        "D_a": [],
        "CI": [],
    }

    sample_fn = (
        diffusion.p_sample_loop_known
        if not args.use_ddim
        else diffusion.ddim_sample_loop_known
    )
    fixed_threshold = None
    auto_threshold = None
    if args.hard_mask_rule == "threshold_0_5":
        fixed_threshold = 0.5
    elif args.hard_mask_rule == "threshold_0_0":
        fixed_threshold = 0.0

    logger.log(f"evaluating {total_cases} case(s)...")
    evaluated = 0
    for batch_idx, (image, expert_masks, path) in enumerate(dataloader, start=1):
        if evaluated >= total_cases:
            break

        batch_start = time.time()
        remaining = total_cases - evaluated
        if image.shape[0] > remaining:
            image = image[:remaining]
            expert_masks = expert_masks[:remaining]
            path = path[:remaining]
        case_ids = [get_case_id(p) for p in path]
        range_info = case_ids[0] if len(case_ids) == 1 else f"{case_ids[0]}..{case_ids[-1]}"
        print(
            f"[batch {batch_idx}/{total_batches}] start "
            f"batch_cases={len(case_ids)} evaluated={evaluated}/{total_cases} "
            f"cases={range_info}"
        )

        image = image.to(dist_util.dev()).float()
        expert_masks = (expert_masks.to(dist_util.dev()).float() > 0.5).float()

        preds = []
        with th.no_grad():
            for ensemble_idx in range(args.num_ensemble):
                ensemble_start = time.time()
                # p_sample_loop_known expects [image_channels + one mask/noise channel].
                input_pair = th.cat((image, th.zeros_like(image[:, :1, ...])), dim=1)
                sample, _, _ = sample_fn(
                    model,
                    (
                        image.shape[0],
                        args.image_in_channels + 1,
                        args.image_size,
                        args.image_size,
                    ),
                    input_pair,
                    clip_denoised=args.clip_denoised,
                    model_kwargs={},
                )
                if fixed_threshold is not None:
                    preds.append(to_hard_mask(sample, fixed_threshold))
                else:
                    if auto_threshold is None:
                        auto_threshold, smin, smax = infer_auto_threshold(sample)
                        print(
                            f"[hard_mask_rule=auto] first sample range=({smin:.4f}, {smax:.4f}) "
                            f"-> fixed threshold={auto_threshold:.1f} for all cases."
                        )
                    preds.append(to_hard_mask(sample, auto_threshold))
                ensemble_elapsed = time.time() - ensemble_start
                print(
                    f"[batch {batch_idx}/{total_batches}] "
                    f"ensemble {ensemble_idx + 1}/{args.num_ensemble} "
                    f"done in {ensemble_elapsed:.2f}s"
                )

        preds = th.stack(preds, dim=0)  # [M, B, C, H, W]
        batch_size_cur = preds.shape[1]
        for b_idx in range(batch_size_cur):
            case_id = get_case_id(path[b_idx])
            preds_case = preds[:, b_idx : b_idx + 1, ...]
            gts_case = expert_masks[b_idx : b_idx + 1].permute(1, 0, 2, 3).unsqueeze(2)

            ged = tensor_scalar(generalized_energy_distance(preds_case, gts_case))
            dmax = tensor_scalar(max_dice(preds_case, gts_case))
            sc = tensor_scalar(combined_sensitivity(preds_case, gts_case))
            da = tensor_scalar(diversity_agreement(preds_case, gts_case))
            ci, _, _, _ = collective_insight(preds_case, gts_case)
            ci = tensor_scalar(ci)

            metric_values["GED"].append(ged)
            metric_values["D_max"].append(dmax)
            metric_values["Sc"].append(sc)
            metric_values["D_a"].append(da)
            metric_values["CI"].append(ci)

            evaluated += 1
            print(
                f"case={case_id} GED={ged:.6f} D_max={dmax:.6f} "
                f"Sc={sc:.6f} D_a={da:.6f} CI={ci:.6f}"
            )
            if evaluated % 20 == 0:
                running = summarize_metrics(metric_values)
                print(
                    f"[running mean @ {evaluated} cases] "
                    f"GED={running['GED'][0]:.6f} "
                    f"D_max={running['D_max'][0]:.6f} "
                    f"Sc={running['Sc'][0]:.6f} "
                    f"D_a={running['D_a'][0]:.6f} "
                    f"CI={running['CI'][0]:.6f}"
                )
        batch_elapsed = time.time() - batch_start
        print(
            f"[batch {batch_idx}/{total_batches}] done in {batch_elapsed:.2f}s "
            f"evaluated={evaluated}/{total_cases}"
        )

    print("")
    print(f"Evaluated cases: {len(metric_values['GED'])}")
    final_summary = summarize_metrics(metric_values)
    for metric_name in ["GED", "D_max", "Sc", "D_a", "CI"]:
        mean_value, std_value = final_summary[metric_name]
        print(
            f"{metric_name}: mean={mean_value:.6f} std={std_value:.6f}"
        )


def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        batch_size=8,
        use_ddim=False,
        model_path="",
        num_ensemble=4,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument(
        "--num_samples",
        type=str,
        default="1",
        help="Number of test cases to evaluate, or 'all'.",
    )
    parser.add_argument(
        "--hard_mask_rule",
        type=str,
        default="threshold_0_5",
        choices=["auto", "threshold_0_5", "threshold_0_0"],
        help=(
            "Rule to binarize sampled masks before metric computation. "
            "'threshold_0_5' is recommended for masks in [0,1]. "
            "'auto' infers the threshold once from the first sample and then keeps it fixed."
        ),
    )
    return parser


if __name__ == "__main__":
    main()
