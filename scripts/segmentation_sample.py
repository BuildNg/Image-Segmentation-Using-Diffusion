"""
Sample LIDC masks on a single GPU and evaluate 4x4 ambiguity metrics.
"""

import argparse
import random
import sys
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


def to_hard_mask(sample, rule):
    sample = sample.float()
    if rule == "auto":
        if th.all((sample == 0) | (sample == 1)):
            return sample
        sample_min = float(sample.min().item())
        sample_max = float(sample.max().item())
        threshold = 0.5 if sample_min >= 0.0 and sample_max <= 1.0 else 0.0
        return (sample > threshold).float()
    if rule == "threshold_0_5":
        return (sample > 0.5).float()
    if rule == "threshold_0_0":
        return (sample > 0.0).float()
    raise ValueError(f"Unsupported hard mask rule: {rule}")


def get_case_id(path_str):
    return Path(path_str).parent.name


def main():
    args = create_argparser().parse_args()
    if args.batch_size != 1:
        raise ValueError("This evaluation script currently supports --batch_size 1 only.")

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

    dataset = LIDCDataset(args.data_dir, test_flag=True)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    total_cases = parse_num_samples(args.num_samples, len(dataset))
    if total_cases == 0:
        raise ValueError(f"No test cases found in {args.data_dir}.")

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

    logger.log(f"evaluating {total_cases} case(s)...")
    for case_idx, (image, expert_masks, path) in enumerate(dataloader):
        if case_idx >= total_cases:
            break

        image = image.to(dist_util.dev()).float()
        expert_masks = (expert_masks.to(dist_util.dev()).float() > 0.5).float()
        case_id = get_case_id(path[0])

        preds = []
        with th.no_grad():
            for _ in range(args.num_ensemble):
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
                preds.append(to_hard_mask(sample, args.hard_mask_rule))

        preds = th.stack(preds, dim=0)  # [M, B, C, H, W]
        gts = expert_masks.permute(1, 0, 2, 3).unsqueeze(2)  # [N, B, C, H, W]

        ged = tensor_scalar(generalized_energy_distance(preds, gts))
        dmax = tensor_scalar(max_dice(preds, gts))
        sc = tensor_scalar(combined_sensitivity(preds, gts))
        da = tensor_scalar(diversity_agreement(preds, gts))
        ci, _, _, _ = collective_insight(preds, gts)
        ci = tensor_scalar(ci)

        metric_values["GED"].append(ged)
        metric_values["D_max"].append(dmax)
        metric_values["Sc"].append(sc)
        metric_values["D_a"].append(da)
        metric_values["CI"].append(ci)

        print(
            f"case={case_id} GED={ged:.6f} D_max={dmax:.6f} "
            f"Sc={sc:.6f} D_a={da:.6f} CI={ci:.6f}"
        )

    print("")
    print(f"Evaluated cases: {len(metric_values['GED'])}")
    for metric_name in ["GED", "D_max", "Sc", "D_a", "CI"]:
        values = np.asarray(metric_values[metric_name], dtype=np.float64)
        print(
            f"{metric_name}: mean={values.mean():.6f} std={values.std(ddof=0):.6f}"
        )


def create_argparser():
    defaults = dict(
        data_dir="./data/testing",
        clip_denoised=True,
        batch_size=1,
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
        default="auto",
        choices=["auto", "threshold_0_5", "threshold_0_0"],
        help="Rule to binarize sampled masks before metric computation.",
    )
    return parser


if __name__ == "__main__":
    main()
