"""
Continue training LDSeg_mod from a checkpoint.

Usage:
    python continue_train.py --checkpoint path/to/checkpoint.pth --config train_config.ini

Loads model weights, optimizer state, and scheduler state from the checkpoint,
then resumes training with the hyperparameters in the config file.

If the checkpoint only contains model weights (e.g. best_model.pth saved via
torch.save(model.state_dict(), ...)), the optimizer and scheduler are
initialised fresh and training starts from epoch 0.

Logging defaults to the same file as train.py (train.log in LogDir), but can
be overridden with --log-file.
"""

import os
import argparse
import configparser
import logging
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
import numpy as np

# Local imports
from dataloader import LIDCDataset, parse_augmentation_config
from LDSeg_mod import build_ldseg_from_config

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
GUIDED_DIFFUSION_DIR = os.path.join(PARENT_DIR, 'guided_diffusion')
GUIDED_DIFFUSION_NESTED_DIR = os.path.join(GUIDED_DIFFUSION_DIR, 'guided_diffusion')

sys_path_candidates = [PARENT_DIR]
if os.path.isdir(GUIDED_DIFFUSION_NESTED_DIR):
    sys_path_candidates.insert(0, GUIDED_DIFFUSION_DIR)

for path in reversed(sys_path_candidates):
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    from guided_diffusion.script_util import create_gaussian_diffusion
except ImportError as e:
    print(
        "Error: Could not import guided_diffusion. "
        f"Expected package path at: {GUIDED_DIFFUSION_DIR}"
    )
    print(f"ImportError details: {e}")
    sys.exit(1)


# ---------------------------------------------------------------------------
#  Re-use helpers from train.py
# ---------------------------------------------------------------------------
from train import (
    DiceLoss,
    get_lr_scheduler,
    validate,
    load_config,
    resolve_path,
)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue training LDSeg from a checkpoint"
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to the .pth checkpoint file'
    )
    parser.add_argument(
        '--config', type=str, default='train_config.ini',
        help='Path to training config file (default: train_config.ini)'
    )
    parser.add_argument(
        '--log-file', type=str, default=None,
        help='Override log filename (default: train.log inside LogDir)'
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
#  Checkpoint loading
# ---------------------------------------------------------------------------
def load_checkpoint(path, model, optimizer=None, scheduler=None, device='cpu'):
    """Load a checkpoint and return the starting epoch.

    Supports two checkpoint formats:
      1. Full checkpoint dict  (from train.py's periodic saves)
         Keys: model_state_dict, optimizer_state_dict, scheduler_state_dict,
               epoch, loss, val_loss
      2. Raw state_dict        (from best_model.pth)

    Returns
    -------
    start_epoch : int
        The epoch to resume from (0 if raw state_dict).
    checkpoint_info : dict
        Any extra info stored in the checkpoint (loss, val_loss, etc.).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # --- Full checkpoint dict ------------------------------------------------
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint.get('epoch', 0)
        info = {
            'loss': checkpoint.get('loss', None),
            'val_loss': checkpoint.get('val_loss', None),
        }
        return start_epoch, info

    # --- Raw state_dict ------------------------------------------------------
    if isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(
            f"Unrecognised checkpoint format in {path}. "
            "Expected a state_dict or a checkpoint dict with 'model_state_dict'."
        )
    return 0, {}


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def continue_train(args):
    # 1. Load Config
    config_path = resolve_path(args.config, SCRIPT_DIR)
    train_cfg = load_config(config_path)
    config_dir = os.path.dirname(config_path)

    # 2. Setup Logging — default to the same file as train.py
    log_dir = resolve_path(train_cfg.get('Logging', 'LogDir'), config_dir)
    os.makedirs(log_dir, exist_ok=True)

    log_filename = args.log_file if args.log_file else 'train.log'
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='a'  # APPEND so we don't overwrite the previous log
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    logging.info("=" * 60)
    logging.info("Resuming training from checkpoint: %s", args.checkpoint)
    logging.info("Config: %s", config_path)
    logging.info("=" * 60)

    # 3. Device & Seed
    device_name = train_cfg.get('Device', 'Device', fallback=None)
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    seed = train_cfg.getint('Training', 'Seed')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if device.type == 'cuda' and device.index is not None:
            torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)

    # 4. Data Loaders
    dataset_dir = resolve_path(train_cfg.get('Data', 'DatasetDir'), config_dir)
    val_dir = resolve_path(train_cfg.get('Data', 'ValidationDir'), config_dir)
    batch_size = train_cfg.getint('Training', 'BatchSize')
    num_workers = train_cfg.getint('Data', 'NumWorkers')

    if not os.path.exists(dataset_dir):
        logging.error(f"Dataset directory not found: {dataset_dir}")
        return

    aug_cfg = parse_augmentation_config(config_path)
    if aug_cfg:
        logging.info(f"Data augmentation ENABLED (p={aug_cfg['probability']:.1f}, "
                     f"rot={aug_cfg['rotation_degrees']}°, "
                     f"trans={aug_cfg['translation_fraction']}, "
                     f"elastic={aug_cfg['elastic_deformation']})")
    else:
        logging.info("Data augmentation DISABLED")

    train_dataset = LIDCDataset(dataset_dir, test_flag=False, augmentation_cfg=aug_cfg)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )
    logging.info(f"Train Dataset loaded with {len(train_dataset)} samples.")

    eval_loader = None
    if os.path.exists(val_dir):
        val_dataset = LIDCDataset(val_dir, test_flag=False)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False
        )
        logging.info(f"Validation Dataset loaded with {len(val_dataset)} samples.")

        eval_dataset = LIDCDataset(val_dir, test_flag=True)
        eval_loader = DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=False
        )
    else:
        logging.warning(f"Validation directory not found: {val_dir}. Skipping validation.")
        val_loader = None

    # 5. Model
    model_config_path = os.path.join(SCRIPT_DIR, 'model_config.ini')
    model = build_ldseg_from_config(model_config_path)
    model = model.to(device)

    # 6. Noise Scheduler
    model_cfg = load_config(model_config_path)
    schedule_type = model_cfg.get('NoiseScheduler', 'Scheduler')
    timesteps = model_cfg.getint('NoiseScheduler', 'Timesteps')
    learn_sigma = model.denoiser.learn_sigma
    diffusion = create_gaussian_diffusion(steps=timesteps, noise_schedule=schedule_type, learn_sigma=learn_sigma)
    logging.info(f"Noise scheduler: {schedule_type}, steps: {timesteps}, learn_sigma: {learn_sigma}")

    # 7. Optimizer & Scheduler
    lr = train_cfg.getfloat('Optimizer', 'LearningRate')
    weight_decay = train_cfg.getfloat('Optimizer', 'WeightDecay')
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_lr_scheduler(optimizer, train_cfg, len(train_loader))

    use_amp = train_cfg.getboolean('Training', 'MixedPrecision')
    grad_clip_norm = train_cfg.getfloat('Training', 'GradClipNorm', fallback=1.0)

    # 8. Load checkpoint
    checkpoint_path = resolve_path(args.checkpoint, os.getcwd())
    start_epoch, ckpt_info = load_checkpoint(
        checkpoint_path, model, optimizer, scheduler, device=device
    )
    logging.info(f"Loaded checkpoint from epoch {start_epoch}")
    if ckpt_info.get('loss') is not None:
        logging.info(f"  Last train loss: {ckpt_info['loss']:.6f}")
    if ckpt_info.get('val_loss') is not None:
        logging.info(f"  Last val   loss: {ckpt_info['val_loss']:.6f}")

    # 9. Losses
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss()
    criterion_mse = nn.MSELoss()

    lambda_diff = train_cfg.getfloat('Losses', 'Lambda_Diffusion')
    lambda_kl = train_cfg.getfloat('Losses', 'Lambda_KL')
    lambda_ce = train_cfg.getfloat('Losses', 'Lambda_CE')
    gamma_dice = train_cfg.getfloat('Losses', 'Gamma_Dice')

    # 10. Training loop (continued)
    epochs = train_cfg.getint('Training', 'Epochs')
    save_interval = train_cfg.getint('Logging', 'SaveInterval')

    # Early Stopping
    early_stop_enable = train_cfg.getboolean('EarlyStopping', 'Enable')
    monitor_metric = train_cfg.get('EarlyStopping', 'Monitor')
    patience = train_cfg.getint('EarlyStopping', 'Patience')
    min_delta = train_cfg.getfloat('EarlyStopping', 'MinDelta')
    best_metric = ckpt_info.get('val_loss', None) or ckpt_info.get('loss', None) or float('inf')
    patience_counter = 0

    num_train_batches = len(train_loader)
    total_epochs = start_epoch + epochs

    logging.info(f"Continuing for {epochs} epochs (epoch {start_epoch + 1} → {total_epochs})")

    for epoch in range(start_epoch, total_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}")
        for i, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            t = torch.randint(0, timesteps, (images.shape[0],), device=device).long()

            with autocast(enabled=use_amp, dtype=torch.bfloat16):
                with torch.no_grad():
                    clean_encoded = model.label_encoder(masks)
                    noise = torch.randn_like(clean_encoded)
                    noisy_encoded = diffusion.q_sample(clean_encoded, t, noise=noise)

                output = model(images, masks, t, noisy_encoded=noisy_encoded)

                loss_ce = criterion_ce(output['decoded'], masks.squeeze(1).long())
                loss_dice = criterion_dice(output['decoded'], masks)
                loss_recon = lambda_ce * loss_ce + gamma_dice * loss_dice
                loss_diff = criterion_mse(output['denoiser_out'], noise)
                loss_kl = output['kl_div'].mean()

                loss_total = loss_recon + lambda_diff * loss_diff + lambda_kl * loss_kl

            # Backward
            loss_total.backward()

            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss_total.item()

            pbar.set_postfix({
                'Loss': f"{loss_total.item():.4f}",
                'Recon': f"{loss_recon.item():.4f}",
                'Diff': f"{loss_diff.item():.4f}",
                'KL': f"{loss_kl.item():.4f}"
            })

            logging.info(
                "Epoch %d/%d Batch %d/%d - Loss: %.6f (Recon: %.6f, Diff: %.6f, KL: %.6f)",
                epoch + 1, total_epochs, i + 1, num_train_batches,
                loss_total.item(), loss_recon.item(),
                loss_diff.item(), loss_kl.item()
            )

        avg_train_loss = epoch_loss / num_train_batches
        logging.info(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.6f}")

        # Validation
        avg_val_loss = float('inf')
        if val_loader:
            val_metrics = validate(
                model, val_loader, diffusion, timesteps,
                criterion_ce, criterion_dice, criterion_mse,
                train_cfg['Losses'], device
            )
            avg_val_loss = val_metrics['loss']
            logging.info(
                f"Epoch {epoch + 1} Val Loss: {avg_val_loss:.6f} "
                f"(Recon: {val_metrics['recon']:.4f}, "
                f"Diff: {val_metrics['diff']:.4f}, "
                f"KL: {val_metrics['kl']:.4f})"
            )
            print(f"Epoch {epoch + 1} Val Loss: {avg_val_loss:.6f}")

        # Ambiguous Segmentation Metrics (Every 10 epochs)
        if (epoch + 1) % 10 == 0 and eval_loader:
            try:
                from sampling import compute_metrics_for_dataloader
                print(f"Computing Ambiguous Segmentation Metrics for Epoch {epoch + 1}...")
                metrics = compute_metrics_for_dataloader(
                    model, diffusion, eval_loader, num_samples=4, device=device
                )
                logging.info(
                    f"Epoch {epoch + 1} Metrics: GED={metrics['GED']:.4f}, "
                    f"MaxDice={metrics['MaxDice']:.4f}, CI={metrics['CI']:.4f}, "
                    f"Sensitivity={metrics['Sensitivity']:.4f}, "
                    f"Agreement={metrics['Agreement']:.4f}, "
                    f"OldCI={metrics['OldCI']:.4f}"
                )
                print(
                    f"Metrics: GED={metrics['GED']:.4f}, "
                    f"MaxDice={metrics['MaxDice']:.4f}, CI={metrics['CI']:.4f}, "
                    f"Sensitivity={metrics['Sensitivity']:.4f}, "
                    f"Agreement={metrics['Agreement']:.4f}, "
                    f"OldCI={metrics['OldCI']:.4f}"
                )
            except Exception as e:
                logging.error(f"Failed to compute metrics: {e}")
                print(f"Failed to compute metrics: {e}")

        # Checkpointing
        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, ckpt_path)
            logging.info(f"Saved checkpoint to {ckpt_path}")

        # Early Stopping
        if early_stop_enable:
            current_metric = avg_val_loss if monitor_metric == 'val_loss' else avg_train_loss

            if current_metric < best_metric - min_delta:
                best_metric = current_metric
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
                logging.info(f"New best model saved with {monitor_metric}: {best_metric:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered after epoch {epoch + 1}.")
                    print(f"Early stopping triggered. Best {monitor_metric}: {best_metric:.6f}")
                    break

    logging.info("Training complete.")


if __name__ == "__main__":
    args = parse_args()
    continue_train(args)
