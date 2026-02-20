"""
Training script for LDSeg_mod.

Usage:
    python train.py --config train_config.ini

Features:
- Loads LIDC dataset via `dataloader.py`.
- Instantiates model from `model_config.ini`.
- Uses `guided_diffusion` noise schedule.
- Implements mixed precision training (AMP).
- Uses AdamW optimizer with configurable LR scheduler.
    L_total = lambda_recon * (L_CE + gamma * L_Dice) + lambda_diff * L_MSE + lambda_kl * L_KL
- Validation step with loss logging.
- Includes component-specific learning rates and EMA for UNet.
- Early stopping based on configured metric (e.g. val_loss).
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
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import copy

# Local imports
from dataloader import LIDCDataset, parse_augmentation_config
from LDSeg_mod import build_ldseg_from_config
from guided_diffusion.nn import update_ema

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
GUIDED_DIFFUSION_DIR = os.path.join(PARENT_DIR, 'guided_diffusion')
GUIDED_DIFFUSION_NESTED_DIR = os.path.join(GUIDED_DIFFUSION_DIR, 'guided_diffusion')

sys_path_candidates = [PARENT_DIR]
if os.path.isdir(GUIDED_DIFFUSION_NESTED_DIR):
    # Support repo layout like: guided_diffusion/guided_diffusion/script_util.py
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
    print(f"guided_diffusion dir exists: {os.path.isdir(GUIDED_DIFFUSION_DIR)}")
    print(f"nested guided_diffusion dir exists: {os.path.isdir(GUIDED_DIFFUSION_NESTED_DIR)}")
    print(f"sys.path candidates used: {sys_path_candidates}")
    print(f"ImportError details: {e}")
    sys.exit(1)

# --- Configuration Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train LDSeg model")
    parser.add_argument('--config', type=str, default='train_config.ini', help='Path to training config file')
    return parser.parse_args()

def resolve_path(path, base_dir):
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(base_dir, path))

def load_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

# --- Loss Functions ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, input, target):
        # input: (B, C, H, W) logits
        # target: (B, 1, H, W) integer labels
        
        # Apply softmax to logits
        input = torch.softmax(input, dim=1)
        
        # One-hot encode target
        num_classes = input.shape[1]
        target_one_hot = torch.eye(num_classes, device=input.device)[target.squeeze(1).long()]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        # Compute Dice for each class
        intersection = (input * target_one_hot).sum(dim=(2, 3))
        union = input.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# --- Training Logic ---
def get_lr_scheduler(optimizer, config, steps_per_epoch):
    scheduler_type = config.get('Optimizer', 'LRScheduler')
    epochs = config.getint('Training', 'Epochs')
    warmup_epochs = config.getint('Optimizer', 'WarmupEpochs')
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    if scheduler_type == 'constant':
        return optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    elif scheduler_type == 'linear':
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type == 'cosine':
        # Cosine annealing with warmup, decaing to 0.1x of initial LR
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            return 0.1 + 0.9 * cosine_decay
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def validate(model, val_loader, diffusion, timesteps, criterion_ce, criterion_dice, criterion_mse, losses_cfg, device):
    """
    Validation loop.
    Returns average validation loss.
    """
    model.eval()
    val_loss_total = 0.0
    val_recon_total = 0.0
    val_diff_total = 0.0
    val_kl_total = 0.0
    val_vae_kl_total = 0.0
    
    lambda_diff = losses_cfg.getfloat('Lambda_Diffusion')
    lambda_kl = losses_cfg.getfloat('Lambda_KL')
    lambda_vae_kl = losses_cfg.getfloat('Lambda_VAE_KL', fallback=0.0)
    gamma_dice = losses_cfg.getfloat('Gamma_Dice')
    lambda_ce = losses_cfg.getfloat('Lambda_CE')
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Sample timesteps
            t = torch.randint(0, timesteps, (images.shape[0],), device=device).long()
            
            # Encode and add noise
            # Note: For testing/validation, we use the mode of the distribution (mu) as the clean encoded representation
            _, clean_encoded, _ = model.label_encoder(masks)
            noise = torch.randn_like(clean_encoded)
            noisy_encoded = diffusion.q_sample(clean_encoded, t, noise=noise)
            
            # Forward pass
            output = model(images, masks, t, noisy_encoded=noisy_encoded)
            
            # Losses
            loss_ce = criterion_ce(output['decoded'], masks.squeeze(1).long())
            loss_dice = criterion_dice(output['decoded'], masks)
            loss_recon = lambda_ce * loss_ce + gamma_dice * loss_dice
            loss_diff = criterion_mse(output['denoiser_out'], noise)
            loss_kl = output['kl_div'].mean()
            loss_vae_kl = output['vae_kl_div'].mean()
            
            loss_total = loss_recon + lambda_diff * loss_diff + lambda_kl * loss_kl + lambda_vae_kl * loss_vae_kl
            
            val_loss_total += loss_total.item()
            val_recon_total += loss_recon.item()
            val_diff_total += loss_diff.item()
            val_kl_total += loss_kl.item()
            val_vae_kl_total += loss_vae_kl.item()
            
    num_batches = len(val_loader)
    return {
        'loss': val_loss_total / num_batches,
        'recon': val_recon_total / num_batches,
        'diff': val_diff_total / num_batches,
        'kl': val_kl_total / num_batches,
        'vae_kl': val_vae_kl_total / num_batches
    }

def train(args):
    # 1. Load Config
    config_path = resolve_path(args.config, SCRIPT_DIR)
    train_cfg = load_config(config_path)
    config_dir = os.path.dirname(config_path)
    
    # 2. Setup Logging
    log_dir = resolve_path(train_cfg.get('Logging', 'LogDir'), config_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        filename=os.path.join(log_dir, 'train.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    logging.info("Starting training...")
    logging.info(f"Loaded config from {config_path}")
    
    # 3. Setup Device & Seed
    device_name = train_cfg.get('Device', 'Device', fallback=None)
    if device_name:
        device = torch.device(device_name)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    if device.type == 'cpu' and torch.cuda.is_available():
        logging.warning(
            "CUDA is available but training is set to CPU. "
            "Check [Device] Device in train_config.ini."
        )
    
    seed = train_cfg.getint('Training', 'Seed')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        if device.type == 'cuda':
            # Set specific device if index provided (e.g. cuda:1)
            if device.index is not None:
                torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    
    # 4. Data Loaders
    dataset_dir = resolve_path(train_cfg.get('Data', 'DatasetDir'), config_dir)
    val_dir = resolve_path(train_cfg.get('Data', 'ValidationDir'), config_dir)
    batch_size = train_cfg.getint('Training', 'BatchSize')
    num_workers = train_cfg.getint('Data', 'NumWorkers')
    
    # Check if dataset exists
    if not os.path.exists(dataset_dir):
        logging.error(f"Dataset directory not found: {dataset_dir}")
        return

    # Augmentation config
    aug_cfg = parse_augmentation_config(config_path)
    if aug_cfg:
        logging.info(f"Data augmentation ENABLED (p={aug_cfg['probability']:.1f}, "
                     f"rot={aug_cfg['rotation_degrees']}°, "
                     f"trans={aug_cfg['translation_fraction']}, "
                     f"elastic={aug_cfg['elastic_deformation']})")
    else:
        logging.info("Data augmentation DISABLED")

    # Train Dataset (with augmentation)
    train_dataset = LIDCDataset(dataset_dir, test_flag=False, augmentation_cfg=aug_cfg)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    logging.info(f"Train Dataset loaded with {len(train_dataset)} samples.")
    
    # Val Dataset (test_flag=False: returns single random label for loss computation)
    eval_loader = None
    if os.path.exists(val_dir):
        val_dataset = LIDCDataset(val_dir, test_flag=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        logging.info(f"Validation Dataset loaded with {len(val_dataset)} samples.")
        
        # Eval Dataset (test_flag=True: returns ALL 4 GT masks for metric computation)
        eval_dataset = LIDCDataset(val_dir, test_flag=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    else:
        logging.warning(f"Validation directory not found: {val_dir}. Skipping validation.")
        val_loader = None

    # 5. Model
    model_config_path = os.path.join(SCRIPT_DIR, 'model_config.ini')
    model = build_ldseg_from_config(model_config_path)
    model = model.to(device)
    
    # 6. Noise Scheduler
    # Read model config for schedule params
    model_cfg = load_config(model_config_path)
    schedule_type = model_cfg.get('NoiseScheduler', 'Scheduler')
    timesteps = model_cfg.getint('NoiseScheduler', 'Timesteps')
    diffusion = create_gaussian_diffusion(steps=timesteps, noise_schedule=schedule_type)
    logging.info(f"Noise scheduler: {schedule_type}, steps: {timesteps}")

    # 7. Optimizer & Scheduler
    unet_params = list(model.denoiser.parameters())
    vae_params = list(model.label_encoder.parameters()) + \
                 list(model.label_decoder.parameters()) + \
                 list(model.prior.parameters()) + \
                 list(model.posterior.parameters())
    cond_params = list(model.image_encoder.parameters())

    optimizer = optim.AdamW([
        {'params': unet_params, 'lr': train_cfg.getfloat('Optimizer', 'UNetLR'), 'weight_decay': train_cfg.getfloat('Optimizer', 'UNetWeightDecay')},
        {'params': vae_params, 'lr': train_cfg.getfloat('Optimizer', 'VaeLR'), 'weight_decay': train_cfg.getfloat('Optimizer', 'VaeWeightDecay')},
        {'params': cond_params, 'lr': train_cfg.getfloat('Optimizer', 'CondLR'), 'weight_decay': train_cfg.getfloat('Optimizer', 'CondWeightDecay')}
    ])
    
    scheduler = get_lr_scheduler(optimizer, train_cfg, len(train_loader))
    
    # GradScaler is only needed for float16, NOT bfloat16.
    # bfloat16 has the same exponent range as float32, so no loss scaling is required.
    use_amp = train_cfg.getboolean('Training', 'MixedPrecision')
    scaler = GradScaler(enabled=False)  # bfloat16 does not need GradScaler
    
    # Gradient clipping
    grad_clip_norm = train_cfg.getfloat('Training', 'GradClipNorm', fallback=1.0)
    
    # 8. Losses
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss()
    criterion_mse = nn.MSELoss()
    
    lambda_diff = train_cfg.getfloat('Losses', 'Lambda_Diffusion')
    lambda_kl = train_cfg.getfloat('Losses', 'Lambda_KL')
    lambda_vae_kl = train_cfg.getfloat('Losses', 'Lambda_VAE_KL', fallback=0.0)
    
    # 9. EMA Setup
    ema_enable = train_cfg.getboolean('EMA', 'Enable', fallback=False)
    ema_decay = train_cfg.getfloat('EMA', 'Decay', fallback=0.9999)
    if ema_enable:
        logging.info(f"UNet EMA enabled with decay {ema_decay}")
        ema_unet_params = [copy.deepcopy(p).detach() for p in model.denoiser.parameters()]
    else:
        ema_unet_params = None
    
    # 10. Training Loop
    epochs = train_cfg.getint('Training', 'Epochs')
    save_interval = train_cfg.getint('Logging', 'SaveInterval')
    
    # Early Stopping
    early_stop_enable = train_cfg.getboolean('EarlyStopping', 'Enable')
    monitor_metric = train_cfg.get('EarlyStopping', 'Monitor')
    patience = train_cfg.getint('EarlyStopping', 'Patience')
    min_delta = train_cfg.getfloat('EarlyStopping', 'MinDelta')
    best_metric = float('inf')
    patience_counter = 0

    num_train_batches = len(train_loader)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (images, masks) in enumerate(pbar):
            images = images.to(device) # (B, 1, H, W)
            masks = masks.to(device)   # (B, 1, H, W)
            
            optimizer.zero_grad()
            
            # Sample timesteps
            t = torch.randint(0, timesteps, (images.shape[0],), device=device).long()
            
            with autocast(enabled=use_amp, dtype=torch.bfloat16):
                # Manual noise injection step for scheduler usage:
                with torch.no_grad():
                    # For adding noise, we take the sampled z from the VAE LabelEncoder
                    clean_encoded, _, _ = model.label_encoder(masks)
                    noise = torch.randn_like(clean_encoded)
                    noisy_encoded = diffusion.q_sample(clean_encoded, t, noise=noise)
                
                # Forward pass
                output = model(images, masks, t, noisy_encoded=noisy_encoded)
                
                # Compute Losses
                loss_ce = criterion_ce(output['decoded'], masks.squeeze(1).long())
                loss_dice = criterion_dice(output['decoded'], masks)
                loss_recon = train_cfg.getfloat('Losses', 'Lambda_CE') * loss_ce + train_cfg.getfloat('Losses', 'Gamma_Dice') * loss_dice
                loss_diff = criterion_mse(output['denoiser_out'], noise)
                loss_kl = output['kl_div'].mean()
                loss_vae_kl = output['vae_kl_div'].mean()
                
                loss_total = loss_recon + lambda_diff * loss_diff + lambda_kl * loss_kl + lambda_vae_kl * loss_vae_kl
            
            # Backward
            loss_total.backward()
            
            # Gradient clipping to prevent NaN from exploding gradients
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            
            optimizer.step()
            scheduler.step()
            
            # Update EMA
            if ema_enable:
                update_ema(ema_unet_params, list(model.denoiser.parameters()), rate=ema_decay)
            
            epoch_loss += loss_total.item()
            
            pbar.set_postfix({
                'Loss': f"{loss_total.item():.4f}", 
                'Recon': f"{loss_recon.item():.4f}",
                'Diff': f"{loss_diff.item():.4f}",
                'KL': f"{loss_kl.item():.4f}",
                'VaeKL': f"{loss_vae_kl.item():.4f}"
            })

            logging.info(
                "Epoch %d/%d Batch %d/%d - Loss: %.6f (Recon: %.6f, Diff: %.6f, KL: %.6f, VaeKL: %.6f)",
                epoch + 1,
                epochs,
                i + 1,
                num_train_batches,
                loss_total.item(),
                loss_recon.item(),
                loss_diff.item(),
                loss_kl.item(),
                loss_vae_kl.item()
            )
            
        avg_train_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.6f}")
        
        # Validation
        avg_val_loss = float('inf')
        if val_loader:
            if ema_enable:
                # Swap original parameters with EMA parameters for validation
                orig_unet_params = [copy.deepcopy(p).detach() for p in model.denoiser.parameters()]
                for p, ema_p in zip(model.denoiser.parameters(), ema_unet_params):
                    p.data.copy_(ema_p.data)
            
            val_metrics = validate(model, val_loader, diffusion, timesteps, criterion_ce, criterion_dice, criterion_mse, train_cfg['Losses'], device)
            avg_val_loss = val_metrics['loss']
            logging.info(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.6f} (Recon: {val_metrics['recon']:.4f}, Diff: {val_metrics['diff']:.4f}, KL: {val_metrics['kl']:.4f}, VaeKL: {val_metrics['vae_kl']:.4f})")
            print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.6f}")
            
            if ema_enable:
                # Swap back to original parameters for training
                for p, orig_p in zip(model.denoiser.parameters(), orig_unet_params):
                    p.data.copy_(orig_p.data)
        
        # Ambiguous Segmentation Metrics (Every 10 epochs)
        if (epoch + 1) % 10 == 0 and eval_loader:
            try:
                from sampling import compute_metrics_for_dataloader
                print(f"Computing Ambiguous Segmentation Metrics for Epoch {epoch+1}...")
                metrics = compute_metrics_for_dataloader(model, diffusion, eval_loader, num_samples=4, device=device)
                logging.info(f"Epoch {epoch+1} Metrics: GED={metrics['GED']:.4f}, MaxDice={metrics['MaxDice']:.4f}, CI={metrics['CI']:.4f}, Sensitivity={metrics['Sensitivity']:.4f}, Agreement={metrics['Agreement']:.4f}")
                print(f"Metrics: GED={metrics['GED']:.4f}, MaxDice={metrics['MaxDice']:.4f}, CI={metrics['CI']:.4f}")
            except Exception as e:
                logging.error(f"Failed to compute metrics: {e}")
                print(f"Failed to compute metrics: {e}")

        
        # Checkpointing
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch+1}.pth")
            state_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_train_loss,
                'val_loss': avg_val_loss
            }
            if ema_enable:
                state_dict['ema_unet_params'] = [p.data.cpu() for p in ema_unet_params]
                
            torch.save(state_dict, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

        # Early Stopping
        if early_stop_enable:
            current_metric = avg_val_loss if monitor_metric == 'val_loss' else avg_train_loss
            
            if current_metric < best_metric - min_delta:
                best_metric = current_metric
                patience_counter = 0
                # Save best model
                state_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_train_loss,
                    'val_loss': avg_val_loss
                }
                if ema_enable:
                    state_dict['ema_unet_params'] = [p.data.cpu() for p in ema_unet_params]
                    
                torch.save(state_dict, os.path.join(log_dir, "best_model.pth"))
                logging.info(f"New best model saved with {monitor_metric}: {best_metric:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logging.info(f"Early stopping triggered after {epoch+1} epochs.")
                    print(f"Early stopping triggered. Best {monitor_metric}: {best_metric:.6f}")
                    break

if __name__ == "__main__":
    args = parse_args()
    train(args)
