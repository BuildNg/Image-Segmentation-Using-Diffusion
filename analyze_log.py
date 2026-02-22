"""
Log analysis script for training logs.

Parses train.log to extract:
1. Training loss (per batch and averaged per epoch)
2. Loss components: Recon, Diff, KL
3. Validation metrics: GED, MaxDice (Dmax), CI, Sensitivity, Agreement

Provides functions to:
- Extract all metrics as structured data
- Plot metrics over epochs
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


def parse_train_log(log_path: str = "train.log") -> Dict:
    """
    Parse the training log file and extract all metrics.
    
    Args:
        log_path: Path to the train.log file
        
    Returns:
        Dictionary containing:
        - 'train_loss': dict with epoch -> loss values
        - 'train_components': dict with epoch -> {'Recon': float, 'Diff': float, 'KL': float}
        - 'val_loss': dict with epoch -> loss values
        - 'val_components': dict with epoch -> {'Recon': float, 'Diff': float, 'KL': float}
        - 'metrics': dict with epoch -> {'GED': float, 'MaxDice': float, 'CI': float, 
                                          'Sensitivity': float, 'Agreement': float}
        - 'batch_data': list of (epoch, batch, loss, recon, diff, kl) tuples for detailed analysis
    """
    
    train_loss = {}  # epoch -> avg loss
    train_components = {}  # epoch -> {Recon, Diff, KL}
    val_loss = {}  # epoch -> val loss
    val_components = {}  # epoch -> {Recon, Diff, KL}
    metrics = {}  # epoch -> {GED, MaxDice, CI, Sensitivity, Agreement}
    batch_data = []  # All batch-level data
    
    # Regex patterns
    # Epoch 1/100 Batch 1/650 - Loss: 2.689331 (Recon: 1.428734, Diff: 1.256600, KL: 3.997421, VaeKL: 0.0001)
    batch_pattern = re.compile(
        r'Epoch (\d+)/\d+ Batch (\d+)/\d+ - Loss: ([\d.]+) '
        r'\(Recon: ([\d.]+), Diff: ([\d.]+), KL: ([\d.]+), VaeKL: ([\d.]+)\)'
    )
    
    # Epoch 1 Train Loss: 2.246044
    train_epoch_pattern = re.compile(r'Epoch (\d+) Train Loss: ([\d.]+)')
    
    # Epoch 1 Val Loss: 1.713988 (Recon: 1.2740, Diff: 0.4399, KL: 0.0180, VaeKL: 0.0001)
    val_pattern = re.compile(
        r'Epoch (\d+) Val Loss: ([\d.]+) '
        r'\(Recon: ([\d.]+), Diff: ([\d.]+), KL: ([\d.]+), VaeKL: ([\d.]+)\)'
    )
    
    # Epoch 4 Metrics: GED=1.8717, MaxDice=0.0177, CI=0.0248, Sensitivity=0.5660, Agreement=0.9322
    metrics_pattern = re.compile(
        r'Epoch (\d+) Metrics: '
        r'GED=([\d.]+), MaxDice=([\d.]+), CI=([\d.]+), '
        r'Sensitivity=([\d.]+), Agreement=([\d.]+)'
    )
    
    with open(log_path, 'r') as f:
        for line in f:
            # Parse batch data
            match = batch_pattern.search(line)
            if match:
                epoch, batch, loss, recon, diff, kl, vae_kl = match.groups()
                batch_data.append((
                    int(epoch), int(batch),
                    float(loss), float(recon), float(diff), float(kl), float(vae_kl)
                ))
                continue
            
            # Parse epoch training loss
            match = train_epoch_pattern.search(line)
            if match:
                epoch, loss = match.groups()
                train_loss[int(epoch)] = float(loss)
                continue
            
            # Parse validation loss
            match = val_pattern.search(line)
            if match:
                epoch, loss, recon, diff, kl, vae_kl = match.groups()
                val_loss[int(epoch)] = float(loss)
                val_components[int(epoch)] = {
                    'Recon': float(recon),
                    'Diff': float(diff),
                    'KL': float(kl),
                    'VaeKL': float(vae_kl)
                }
                continue
            
            # Parse metrics
            match = metrics_pattern.search(line)
            if match:
                epoch, ged, max_dice, ci, sensitivity, agreement = match.groups()
                metrics[int(epoch)] = {
                    'GED': float(ged),
                    'MaxDice': float(max_dice),
                    'CI': float(ci),
                    'Sensitivity': float(sensitivity),
                    'Agreement': float(agreement)
                }
                continue
    
    # Compute average training components per epoch from batch data
    for epoch in train_loss.keys():
        epoch_batches = [(recon, diff, kl, vae_kl) for e, b, l, recon, diff, kl, vae_kl in batch_data if e == epoch]
        if epoch_batches:
            avg_recon = np.mean([r for r, d, k, v in epoch_batches])
            avg_diff = np.mean([d for r, d, k, v in epoch_batches])
            avg_kl = np.mean([k for r, d, k, v in epoch_batches])
            avg_vae_kl = np.mean([v for r, d, k, v in epoch_batches])
            train_components[epoch] = {
                'Recon': avg_recon,
                'Diff': avg_diff,
                'KL': avg_kl,
                'VaeKL': avg_vae_kl
            }
    
    return {
        'train_loss': train_loss,
        'train_components': train_components,
        'val_loss': val_loss,
        'val_components': val_components,
        'metrics': metrics,
        'batch_data': batch_data
    }


def get_training_metrics(log_path: str = "train.log") -> Tuple[Dict, Dict, Dict, Dict, Dict]:
    """
    Extract and return all training metrics.
    
    Args:
        log_path: Path to the train.log file
        
    Returns:
        Tuple of (train_loss, train_components, val_loss, val_components, metrics)
    """
    data = parse_train_log(log_path)
    return (
        data['train_loss'],
        data['train_components'],
        data['val_loss'],
        data['val_components'],
        data['metrics']
    )


def plot_training_progress(log_path: str = "train.log", save_path: str = None):
    """
    Plot all training metrics over epochs.
    
    Creates 3 subplots:
    1. Training and validation loss
    2. Loss components (Recon, Diff, KL) for both train and val
    3. Metrics (GED, CI, MaxDice)
    
    Args:
        log_path: Path to the train.log file
        save_path: Optional path to save the figure
    """
    data = parse_train_log(log_path)
    
    train_loss = data['train_loss']
    train_comp = data['train_components']
    val_loss = data['val_loss']
    val_comp = data['val_components']
    metrics = data['metrics']
    
    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(3, 2)
    
    # --- Plot 1: Overall Loss ---
    ax = fig.add_subplot(gs[0, :])
    epochs_train = sorted(train_loss.keys())
    epochs_val = sorted(val_loss.keys())
    
    ax.plot(epochs_train, [train_loss[e] for e in epochs_train], 
            marker='o', label='Train Loss', linewidth=2)
    ax.plot(epochs_val, [val_loss[e] for e in epochs_val], 
            marker='s', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # --- Plot 2: Recon & Diff Components ---
    ax = fig.add_subplot(gs[1, 0])
    epochs_train = sorted(train_comp.keys())
    epochs_val = sorted(val_comp.keys())
    
    # Plot training components
    ax.plot(epochs_train, [train_comp[e]['Recon'] for e in epochs_train], 
            marker='o', label='Train Recon', linewidth=2, linestyle='-')
    ax.plot(epochs_train, [train_comp[e]['Diff'] for e in epochs_train], 
            marker='s', label='Train Diff', linewidth=2, linestyle='-')
    
    # Plot validation components
    ax.plot(epochs_val, [val_comp[e]['Recon'] for e in epochs_val], 
            marker='o', label='Val Recon', linewidth=2, linestyle='--', alpha=0.7)
    ax.plot(epochs_val, [val_comp[e]['Diff'] for e in epochs_val], 
            marker='s', label='Val Diff', linewidth=2, linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss Component Value', fontsize=12)
    ax.set_title('Reconstruction & Diffusion Losses', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # --- Plot 3: KL Components ---
    ax = fig.add_subplot(gs[1, 1])
    
    # Plot training components
    ax.plot(epochs_train, [train_comp[e]['KL'] for e in epochs_train], 
            marker='^', label='Train Diff KL', linewidth=2, linestyle='-')
    ax.plot(epochs_train, [train_comp[e]['VaeKL'] for e in epochs_train], 
            marker='d', label='Train VAE KL', linewidth=2, linestyle='-')
    
    # Plot validation components
    ax.plot(epochs_val, [val_comp[e]['KL'] for e in epochs_val], 
            marker='^', label='Val Diff KL', linewidth=2, linestyle='--', alpha=0.7)
    ax.plot(epochs_val, [val_comp[e]['VaeKL'] for e in epochs_val], 
            marker='d', label='Val VAE KL', linewidth=2, linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('KL Loss Value', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('KL Divergence Losses', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    # --- Plot 4: Metrics (GED, CI, MaxDice) ---
    ax = fig.add_subplot(gs[2, :])
    if metrics:
        epochs_metrics = sorted(metrics.keys())
        
        # Create twin axes for different scales
        ax2 = ax.twinx()
        
        # GED on left axis
        l1 = ax.plot(epochs_metrics, [metrics[e]['GED'] for e in epochs_metrics], 
                marker='o', label='GED', linewidth=2, color='tab:blue')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('GED', fontsize=12, color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        
        # CI and MaxDice on right axis
        l2 = ax2.plot(epochs_metrics, [metrics[e]['CI'] for e in epochs_metrics], 
                marker='s', label='CI', linewidth=2, color='tab:orange')
        l3 = ax2.plot(epochs_metrics, [metrics[e]['MaxDice'] for e in epochs_metrics], 
                marker='^', label='MaxDice (Dmax)', linewidth=2, color='tab:green')
        ax2.set_ylabel('CI / MaxDice', fontsize=12, color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        
        # Combine legends
        lines = l1 + l2 + l3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=10, loc='best')
        
        ax.set_title('Metrics (GED, CI, MaxDice)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No metrics data available', 
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('Metrics (GED, CI, MaxDice)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_batch_loss_history(log_path: str = "train.log", max_batches: int = None, save_path: str = None):
    """
    Plot batch-level loss progression (useful for diagnosing training dynamics).
    
    Args:
        log_path: Path to the train.log file
        max_batches: Optional limit on number of batches to plot (for large logs)
        save_path: Optional path to save the figure
    """
    data = parse_train_log(log_path)
    batch_data = data['batch_data']
    
    if max_batches:
        batch_data = batch_data[:max_batches]
    
    # Create global batch index
    batch_indices = list(range(len(batch_data)))
    losses = [loss for _, _, loss, _, _, _, _ in batch_data]
    
    plt.figure(figsize=(14, 6))
    plt.plot(batch_indices, losses, alpha=0.5, linewidth=0.5)
    
    # Add epoch boundaries
    epochs = [e for e, _, _, _, _, _, _ in batch_data]
    epoch_changes = [i for i in range(1, len(epochs)) if epochs[i] != epochs[i-1]]
    for idx in epoch_changes:
        plt.axvline(idx, color='red', alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.xlabel('Batch Index', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Batch-Level Loss Progression', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Parsing training log...")
    data = parse_train_log("train.log")
    
    print(f"\nFound {len(data['train_loss'])} training epochs")
    print(f"Found {len(data['val_loss'])} validation epochs")
    print(f"Found {len(data['metrics'])} metric evaluations")
    print(f"Total batches: {len(data['batch_data'])}")
    
    # Print latest metrics
    if data['val_loss']:
        latest_epoch = max(data['val_loss'].keys())
        print(f"\nLatest epoch ({latest_epoch}):")
        print(f"  Train Loss: {data['train_loss'].get(latest_epoch, 'N/A'):.4f}")
        print(f"  Val Loss: {data['val_loss'][latest_epoch]:.4f}")
        if latest_epoch in data['metrics']:
            m = data['metrics'][latest_epoch]
            print(f"  GED: {m['GED']:.4f}, MaxDice: {m['MaxDice']:.4f}, CI: {m['CI']:.4f}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_training_progress("train.log", save_path="training_progress.png")
