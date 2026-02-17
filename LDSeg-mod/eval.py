"""
Evaluation script for LDSeg.

Usage:
    python eval.py --config eval.ini
"""

import os
import argparse
import configparser
import torch
from torch.utils.data import DataLoader
import sys

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# Ensure guided_diffusion can be imported from parent dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataloader import LIDCDataset
from LDSeg_mod import build_ldseg_from_config
from sampling import compute_metrics_for_dataloader
from guided_diffusion.script_util import create_gaussian_diffusion

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LDSeg model")
    parser.add_argument('--config', type=str, default='eval.ini', help='Path to evaluation config file')
    return parser.parse_args()

def load_config(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def main():
    args = parse_args()
    eval_cfg = load_config(args.config)
    
    # 1. Setup Device
    device_name = eval_cfg.get('Device', 'Device', fallback='cpu')
    device = torch.device(device_name)
    print(f"Using device: {device}")
    
    # 2. Build Model
    model_config_path = eval_cfg.get('Model', 'ModelConfigPath')
    print(f"Building model from {model_config_path}...")
    try:
        model = build_ldseg_from_config(model_config_path)
    except Exception as e:
        print(f"Error building model: {e}")
        return

    # 3. Load Checkpoint
    checkpoint_path = eval_cfg.get('Model', 'CheckpointPath')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle cases where checkpoint might be nested (e.g. {'model_state_dict': ...})
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Evaluating untrained model.")
        
    model = model.to(device)
    model.eval()
    
    # 4. Setup Diffusion
    # Need to get timesteps from model config if possible
    model_cfg = load_config(model_config_path)
    diffusion_steps = model_cfg.getint('NoiseScheduler', 'Timesteps', fallback=1000)
    noise_schedule = model_cfg.get('NoiseScheduler', 'Scheduler', fallback='linear')
    
    print(f"Creating diffusion process (Steps: {diffusion_steps}, Schedule: {noise_schedule})...")
    diffusion = create_gaussian_diffusion(steps=diffusion_steps, noise_schedule=noise_schedule)
    
    # 5. Setup DataLoader
    data_dir = eval_cfg.get('Data', 'DatasetDir')
    img_size = eval_cfg.getint('Data', 'ImageSize')
    batch_size = eval_cfg.getint('Data', 'BatchSize')
    num_workers = eval_cfg.getint('Data', 'NumWorkers')
    
    print(f"Loading dataset from {data_dir}...")
    dataset = LIDCDataset(data_dir, img_size) # Assuming same params
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # 6. Compute Metrics
    num_samples = eval_cfg.getint('Evaluation', 'NumSamples')
    print(f"Starting evaluation (Samples per image: {num_samples})...")
    
    metrics = compute_metrics_for_dataloader(
        model, 
        diffusion, 
        dataloader, 
        num_samples=num_samples, 
        device=device
    )
    
    # 7. Print Results
    print("\n--- Evaluation Results ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    # Optional: Save results to a file
    results_path = os.path.join(os.path.dirname(checkpoint_path), 'eval_results.txt') if os.path.dirname(checkpoint_path) else 'eval_results.txt'
    with open(results_path, 'w') as f:
        f.write("Evaluation Results:\n")
        for name, value in metrics.items():
            f.write(f"{name}: {value:.4f}\n")
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
