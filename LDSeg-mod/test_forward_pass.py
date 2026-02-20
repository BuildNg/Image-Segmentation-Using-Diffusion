"""
Verification script for LDSeg_mod end-to-end forward pass with LIDC dataloader.

1. Creates a temporary dummy LIDC dataset structure (images + labels).
2. Loads data using the custom LIDCDataset.
3. Initializes LDSeg_mod from config.
4. Runs a forward pass.
5. Verifies shapes and cleanup.
"""
import os
import shutil
import numpy as np
import torch
from skimage import io
from dataloader import LIDCDataset
from LDSeg_mod import build_ldseg_from_config

def create_dummy_data(root_dir):
    # Create valid sample folder structure
    sample_dir = os.path.join(root_dir, 'sample_001')
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create 1 image and 4 labels (grayscale noise)
    # Shape 128x128
    img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    
    files = {
        'image_001.png': img,
        'label0_001.png': (img > 100).astype(np.uint8) * 255,
        'label1_001.png': (img > 120).astype(np.uint8) * 255,
        'label2_001.png': (img > 90).astype(np.uint8) * 255,
        'label3_001.png': (img > 110).astype(np.uint8) * 255,
    }
    
    for fname, data in files.items():
        io.imsave(os.path.join(sample_dir, fname), data, check_contrast=False)
    
    print(f"Created dummy sample in {sample_dir}")

def test_forward_pass():
    temp_dir = 'temp_lidc_test_data'
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        # 1. Setup Data
        create_dummy_data(temp_dir)
        
        dataset = LIDCDataset(temp_dir, test_flag=False)
        print(f"Dataset length: {len(dataset)}")
        assert len(dataset) == 1, "Should have found 1 sample"
        
        # 2. Load Batch
        # Use simple manual batching since we only have 1 sample
        # Dataloader would add batch dim automatically
        img, mask = dataset[0]
        # Dataset returns: image (1, H, W), mask (1, H, W)
        print(f"Loaded item shapes: img={img.shape}, mask={mask.shape}")
        assert img.shape[0] == 4, f"Expected 4-channel image, got {img.shape[0]} channels"
        
        # Add batch dim -> (1, 1, H, W)
        img = img.unsqueeze(0)
        mask = mask.unsqueeze(0)
        
        # 3. Model & Config
        config_path = 'model_config.ini'
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(config_path)
        
        # Retrieve scheduler settings
        schedule_type = cfg.get('NoiseScheduler', 'Scheduler')
        timesteps = cfg.getint('NoiseScheduler', 'Timesteps')
        print(f"Using Noise Schedule: {schedule_type}, Steps: {timesteps}")

        # Setup Diffusion (requires adding parent dir to path for guided_diffusion imports)
        import sys
        if '..' not in sys.path:
            sys.path.append('..')
        
        # Import create_gaussian_diffusion from guided_diffusion
        try:
            from guided_diffusion.script_util import create_gaussian_diffusion
        except ImportError as e:
            import traceback
            traceback.print_exc()
            print(f"WARNING: Could not import guided_diffusion. Skipping noise schedule test. Error: {e}")
            diffusion = None
        else:
            diffusion = create_gaussian_diffusion(
                steps=timesteps,
                noise_schedule=schedule_type,
            )

        model = build_ldseg_from_config(config_path)
        model.eval()
        
        # 4. Forward Pass
        timestep = torch.tensor([10]) # Batch size 1
        
        with torch.no_grad():
            if diffusion:
                # Need latent encoded first to noise it
                # But LDSeg handles encoding internally.
                # To test q_sample, we should ideally encode manually OR trust LDSeg's fallback.
                # However, LDSeg forward takes `noisy_encoded`.
                # So we can encode manually here just for the test:
                clean_encoded, _, _ = model.label_encoder(mask)
                t_batch = torch.full((1,), 10,  device=clean_encoded.device, dtype=torch.long)
                
                # Sample noise and add via scheduler
                noise = torch.randn_like(clean_encoded)
                noisy_encoded = diffusion.q_sample(clean_encoded, t_batch, noise=noise)
                
                # Run forward pass with explicit noisy input
                output = model(img, mask, timestep, noisy_encoded=noisy_encoded)
                print("Forward pass WITH scheduler-generated noise successful.")
            else:
                # Fallback to internal noise generation
                output = model(img, mask, timestep)
                print("Forward pass WITHOUT scheduler (fallback) successful.")
        
        # 5. Verify Output Shapes
        for k, v in output.items():
            if hasattr(v, 'shape'):
                print(f"  {k}: {v.shape}")
            elif hasattr(v, 'batch_shape'):
                print(f"  {k}: dist batch={v.batch_shape} event={v.event_shape}")
        
        # Latent output check (downsample 8x) -> 128/8 = 16
        assert output['encoded'].shape == (1, 1, 16, 16)
        assert output['denoiser_out'].shape == (1, 1, 16, 16)
        
        print("ALL TESTS PASSED")

    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("Cleaned up temp data.")

if __name__ == "__main__":
    test_forward_pass()
