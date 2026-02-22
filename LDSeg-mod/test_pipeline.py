import unittest
import torch
import os
import sys
import shutil
import configparser
from torch.utils.data import DataLoader, Dataset
import tempfile

# Ensure local imports work and parent directory is in path for guided_diffusion
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from LDSeg_mod import build_ldseg_from_config
from sampling import compute_metrics_for_dataloader
from guided_diffusion.script_util import create_gaussian_diffusion

class DummyDataset(Dataset):
    def __init__(self, length=4):
        self.length = length
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Image: (3, 128, 128) - Assuming RGB for now, or 1 channel? 
        # Train config says ImageSize=128. Dataloader usually returns (B, C, H, W).
        # LIDC is usually grayscale but might be duplicated to 3 channels or kept 1.
        # Let's check train.py: "images.shape[0], 4, H, W" comment in train loop line 273!
        # Wait, line 273 says: images = images.to(device) # (B, 4, H, W) ??
        # Let's assume 4 channels based on comment, or maybe previous frames?
        # Re-reading train.py line 273: "images = images.to(device) # (B, 4, H, W)"
        # This is quite specific. Maybe 4 slices?
        # Let's produce (4, 128, 128) random images.
        image = torch.randn(4, 128, 128)  # 4-channel (grayscale repeated 4x)
        
        # all_masks: (4, 1, 128, 128) — 4 annotator masks, as expected by
        # compute_metrics_for_dataloader which unpacks (images, all_masks, _paths)
        all_masks = torch.randint(0, 2, (4, 1, 128, 128)).float()
        
        return image, all_masks, f"dummy_sample_{idx}"

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, 'test_model_config.ini')
        
        # Create a minimal model config
        config = configparser.ConfigParser()
        config['Model'] = {
            'InChannels': '1',
            'OutChannels': '1',
            'ModelChannels': '32',
            'NumResBlocks': '1',
            'ChannelMult': '1,2',
            'AttentionResolutions': '16', # minimal attention
            'NumHeads': '1',
            'UseScaleShiftNorm': 'True',
            'ResblockUpdown': 'True',
            'Dropout': '0.0'
        }
        config['Diffusion'] = {
            'ImageSize': '128',
            'NumChannels': '128', # Latent dim?
            'NumResBlocks': '1'
        }
        config['NoiseScheduler'] = {
            'Scheduler': 'linear',
            'Timesteps': '100' # Small but valid number for test
        }
        # Needed for AxisAlignedConvGaussian (Prior/Posterior)
        config['Prior'] = {'NumFilters': '32', 'NoConvsPerBlock': '1', 'LatentDim': '3'} 
        # The key names here depend on what build_ldseg_from_config expects. 
        # I might be guessing. Let's try to mock the build function or use existing 'model_config.ini' if available.
        # But 'model_config.ini' might be large.
        # Let's rely on 'model_config.ini' existing in CWD or mocking.
        # For robustness, I'll write the config file based on typical keys I saw in code view.
        # Actually, let's just use the existing 'model_config.ini' if it exists, otherwise write one.
        
        if os.path.exists('model_config.ini'):
             self.config_path = 'model_config.ini'
        else:
             # Fallback: assume standard keys
             with open(self.config_path, 'w') as f:
                 config.write(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_pipeline(self):
        print("\n--- Starting Pipeline Test ---")
        
        # 1. Initialize Model & Diffusion
        try:
            model = build_ldseg_from_config(self.config_path)
            model = model.to(self.device)
            print("Model initialized.")
        except Exception as e:
            self.fail(f"Failed to build model: {e}")
            
        diffusion = create_gaussian_diffusion(steps=100, noise_schedule='linear', learn_sigma=model.denoiser.learn_sigma)
        
        # 2. Setup Dummy Data
        dataset = DummyDataset(length=4)
        dataloader = DataLoader(dataset, batch_size=2)
        
        # 3. Test Training Step (One Batch)
        print("Testing Training Step...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion_ce = torch.nn.CrossEntropyLoss()
        criterion_mse = torch.nn.MSELoss()
        
        model.train()
        for images, all_masks, _paths in dataloader:
            images = images.to(self.device)
            # Use first annotator mask for training (same as real train loop)
            masks = all_masks[:, 0].to(self.device)  # (B, 1, H, W)
            
            optimizer.zero_grad()
            t = torch.randint(0, 100, (images.shape[0],), device=self.device).long()
            
            # Forward
            try:
                # Helper to encode masks similar to train.py
                clean_encoded, _, _ = model.label_encoder(masks)
                noise = torch.randn_like(clean_encoded)
                noisy_encoded = diffusion.q_sample(clean_encoded, t, noise=noise)
                
                output = model(images, masks, t, noisy_encoded=noisy_encoded)
                
                denoiser_out = output['denoiser_out']
                L = noise.shape[1]
                if denoiser_out.shape[1] == 2 * L:
                    eps_pred, _ = torch.split(denoiser_out, L, dim=1)
                else:
                    eps_pred = denoiser_out
                loss = criterion_mse(eps_pred, noise) + output['kl_div'].mean() + output['vae_kl_div'].mean()
                loss.backward()
                optimizer.step()
                print(f"Train step successful. Loss: {loss.item()}")
            except Exception as e:
                self.fail(f"Training step failed: {e}")
            break # Only 1 step
            
        # 4. Test Validation/Metrics (One Epoch equivalent)
        print("Testing Metrics Computation...")
        try:
            # We use the same dataloader as 'val_loader'
            metrics = compute_metrics_for_dataloader(model, diffusion, dataloader, num_samples=2, device=self.device, cfg_scale=3.0)
            print("Metrics Computed:", metrics)
            
            self.assertIn('GED', metrics)
            self.assertIn('MaxDice', metrics)
            self.assertIn('CI', metrics)
            self.assertIn('Sensitivity', metrics)
            self.assertIn('Agreement', metrics)
            
            # Check values are float and reasonably bounded (0-1 usually, GED can be higher?)
            self.assertIsInstance(metrics['CI'], float)
            self.assertTrue(0.0 <= metrics['CI'] <= 1.0 + 1e-6) # Allow small epsilon
            
        except Exception as e:
            self.fail(f"Metrics computation failed: {e}")

if __name__ == '__main__':
    unittest.main()
