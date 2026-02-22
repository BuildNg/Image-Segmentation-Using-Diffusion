import unittest
import torch
import os
import sys
import tempfile
import configparser

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LDSeg_mod import build_ldseg_from_config
from sampling import sample_segmentation
from guided_diffusion.script_util import create_gaussian_diffusion

class TestVariancePrediction(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_config(self, learn_sigma=False):
        config = configparser.ConfigParser()
        config['Denoiser'] = {
            'LearnSigma': str(learn_sigma),
            'LatentChannels': '1',
            'CondChannels': '1',
            'FirstConvChannels': '8',
            'Widths': '8, 16',
            'HasAttention': 'False, False',
            'NumResBlocks': '1',
            'NormGroups': '2',
            'Interpolation': 'nearest',
            'Activation': 'swish'
        }
        config['LabelEncoder'] = {
            'InChannels': '1',
            'EncoderLayers': '1, 2, 4',
            'FilterNum': '8',
            'FilterSize': '3',
            'Dropout': '0.0',
            'UseBatchNorm': 'True',
            'Activation': 'swish'
        }
        config['LabelDecoder'] = {
            'InChannels': '1',
            'DecoderLayers': '2, 1',
            'FilterNum': '8',
            'FilterSize': '3',
            'Dropout': '0.0',
            'UseBatchNorm': 'True',
            'NumClasses': '2',
            'Activation': 'swish'
        }
        config['ImageEncoder'] = {
            'InChannels': '1',
            'FilterSize': '8',
            'KernelSize': '3',
            'Dropout': '0.0',
            'Groups': '2',
            'OutChannels': '1',
            'BlockMults': '1, 2',
            'AttentionAfter': '1',
            'Activation': 'swish'
        }
        config['Distribution'] = {
            'InputChannels': '1',
            'NumFilters': '8, 16',
            'NoConvsPerBlock': '1',
            'LatentDim': '3'
        }
        
        fd, path = tempfile.mkstemp(suffix='.ini')
        with os.fdopen(fd, 'w') as f:
            config.write(f)
        return path

    def test_learn_sigma_false(self):
        print("\nTesting LearnSigma = False...")
        cfg_path = self.create_config(learn_sigma=False)
        model = build_ldseg_from_config(cfg_path).to(self.device)
        model.eval()
        
        # Check Denoiser out channels
        # Out channels should be 1 (LatentChannels)
        x = torch.randn(2, 1, 16, 16, device=self.device) # noisy latent
        cond = torch.randn(2, 1, 16, 16, device=self.device)
        t = torch.tensor([10, 20], device=self.device)
        
        out = model.denoiser(x, cond, t)
        self.assertEqual(out.shape[1], 1, "Denoiser should output 1 channel when LearnSigma is False")
        os.remove(cfg_path)

    def test_learn_sigma_true(self):
        print("\nTesting LearnSigma = True...")
        cfg_path = self.create_config(learn_sigma=True)
        model = build_ldseg_from_config(cfg_path).to(self.device)
        model.eval()
        
        # Check Denoiser out channels
        x = torch.randn(2, 1, 16, 16, device=self.device)
        cond = torch.randn(2, 1, 16, 16, device=self.device)
        t = torch.tensor([10, 20], device=self.device)
        
        out = model.denoiser(x, cond, t)
        self.assertEqual(out.shape[1], 2, "Denoiser should output 2 channels when LearnSigma is True")
        
        # Test CFG compatibility
        diffusion = create_gaussian_diffusion(steps=50, noise_schedule='linear', learn_sigma=True)
        image = torch.randn(1, 1, 64, 64, device=self.device)
        try:
            pred_mask = sample_segmentation(model, diffusion, image, num_samples=1, device=self.device, latent_size=16, cfg_scale=2.0)
            self.assertEqual(pred_mask.shape, (1, 1, 64, 64))
            print("CFG Sampling with LearnSigma=True successful!")
        except Exception as e:
            self.fail(f"sample_segmentation failed with LearnSigma=True: {e}")
            
        # Test Training step / VLB Loss Computation
        from train import compute_explicit_vlb_loss
        model.train()
        masks = torch.randint(0, 2, (1, 1, 64, 64), device=self.device).float()
        clean_encoded, _, _ = model.label_encoder(masks)
        noise = torch.randn_like(clean_encoded)
        t_train = torch.tensor([10], device=self.device)
        noisy_encoded = diffusion.q_sample(clean_encoded, t_train, noise=noise)
        
        try:
            output = model(image, masks, t_train, noisy_encoded=noisy_encoded)
            denoiser_out = output['denoiser_out']
            eps_pred, _ = torch.split(denoiser_out, 1, dim=1)
            
            # This is precisely what's inside train.py
            loss_mse = torch.nn.MSELoss()(eps_pred, noise)
            loss_vlb = compute_explicit_vlb_loss(diffusion, clean_encoded.detach(), noisy_encoded.detach(), t_train, denoiser_out)
            
            # Should be valid tensor scalars
            self.assertFalse(torch.isnan(loss_vlb))
            print("VLB Loss Computation successful!", loss_vlb.item())
        except Exception as e:
            self.fail(f"Training loss computation failed: {e}")

        os.remove(cfg_path)

if __name__ == '__main__':
    unittest.main()
