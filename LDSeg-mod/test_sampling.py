"""
Verification script for LDSeg_mod sampling utilities.

1. Initializes dummy model and diffusion process (short steps).
2. Runs sample_segmentation on random input.
3. Runs get_distribution_params on random input/mask.
4. Asserts correct output shapes/keys.
"""
import torch
import os
import sys

# Ensure guided_diffusion can be imported
if '..' not in sys.path:
    # If running from LDSeg-mod/, guided_diffusion is in parent
    sys.path.append('..')

try:
    from LDSeg_mod import build_ldseg_from_config
    from sampling import sample_segmentation, get_distribution_params
    from guided_diffusion.script_util import create_gaussian_diffusion
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def test_sampling():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing on {device}")
    
    # Init model
    if not os.path.exists('model_config.ini'):
        raise FileNotFoundError("model_config.ini not found")
        
    model = build_ldseg_from_config('model_config.ini')
    model = model.to(device)
    model.eval()
    
    # Init diffusion
    # Use small steps for speed in test
    diffusion = create_gaussian_diffusion(steps=10, noise_schedule='cosine')
    
    # Dummy Input
    B, C, H, W = 1, 4, 128, 128
    image = torch.randn(B, C, H, W).to(device)
    
    print("Testing sample_segmentation...")
    try:
        mask_pred = sample_segmentation(model, diffusion, image, num_samples=1, device=device)
        print(f"Output mask shape: {mask_pred.shape}")
        
        assert mask_pred.shape == (B, 1, H, W)
        print("sample_segmentation PASSED")
    except Exception as e:
        print(f"sample_segmentation FAILED: {e}")
        raise e
    
    print("Testing get_distribution_params...")
    try:
        mask_gt = torch.zeros(B, 1, H, W).to(device)
        t = torch.tensor([5]).to(device)
        out = get_distribution_params(model, image, mask_gt, t, device=device)
        
        assert 'prior_dist' in out
        assert 'posterior_dist' in out
        print("get_distribution_params PASSED")
    except Exception as e:
        print(f"get_distribution_params FAILED: {e}")
        raise e

if __name__ == "__main__":
    test_sampling()
