"""
Sampling utilities for LDSeg_mod.

Functions:
1. sample_segmentation: End-to-end sampling from input image to segmentation mask.
2. get_distribution_params: Forward pass to retrieve Prior/Posterior distributions.
3. segmentation_sample_loop: Wraps sampling loop for easy import.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

def sample_segmentation(model, diffusion, image, num_samples=1, device='cuda', use_ddim=False, latent_size=None):
    """
    Sample segmentation mask(s) for a given input image.

    Args:
        model (LDSeg): The trained LDSeg_mod model.
        diffusion (GaussianDiffusion): The diffusion process object.
        image (torch.Tensor): Input image tensor (B, C, H, W).
        num_samples (int): Number of samples per image in batch.
        device (str/torch.device): Device to run on.
        use_ddim (bool): Whether to use DDIM sampling (faster) or DDPM (default).
        latent_size (int or None): Spatial size of the latent. If None, reads
            from model.latent_size or defaults to H // 8.

    Returns:
        torch.Tensor: Predicted segmentation mask (B, 1, H, W) integer labels.
    """
    model.eval()
    image = image.to(device)
    B, C, H, W = image.shape
    
    # Determine latent spatial dimensions
    if latent_size is not None:
        H_lat = W_lat = latent_size
    elif hasattr(model, 'latent_size') and model.latent_size is not None:
        H_lat = W_lat = model.latent_size
    else:
        H_lat, W_lat = H // 8, W // 8  # default fallback
    
    with torch.no_grad():
        # 1. Encode Image to get conditioning embedding
        # Result: (B, LatentChannels, H_lat, W_lat)
        img_embedding = model.image_encoder(image)
        
        # 2. Define the denoising loop function
        # The diffusion loop expects a model function: (x_t, t, **kwargs) -> output
        # We pass img_embedding as context via model_kwargs.
        def model_fn(x, t, context=None, **kwargs):
            # x: (B, 1, H_lat, W_lat) noisy latent
            # context: (B, ..., H_lat, W_lat) image embedding
            # t: (B,) timesteps
            return model.denoiser(x, context, t)
        
        # 3. Create noise and sample
        # If batch size > 1, we might want to sample multiple times per image?
        # For now, assume 1-to-1 mapping or broadcast img_embedding if num_samples > 1.
        # If num_samples > 1 and B=1, we replicate img_embedding.
        
        if num_samples > 1 and B == 1:
            img_embedding = img_embedding.repeat(num_samples, 1, 1, 1)
            B = num_samples
        
        shape = (B, 1, H_lat, W_lat) # Latent shape (1 channel)
        model_kwargs = {'context': img_embedding}
        
        # Sampling loop
        if use_ddim:
            sample_iter = diffusion.ddim_sample_loop_progressive(
                model_fn,
                shape,
                time=diffusion.num_timesteps, # Pass correct steps
                clip_denoised=False,
                model_kwargs=model_kwargs,
                device=device,
                progress=False
            )
        else:
            sample_iter = diffusion.p_sample_loop_progressive(
                model_fn,
                shape,
                time=diffusion.num_timesteps, # Pass correct steps
                clip_denoised=False,
                model_kwargs=model_kwargs,
                device=device,
                progress=False
            )
            
        # Run sampling (reverse diffusion) manually
        final = None
        for sample in sample_iter:
            final = sample
        final_latent = final['sample']
        
        # 4. Decode latent to segmentation logits
        decoded_logits = model.label_decoder(final_latent) # (B, NumClasses, H, W)
        
        # 5. Orgmax to get integer mask
        pred_mask = torch.argmax(decoded_logits, dim=1, keepdim=True) # (B, 1, H, W)
        
        return pred_mask


def get_distribution_params(model, image, mask, t, device='cuda'):
    """
    Run a full forward pass to get distribution parameters (Prior/Posterior).
    
    Args:
        model (LDSeg): Trained model.
        image (torch.Tensor): Input image (B, C, H, W).
        mask (torch.Tensor): Ground truth mask (B, 1, H, W).
        t (torch.Tensor): Timesteps (B,).
        device: Device.
        
    Returns:
        dict: Model output dictionary containing 'prior_dist', 'posterior_dist', etc.
    """
    model.eval()
    image = image.to(device)
    mask = mask.to(device)
    t = t.to(device)
    
    with torch.no_grad():
        output = model(image, mask, t)
        
    return output

def compute_metrics_for_dataloader(model, diffusion, dataloader, num_samples=16, device='cuda', use_ddim=False, latent_size=None):
    """
    Compute GED, Max Dice, and Collective Insight for a given dataloader.
    
    Args:
        model (LDSeg): Trained model.
        diffusion (GaussianDiffusion): Diffusion process.
        dataloader (DataLoader): Validation dataloader.
        num_samples (int): Number of samples per image to generate for metrics.
        device: Device.
        use_ddim (bool): Whether to use DDIM sampling (faster) or DDPM (default).
        
    Returns:
        dict: Aggregated metrics.
    """
    try:
        from .metrics import generalized_energy_distance, max_dice, collective_insight
    except ImportError:
        from metrics import generalized_energy_distance, max_dice, collective_insight

    
    model.eval()
    
    all_ged = []
    all_max_dice = []
    all_ci = []
    all_sc = []
    all_da = []
    
    for images, all_masks, _paths in tqdm(dataloader, desc="Computing Metrics"):
        # images: (B, C, H, W)
        # all_masks: (B, 4, 1, H, W) - All 4 annotator ground truth masks
        # _paths: list of str (not used here)
        
        images = images.to(device)
        all_masks = all_masks.to(device)
        B = images.shape[0]
        
        # Generate M samples for each image in batch
        # shape: (M, B, 1, H, W)
        preds = []
        for _ in range(num_samples):
            # sample_segmentation returns (B, 1, H, W)
            pred = sample_segmentation(model, diffusion, images, num_samples=1, device=device, use_ddim=use_ddim, latent_size=latent_size)
            preds.append(pred)
        
        preds = torch.stack(preds, dim=0) # (M, B, 1, H, W)
        
        # Ground truths: (B, 4, 1, H, W) -> (N, B, 1, H, W)  where N=4
        gts = all_masks.permute(1, 0, 2, 3, 4)  # (4, B, 1, H, W)
        
        # Binarize inputs for metrics
        preds_bin = (preds > 0).float()
        gts_bin = (gts > 0).float()
        
        # Compute metrics for this batch
        ged = generalized_energy_distance(preds_bin, gts_bin)
        md = max_dice(preds_bin, gts_bin)
        ci, sc, _, da = collective_insight(preds_bin, gts_bin)
        
        all_ged.append(ged.mean().item())
        all_max_dice.append(md.mean().item())
        all_ci.append(ci.mean().item())
        all_sc.append(sc.mean().item())
        all_da.append(da.mean().item())
        
    return {
        'GED': sum(all_ged) / len(all_ged),
        'MaxDice': sum(all_max_dice) / len(all_max_dice),
        'CI': sum(all_ci) / len(all_ci),
        'Sensitivity': sum(all_sc) / len(all_sc),
        'Agreement': sum(all_da) / len(all_da)
    }

