"""
Data loader for the LIDC dataset.

Replicates the logic from `guided_diffusion.lidcloader` but adapted as a standalone module
for use with LDSeg_mod.

Expected folder structure:
  root_dir/
    subfolder/ (optional)
      prefix_image.png
      prefix_label0.png
      prefix_label1.png
      prefix_label2.png
      prefix_label3.png

The loader:
  1. Finds groups of 5 files (image + 4 labels) based on prefixes.
  2. Reads them using skimage.io.imread.
  3. Scales by 1/255.
  4. Optionally applies data augmentation (geometric + intensity).
  5. Returns:
        - image: (4, H, W) tensor (grayscale image repeated 4×, matching the original lidcloader)
       - label: (1, H, W) tensor (randomly sampled from label0..3)
       - path:  (optional, if test_flag=True) path to image file
"""

import os
import random
import configparser
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io
from scipy.ndimage import (
    rotate as ndimage_rotate,
    shift as ndimage_shift,
    zoom as ndimage_zoom,
    gaussian_filter,
    map_coordinates,
)


# ========================================================================== #
#  Augmentation helpers                                                       #
# ========================================================================== #

def _elastic_deformation(image, label, alpha, sigma, rng):
    """Apply random elastic deformation to image and label jointly."""
    shape = image.shape
    # Random displacement fields
    dx = gaussian_filter(rng.standard_normal(shape), sigma) * alpha
    dy = gaussian_filter(rng.standard_normal(shape), sigma) * alpha

    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    coords = [np.reshape(y + dy, (-1,)), np.reshape(x + dx, (-1,))]

    image_out = map_coordinates(image, coords, order=1, mode='reflect').reshape(shape)
    label_out = map_coordinates(label, coords, order=0, mode='reflect').reshape(shape)
    return image_out, label_out


def _apply_augmentation(image, label, aug_cfg, rng):
    """
    Apply augmentation to a single image-label pair (both as numpy H×W arrays).

    Geometric transforms are applied to *both* image and label (nearest for label).
    Intensity transforms are applied to the *image only*.

    Args:
        image: np.ndarray (H, W)  float32  [0, 1]
        label: np.ndarray (H, W)  float32  [0, 1]
        aug_cfg: dict with augmentation parameters
        rng: numpy RandomState
    Returns:
        image, label (augmented)
    """
    # Bail out with probability (1 - p)
    if rng.random() > aug_cfg['probability']:
        return image, label

    # ---- Geometric transforms ---- #

    # Rotation
    angle = aug_cfg.get('rotation_degrees', 0)
    if angle > 0:
        theta = rng.uniform(-angle, angle)
        image = ndimage_rotate(image, theta, reshape=False, order=1, mode='reflect')
        label = ndimage_rotate(label, theta, reshape=False, order=0, mode='reflect')

    # Translation
    frac = aug_cfg.get('translation_fraction', 0.0)
    if frac > 0:
        h, w = image.shape
        shift_y = rng.uniform(-frac, frac) * h
        shift_x = rng.uniform(-frac, frac) * w
        image = ndimage_shift(image, [shift_y, shift_x], order=1, mode='reflect')
        label = ndimage_shift(label, [shift_y, shift_x], order=0, mode='reflect')

    # Scaling
    smin = aug_cfg.get('scale_min', 1.0)
    smax = aug_cfg.get('scale_max', 1.0)
    if smin != 1.0 or smax != 1.0:
        scale = rng.uniform(smin, smax)
        h, w = image.shape
        image_z = ndimage_zoom(image, scale, order=1, mode='reflect')
        label_z = ndimage_zoom(label, scale, order=0, mode='reflect')
        # Crop or pad back to original size
        image = _center_crop_or_pad(image_z, h, w)
        label = _center_crop_or_pad(label_z, h, w)

    # Horizontal flip
    if aug_cfg.get('horizontal_flip', False) and rng.random() > 0.5:
        image = np.flip(image, axis=1).copy()
        label = np.flip(label, axis=1).copy()

    # Vertical flip
    if aug_cfg.get('vertical_flip', False) and rng.random() > 0.5:
        image = np.flip(image, axis=0).copy()
        label = np.flip(label, axis=0).copy()

    # Elastic deformation
    if aug_cfg.get('elastic_deformation', False):
        alpha = aug_cfg.get('elastic_alpha', 50)
        sigma = aug_cfg.get('elastic_sigma', 5)
        image, label = _elastic_deformation(image, label, alpha, sigma, rng)

    # ---- Intensity transforms (image only) ---- #

    # Brightness
    br = aug_cfg.get('brightness_range', 0.0)
    if br > 0:
        image = image + rng.uniform(-br, br)

    # Contrast
    cr = aug_cfg.get('contrast_range', 0.0)
    if cr > 0:
        factor = 1.0 + rng.uniform(-cr, cr)
        mean = image.mean()
        image = (image - mean) * factor + mean

    # Gamma correction
    gmin = aug_cfg.get('gamma_min', 1.0)
    gmax = aug_cfg.get('gamma_max', 1.0)
    if gmin != 1.0 or gmax != 1.0:
        gamma = rng.uniform(gmin, gmax)
        image = np.clip(image, 0, None)
        image = np.power(image, gamma)

    # Gaussian noise
    noise_std = aug_cfg.get('gaussian_noise_std', 0.0)
    if noise_std > 0:
        image = image + rng.normal(0, noise_std, image.shape)

    # Gaussian blur
    blur_sigma = aug_cfg.get('gaussian_blur_sigma', 0.0)
    if blur_sigma > 0:
        s = rng.uniform(0.1, blur_sigma)
        image = gaussian_filter(image, sigma=s)

    # Final clip
    image = np.clip(image, 0.0, 1.0).astype(np.float32)
    label = np.clip(label, 0.0, 1.0).astype(np.float32)

    return image, label


def _center_crop_or_pad(arr, target_h, target_w):
    """Crop or zero-pad a 2-D array to (target_h, target_w)."""
    h, w = arr.shape
    out = np.zeros((target_h, target_w), dtype=arr.dtype)
    # Compute source and destination slices
    src_y0 = max((h - target_h) // 2, 0)
    src_x0 = max((w - target_w) // 2, 0)
    dst_y0 = max((target_h - h) // 2, 0)
    dst_x0 = max((target_w - w) // 2, 0)
    copy_h = min(h, target_h)
    copy_w = min(w, target_w)
    out[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = \
        arr[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]
    return out


def parse_augmentation_config(config_path):
    """
    Read an [Augmentation] section from a config file and return a dict.
    Returns None if augmentation is disabled or section is missing.
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    if not cfg.has_section('Augmentation'):
        return None

    enabled = cfg.getboolean('Augmentation', 'Enable', fallback=False)
    if not enabled:
        return None

    def _float(key, default=0.0):
        return cfg.getfloat('Augmentation', key, fallback=default)
    def _bool(key, default=False):
        return cfg.getboolean('Augmentation', key, fallback=default)

    return {
        'probability':          _float('Probability', 0.5),
        # Geometric
        'rotation_degrees':     _float('RotationDegrees', 0),
        'translation_fraction': _float('TranslationFraction', 0.0),
        'scale_min':            _float('ScaleMin', 1.0),
        'scale_max':            _float('ScaleMax', 1.0),
        'horizontal_flip':      _bool('HorizontalFlip', False),
        'vertical_flip':        _bool('VerticalFlip', False),
        'elastic_deformation':  _bool('ElasticDeformation', False),
        'elastic_alpha':        _float('ElasticAlpha', 50),
        'elastic_sigma':        _float('ElasticSigma', 5),
        # Intensity
        'brightness_range':     _float('BrightnessRange', 0.0),
        'contrast_range':       _float('ContrastRange', 0.0),
        'gamma_min':            _float('GammaMin', 1.0),
        'gamma_max':            _float('GammaMax', 1.0),
        'gaussian_noise_std':   _float('GaussianNoiseStd', 0.0),
        'gaussian_blur_sigma':  _float('GaussianBlurSigma', 0.0),
    }


# ========================================================================== #
#  Dataset                                                                    #
# ========================================================================== #

class LIDCDataset(Dataset):
    def __init__(self, directory, test_flag=True, augmentation_cfg=None):
        """
        Args:
            directory (str): Path to dataset root.
            test_flag (bool): If True (evaluation), returns (image, all_labels, path)
                              where all_labels is (4, 1, H, W) with all 4 annotator masks.
                              If False (training), returns (image, label) with a single
                              randomly sampled label.
            augmentation_cfg (dict | None): Augmentation config dict from
                ``parse_augmentation_config()``. Pass None to disable.
        """
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.test_flag = test_flag
        self.aug_cfg = augmentation_cfg
        self.rng = np.random.RandomState()
        
        # We expect 5 types of files for each sample
        self.seqtypes = ['image', 'label0', 'label1', 'label2', 'label3']
        self.seqtypes_set = set(self.seqtypes)
        
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we assume it's a data leaf directory
            if not dirs:
                files.sort()
                # The original loader assumes EACH FOLDER contains EXACTLY ONE SAMPLE (5 files).
                # Filenames MUST start with `image_...`, `label0_...`, etc.
                
                datapoint = dict()
                
                # Filter for relevant files (ignore hidden files etc)
                relevant_files = [f for f in files if not f.startswith('.')]
                
                if not relevant_files:
                    continue

                for f in relevant_files:
                    parts = f.split('_')
                    if len(parts) == 0: continue
                    seqtype = parts[0]
                    
                    if seqtype in self.seqtypes_set:
                        datapoint[seqtype] = os.path.join(root, f)
                
                # Verify we have all 5
                if set(datapoint.keys()) == self.seqtypes_set:
                    self.database.append(datapoint)

    def __getitem__(self, x):
        filedict = self.database[x]
        out = []
        
        # Read all 5 files
        for seqtype in self.seqtypes:
            path = filedict[seqtype]
            img = io.imread(path)
            img = img.astype(np.float32) / 255.0
            out.append(img)  # Keep as numpy for augmentation

        image_np = out[0]  # (H, W)

        if self.test_flag:
            # --- Evaluation mode: return ALL 4 ground truth masks ---
            # Replicate single grayscale channel 4× to match guided_diffusion/lidcloader.py
            image_1ch = torch.from_numpy(image_np).unsqueeze(0)  # (1, H, W)
            image = torch.cat((image_1ch, image_1ch, image_1ch, image_1ch), dim=0)  # (4, H, W)
            
            all_labels = []
            for i in range(1, 5):
                lbl = torch.from_numpy(out[i]).unsqueeze(0)  # (1, H, W)
                all_labels.append(lbl)
            all_labels = torch.stack(all_labels, dim=0)  # (4, 1, H, W)
            
            return image, all_labels, filedict['image']
        else:
            # --- Training mode: return single random label ---
            label_idx = random.randint(1, 4)
            label_np = out[label_idx]  # (H, W)

            # Apply augmentation
            if self.aug_cfg is not None:
                image_np, label_np = _apply_augmentation(
                    image_np, label_np, self.aug_cfg, self.rng
                )

            image = torch.from_numpy(image_np).unsqueeze(0)  # (1, H, W)
            label = torch.from_numpy(label_np).unsqueeze(0)  # (1, H, W)

            # Replicate single grayscale channel 4× to match guided_diffusion/lidcloader.py
            image = torch.cat((image, image, image, image), dim=0)  # (4, H, W)
            
            return image, label

    def __len__(self):
        return len(self.database)

