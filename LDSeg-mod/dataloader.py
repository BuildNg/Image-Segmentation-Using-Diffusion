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
  4. Returns:
       - image: (4, H, W) tensor (grayscale image repeated 4 times)
       - label: (1, H, W) tensor (randomly sampled from label0..3)
       - path:  (optional, if test_flag=True) path to image file
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage import io

class LIDCDataset(Dataset):
    def __init__(self, directory, test_flag=True):
        """
        Args:
            directory (str): Path to dataset root.
            test_flag (bool): If True, returns (image, label, path). 
                              If False, returns (image, label).
        """
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.test_flag = test_flag
        
        # We expect 5 types of files for each sample
        self.seqtypes = ['image', 'label0', 'label1', 'label2', 'label3']
        self.seqtypes_set = set(self.seqtypes)
        
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we assume it's a data leaf directory
            if not dirs:
                files.sort()
                # We need to group files by prefix. The original code assumes
                # "prefix_TYPE.ext" or similar where splitting by '_' gives the TYPE
                # at a specific position?
                # Actually, original code splits by '_': `seqtype = f.split('_')[0]`
                # This implies filenames like: `image_001.png`, `label0_001.png`...
                # Wait, original comment says: `brats_train_001_XXX_123_w.nii.gz` 
                # where XXX is one of t1, t2...
                # But logic `seqtype = f.split('_')[0]` implies the TYPE is the PREFIX.
                # Let's check the original code snippet again.
                # "brats_train_001_XXX..." -> split('_')[0] is "brats". That's not right.
                #
                # Re-reading original `lidcloader.py`:
                # `seqtype = f.split('_')[0]` 
                # `datapoint[seqtype] = ...`
                # `assert set(datapoint.keys()) == self.seqtypes_set`
                #
                # The seqtypes are ['image', 'label0', 'label1', 'label2', 'label3'].
                # So filenames MUST start with `image_...`, `label0_...`, etc.
                # Example: `image_123.png`, `label0_123.png`.
                
                # Check if this folder contains valid groups
                # We'll group by "everything after the first underscore"?
                # No, the original code creates ONE datapoint per folder!
                # "if not dirs: ... datapoint = dict() ... for f in files: ... self.database.append(datapoint)"
                # 
                # Wait, the original code puts all files in ONE datapoint dictionary?
                # `datapoint[seqtype] = ...`
                # If a folder has multiple images (e.g. image_01.png, image_02.png),
                # `seqtype = 'image'` for both. It would overwrite!
                #
                # IMPLICATION: The original loader assumes EACH FOLDER contains EXACTLY ONE SAMPLE (5 files).
                #
                # We will stick to this assumption to be compatible "as in lidcloader.py".
                
                datapoint = dict()
                valid_group = True
                
                # Filter for relevant files (ignore hidden files etc)
                relevant_files = [f for f in files if not f.startswith('.')]
                
                if not relevant_files:
                    continue

                for f in relevant_files:
                    # simplistic parsing matching original code:
                    # assumes filename starts with 'image', 'label0', etc.
                    parts = f.split('_')
                    if len(parts) == 0: continue
                    seqtype = parts[0]
                    
                    if seqtype in self.seqtypes_set:
                        datapoint[seqtype] = os.path.join(root, f)
                
                # Verify we have all 5
                if set(datapoint.keys()) == self.seqtypes_set:
                    self.database.append(datapoint)
                else:
                    # If folder doesn't match, we skip it (or warn?). 
                    # Original code asserts. We'll skip to be robust.
                    pass

    def __getitem__(self, x):
        filedict = self.database[x]
        out = []
        
        # Read all 5 files
        for seqtype in self.seqtypes:
            path = filedict[seqtype]
            # Use skimage.io.imread (matches original)
            img = io.imread(path)
            
            # Normalize to 0-1
            img = img.astype(np.float32) / 255.0
            
            # Handle grayscale inputs (H, W) -> (1, H, W)
            # Original code: `out.append(torch.tensor(img))`
            # If img is (H,W), tensor is (H,W).
            # stack(out) -> (5, H, W)
            out.append(torch.from_numpy(img))

        # Stack into (5, H, W) or (5, C, H, W) depending on image
        # Assuming grayscale images (H, W)
        out = torch.stack(out) 
        
        # Extract image: index 0
        image = out[0]  # (H, W)
        image = image.unsqueeze(0) # (1, H, W)
        # Concatenate 4 times to match LIDC config (4 channels)
        image = torch.cat([image, image, image, image], dim=0) # (4, H, W)
        
        # Extract label: indices 1-4 (randomly sample one)
        # Note: indices in `self.seqtypes` correspond to 0:image, 1:label0, 2:label1, 3:label2, 4:label3
        # Original code used `random.randint(1, 4)` on the stacked `out` tensor.
        label_idx = random.randint(1, 4)
        label = out[label_idx] # (H, W)
        label = label.unsqueeze(0) # (1, H, W)

        if self.test_flag:
            return image, label, filedict['image']
        else:
            return image, label

    def __len__(self):
        return len(self.database)
