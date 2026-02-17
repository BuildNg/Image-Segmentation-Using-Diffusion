import os
import random

import numpy as np
import torch
from skimage import io

class LIDCDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=True):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag = test_flag
        self.seqtypes = ["image", "label0", "label1", "label2", "label3"]

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            dirs.sort()
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split("_")[0]
                    if seqtype in self.seqtypes_set:
                        datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)
        self.database.sort(key=lambda item: item["image"])

    def __getitem__(self, x):
        filedict = self.database[x]
        out = {}
        for seqtype in self.seqtypes:
            img = io.imread(filedict[seqtype])
            img = (img.astype(np.float32) / 255.0)
            out[seqtype] = torch.from_numpy(img)

        image = out["image"].unsqueeze(0)

        if self.test_flag:
            expert_masks = torch.stack(
                [out["label0"], out["label1"], out["label2"], out["label3"]],
                dim=0,
            )
            return (image, expert_masks, filedict["image"])

        label_key = f"label{random.randint(0, 3)}"
        label = out[label_key].unsqueeze(0)
        return (image, label)

    def __len__(self):
        return len(self.database)
