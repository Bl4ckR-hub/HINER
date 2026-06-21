import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.nn.functional import interpolate
from torchvision.transforms.functional import center_crop, resize


import os
import numpy as np
import scipy.io as sio


# HyperSpectral Image dataset
class HSIDataSet(Dataset):
    def __init__(self, args):

        self.hsi = np.load(args.data_path).astype("float32")  # Assumes (B, H, W)
        first_frame = self.img_load(0)
        self.final_size = first_frame.size(-2) * first_frame.size(-1)
        self.newid = torch.randperm(
            self.hsi.shape[0]
        )  # internally does dataset shuffling

    def img_load(self, idx):
        img = self.hsi[idx][None, :, :]  # numpy (1, H, W)

        # normalization
        data_norm = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = torch.tensor(data_norm)
        return img

    def __len__(self):
        return self.hsi.shape[0]  # returns the number of spectral bands

    def __getitem__(self, idx):
        idx1 = self.newid[idx]
        tensor_image = self.img_load(idx1)
        norm_idx = float(idx1) / self.hsi.shape[0]
        sample = {"img": tensor_image, "idx": idx1, "norm_idx": norm_idx}
        return sample
