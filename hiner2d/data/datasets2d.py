"""Pixel-coordinate dataset for HINER2D.

Each item is ONE pixel: a normalized coordinate (i, j) in [0,1]^2 and the
per-band-normalized spectral signature (B,) at that pixel.

Masking is a dataset-level operation (this is the elegant part of the 2D view):
  - mask_mode="hard": the training set is only the ROI coordinates (water/cloud
    pixels are never sampled -> cost zero params/bits).
  - mask_mode="soft": all pixels are kept, but each carries a loss weight
    (1.0 inside the ROI, soft_weight outside).
  - mask_mode="none": every pixel, weight 1.0.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class HSIPixelDataSet(Dataset):
    def __init__(self, args):
        cube = np.load(args.data_path).astype("float32")  # (B, H, W)
        B, H, W = cube.shape
        self.B, self.H, self.W = B, H, W
        self.final_size = H * W

        # per-band min-max normalization over the whole cube -> each band in [0,1]
        mn = cube.min(axis=(1, 2), keepdims=True)
        mx = cube.max(axis=(1, 2), keepdims=True)
        denom = np.where((mx - mn) > 0, (mx - mn), 1.0)
        cube = (cube - mn) / denom
        self.band_min = mn.reshape(-1)
        self.band_max = mx.reshape(-1)

        # spectra per pixel: (H*W, B)
        spectra = cube.reshape(B, H * W).T
        self.spectra = torch.from_numpy(np.ascontiguousarray(spectra)).float()

        # normalized coordinate grid: (H*W, 2) in [0,1]
        ii = np.linspace(0.0, 1.0, H, dtype="float32")
        jj = np.linspace(0.0, 1.0, W, dtype="float32")
        gi, gj = np.meshgrid(ii, jj, indexing="ij")
        coords = np.stack([gi.reshape(-1), gj.reshape(-1)], axis=1)
        self.coords = torch.from_numpy(coords).float()

        self.pix_idx = torch.arange(H * W)

        # --- masking ---
        mode = getattr(args, "mask_mode", "none")
        self.weights = torch.ones(H * W)
        self.active = self.pix_idx
        self.roi = torch.ones(H * W, dtype=torch.bool)
        mask_path = getattr(args, "mask_path", "") or ""
        if mode != "none" and mask_path:
            mask = np.load(mask_path).astype(bool)
            assert mask.shape == (H, W), f"mask {mask.shape} != image {(H, W)}"
            mask_flat = torch.from_numpy(mask.reshape(-1))
            self.roi = mask_flat
            if mode == "hard":
                self.active = self.pix_idx[mask_flat]
            elif mode == "soft":
                sw = float(getattr(args, "soft_weight", 0.1))
                self.weights = torch.where(
                    mask_flat, torch.ones(H * W), torch.full((H * W,), sw)
                )
            else:
                raise ValueError(f"unknown mask_mode {mode}")
        self.mask_mode = mode

    def __len__(self):
        return len(self.active)

    def __getitem__(self, k):
        idx = self.active[k]
        return {
            "coord": self.coords[idx],
            "spectrum": self.spectra[idx],
            "idx": idx,
            "weight": self.weights[idx],
        }
