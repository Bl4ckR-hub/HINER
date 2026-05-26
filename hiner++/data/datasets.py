import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.functional import interpolate
from torchvision.transforms.functional import center_crop, resize

import numpy as np
import scipy.io as sio
from skimage import io

# HyperSpectral Image dataset
class HSIDataSet(Dataset):
    def __init__(self, args):
        
        if args.data_path.endswith('mat'):
            hsi = sio.loadmat(args.data_path)
            key = list(hsi.keys())
            self.hsi = torch.tensor(hsi[key[-1]].astype('float32'))
        elif args.data_path.endswith('tif'):
            hsi = io.imread(args.data_path)
            self.hsi = torch.tensor(hsi.astype('float32')).permute(1,2,0)
        else:
            raise ValueError()        
        
        self.data_norm = args.data_norm
        # Resize the input video and center crop
        self.crop_list, self.resize_list = args.crop_list, args.resize_list    
        first_frame = self.img_transform(self.img_load(0))
        self.final_size = first_frame.size(-2) * first_frame.size(-1)
        self.newid = torch.randperm(self.hsi.shape[2])
        
        
    def img_load(self, idx):
        img = self.hsi[:,:,idx].unsqueeze(2).permute(-1,0,1)
        if self.data_norm == -1:
            img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
        else:
            img = img / float(self.data_norm)
        return img
    
    def img_transform(self, img):
        if self.crop_list != '-1': 
            crop_h, crop_w = [int(x) for x in self.crop_list.split('_')[:2]]
            if 'last' not in self.crop_list:
                img = pad(img, crop_h, crop_w)
        if self.resize_list != '-1':
            if '_' in self.resize_list:
                resize_h, resize_w = [int(x) for x in self.resize_list.split('_')]
                img = interpolate(img, (resize_h, resize_w), 'bicubic')
            else:
                resize_hw = int(self.resize_list)
                img = resize(img, resize_hw,  'bicubic')
        if 'last' in self.crop_list:
            img = center_crop(img, (crop_h, crop_w))
        return img
    
    def __len__(self):
        return self.hsi.shape[2]
    
    def __getitem__(self, idx):
        # idx = self.newid[idx]
        tensor_image = self.img_transform(self.img_load(idx))
        # print("---------- shuffle")
        norm_idx = float(idx) / self.hsi.shape[2]
        sample = {'img': tensor_image, 'idx': idx, 'norm_idx': norm_idx}
        return sample
    

# HyperSpectral Noise Image dataset
class NoisedHSIDataSet(Dataset):
    def __init__(self, args):
        
        if args.clean_data_path.endswith('mat'):
            hsi = sio.loadmat(args.clean_data_path)
            key = list(hsi.keys())
            self.clean_hsi = torch.tensor(hsi[key[-1]].astype('float32'))
        elif args.clean_data_path.endswith('tif'):
            hsi = io.imread(args.clean_data_path)
            self.clean_hsi = torch.tensor(hsi.astype('float32')).permute(1,2,0)
        else:
            raise ValueError()   
        
        if args.noise_data_path.endswith('mat'):
            hsi = sio.loadmat(args.noise_data_path)
            key = list(hsi.keys())
            self.noise_hsi = torch.tensor(hsi[key[-1]].astype('float32'))
        elif args.noise_data_path.endswith('tif'):
            hsi = io.imread(args.noise_data_path)
            self.noise_hsi = torch.tensor(hsi.astype('float32')).permute(1,2,0)
        else:
            raise ValueError()        
        
        self.data_norm = args.data_norm
        # Resize the input video and center crop
        self.crop_list, self.resize_list = args.crop_list, args.resize_list    
        clean_img, noise_img = self.img_load(0)
        _, first_frame = self.img_transform(clean_img, noise_img)
        self.final_size = first_frame.size(-2) * first_frame.size(-1)
        self.newid = torch.randperm(self.noise_hsi.shape[2])
        
        
    def img_load(self, idx):
        clean_img = self.clean_hsi[:,:,idx].unsqueeze(2).permute(-1,0,1)
        noise_img = self.noise_hsi[:,:,idx].unsqueeze(2).permute(-1,0,1)
        if self.data_norm == -1:
            clean_img = (clean_img - torch.min(clean_img)) / (torch.max(clean_img) - torch.min(clean_img))
            noise_img = (noise_img - torch.min(noise_img)) / (torch.max(noise_img) - torch.min(noise_img))
        else:
            '''
                Hyperspectral Image Denoising with Realistic Data
            '''
            clean_img = clean_img / float(self.data_norm)
            noise_img = noise_img / float(self.data_norm) * 50
        return clean_img, noise_img
    
    def img_transform(self, clean_img, noise_img):
        if self.crop_list != '-1': 
            crop_h, crop_w = [int(x) for x in self.crop_list.split('_')[:2]]
            if 'last' not in self.crop_list:
                clean_img = pad(clean_img, crop_h, crop_w)
                noise_img = pad(noise_img, crop_h, crop_w)
        if self.resize_list != '-1':
            if '_' in self.resize_list:
                resize_h, resize_w = [int(x) for x in self.resize_list.split('_')]
                clean_img = interpolate(clean_img, (resize_h, resize_w), 'bicubic')
                noise_img = interpolate(noise_img, (resize_h, resize_w), 'bicubic')
            else:
                resize_hw = int(self.resize_list)
                clean_img = resize(clean_img, resize_hw,  'bicubic')
                noise_img = resize(noise_img, resize_hw,  'bicubic')
        if 'last' in self.crop_list:
            clean_img = center_crop(clean_img, (crop_h, crop_w))
            noise_img = center_crop(noise_img, (crop_h, crop_w))
        return clean_img, noise_img
    
    def __len__(self):
        return self.noise_hsi.shape[2]
    
    def __getitem__(self, idx):
        # idx = self.newid[idx]
        clean_img, noise_img = self.img_load(idx)
        clean_img, noise_img = self.img_transform(clean_img, noise_img)
        # print("---------- shuffle")
        norm_idx = float(idx) / self.noise_hsi.shape[2]
        sample = {'clean_img': clean_img, 'noise_img': noise_img, 'idx': idx, 'norm_idx': norm_idx}
        return sample

def pad(x, H, W):
    h, w = x.size(1), x.size(2)
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )

def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )
