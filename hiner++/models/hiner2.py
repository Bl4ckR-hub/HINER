import time
import numpy as np
from math import pi, sqrt, ceil
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from hnerv_utils import quant_tensor
from models.layers import Mlp, NeRVBlock, ModulateNeRVBlock, OutImg, PositionEncoding

class HINER2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = args.embed
        self.arch = args.arch
        ks_enc, ks_dec1, ks_dec2 = [int(x) for x in args.ks.split('_')]

        # BUILD Encoder LAYERS
        enc_dim1, enc_dim2 = [int(x) for x in args.enc_dim.split('_')]
        hnerv_hw = np.prod(args.enc_strds) // np.prod(args.dec_strds)
        self.fc_h, self.fc_w = hnerv_hw, hnerv_hw
        self.fc_h_arg, self.fc_w_arg = [int(x) for x in args.fc_hw.split('_')]
        ch_in = enc_dim2
        self.pe_embed = PositionEncoding(args.embed) 
        self.encoder = Mlp(int(self.pe_embed.embed_length), 2*int(self.pe_embed.embed_length), 16*self.fc_h_arg*self.fc_w_arg)

        # BUILD Decoder LAYERS  
        decoder_layers = []  
        # first part: position embedding for spectral index
        self.pe_embed_t = PositionEncoding(args.embed) 
        stem_t = nn.Sequential(
            nn.Conv2d(int(self.pe_embed_t.embed_length), 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 32, 1),
        )
        decoder_layers.append(stem_t)
        # second part: reconstruction module
              
        ngf = args.fc_dim
        out_f = int(ngf * self.fc_h * self.fc_w)
        decoder_layer1 = NeRVBlock(dec_block=False, conv_type='conv', ngf=ch_in, new_ngf=out_f, ks=0, strd=1, 
            bias=True, norm=args.norm, act=args.act)
        decoder_layers.append(decoder_layer1)
        for i, strd in enumerate(args.dec_strds):                         
            reduction = sqrt(strd) if args.reduce==-1 else args.reduce
            new_ngf = int(max(round(ngf / reduction), args.lower_width))
            cur_blk = ModulateNeRVBlock(dec_block=True, conv_type=args.conv_type[1], ngf=ngf, new_ngf=new_ngf, 
                                ks=min(ks_dec1+2*i, ks_dec2), strd=strd, bias=True, norm=args.norm, act=args.act)
            decoder_layers.append(cur_blk)
            ngf = new_ngf
        
        self.decoder = nn.ModuleList(decoder_layers)
        self.head_layer = nn.Conv2d(ngf, 3, 3, 1, 1) if 'video' in args.data_type else nn.Conv2d(ngf, 1, 3, 1, 1)
        self.out_bias = args.out_bias
        self.quant_bit = args.quant_embed_bit
        self.batch = args.batchSize
        self.max_channel = args.max_channel
        self.embed_dict = {}
        self.spectral_embed_dict = {}

    def forward(self, img, norm_id, img_idx, input_embed=None, spectraSR=-1, spatialSR=-1):
        
        if spectraSR == -1:
            img = self.pe_embed(norm_id[:,None]).float()
            img_embed = self.encoder(img).contiguous().view(img.shape[0],16,self.fc_h_arg,self.fc_w_arg)
            self.embed_dict[f'{img_idx.item()}'] = img_embed.detach()
        else:
            img_embed = linear_interpolate(self.embed_dict, img_idx, spectraSR, self.max_channel)
        
        if input_embed != None:
            _, img_embed = quant_tensor(img_embed, self.quant_bit)

        embed_list = [img_embed]
        dec_start = time.time()
        if spectraSR == -1:
            t_embed = self.pe_embed_t(norm_id[:, None]).float().view(norm_id.size(0), -1, 1, 1)
            t_embed = self.decoder[0](t_embed)
            self.spectral_embed_dict[f'{img_idx.item()}'] = t_embed.detach()
        else:
            t_embed = linear_interpolate(self.spectral_embed_dict, img_idx, spectraSR, self.max_channel)
            
        output = self.decoder[1](img_embed)
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        embed_list.append(output)
        for id, layer in enumerate(self.decoder[2:]):
            output = layer((output, t_embed)) 
            if spatialSR!=-1 and id==4:
                output = torch.nn.functional.interpolate(output, scale_factor=spatialSR, mode='bicubic')
            embed_list.append(output)

        img_out = OutImg(self.head_layer(output), self.out_bias)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start

        return img_out, embed_list, dec_time


class HINER2Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.fc_h, self.fc_w = [torch.tensor(x) for x in [model.fc_h, model.fc_w]]
        self.out_bias = model.out_bias
        self.decoder = model.decoder
        self.head_layer = model.head_layer

    def forward(self, img_embed):
        output = self.decoder[0](img_embed)
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        for layer in self.decoder[1:]:
            output = layer(output) 
        output = self.head_layer(output)

        return  OutImg(output, self.out_bias)


###################################  Tranform input for denoising or inpainting   ###################################
class TransformInput(nn.Module):
    def __init__(self, args):
        super(TransformInput, self).__init__()
        self.restore = args.restore
        if 'inpaint' in self.restore:
            self.inpaint_mode = self.restore.split('_')[-1]
        elif 'denoise' in self.restore:
            self.noise_std = float(self.restore.split('_')[-1])
        elif 'spatialSR' in self.restore:
            self.sr_ratio = int(self.restore.split('_')[-1])

    def forward(self, img):
        gt = img.clone()
        inpaint_mask = torch.ones_like(img).to(img.device)
        h,w = img.shape[-2:]
        if 'inpaint' in self.restore:
            if 'block' in self.inpaint_mode:
                inpaint_size = 15
                for ctr_x, ctr_y in [(1/2, 1/2), (1/4, 1/4), (1/4, 3/4), (3/4, 1/4), (3/4, 3/4)]:
                    ctr_x, ctr_y = int(ctr_x * h), int(ctr_y * w)
                    x0, x1 = max(0, ctr_x - inpaint_size), min(h, ctr_x + inpaint_size)
                    y0, y1 = max(0, ctr_y - inpaint_size), min(w, ctr_y + inpaint_size)
                    inpaint_mask[:, :, x0:x1, y0:y1] = 0
            elif 'hline' in self.inpaint_mode:     
                for ctr_x, width in [(0.1, 10), (0.2, 20), (0.4, 5), (0.5, 8), (0.7, 8), (0.9, 5)]:
                    ctr_x = int(ctr_x * h)
                    ctr_y = min(h, int(ctr_x + width))
                    inpaint_mask[:, :, ctr_x:ctr_y, :] = 0
            # elif 'text' in self.inpaint_mode:   
            #     inpaint_mask = 1 - create_text_mask(h, w).to(img.device)
            else:
                raise ValueError
            paint_img = (img * inpaint_mask).clamp(min=0,max=1)
            return paint_img, gt, inpaint_mask.detach()
        elif 'denoise' in self.restore:
            noise_img = gt + torch.randn_like(gt) * self.noise_std
            noise_img = torch.clamp(noise_img, 0.0, 1.0)
            return noise_img, gt, inpaint_mask.detach()
        elif 'spatialSR' in self.restore:
            # LR = torch.nn.functional.interpolate(img, scale_factor=1/self.sr_ratio, mode='bicubic').clamp_(0,1)
            LR = gaussian_blur(img, sigma=1.2)
            LR = F.interpolate(LR, scale_factor=1./self.sr_ratio, mode='bilinear', align_corners=False)
            LR = LR + torch.randn_like(LR) * 0.01
            LR = torch.clamp(LR, 0.0, 1.0)
            return LR, gt, inpaint_mask.detach()
        else:
            input, gt = img, img
            return input, gt, inpaint_mask.detach()

def gaussian_blur(x, sigma=1.5, kernel_size=9):
    # x: [B, C, H, W]
    B, C, H, W = x.shape
    coords = torch.arange(kernel_size).float() - kernel_size // 2
    grid = coords.view(1, -1) ** 2 + coords.view(-1, 1) ** 2
    kernel = torch.exp(-grid / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).to(x.device)
    kernel = kernel.repeat(C, 1, 1, 1)
    return F.conv2d(x, kernel, padding=kernel_size // 2, groups=C)

# def create_text_mask(H, W, text1="Inpainting", text2=" This is Pavia University", text3="Font is FREESCPT", font_size=30):
#     image = Image.new("L", (W, H), 0)
#     draw = ImageDraw.Draw(image)

#     try:
#         font = ImageFont.truetype("./IMPACT.TTF", font_size)
#         # font = ImageFont.truetype("./FREESCPT.TTF", font_size)
#     except IOError:
#         font = ImageFont.load_default(font_size)

#     bbox1 = draw.textbbox((0, 0), text1, font=font)
#     text1_width, text1_height = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
#     bbox2 = draw.textbbox((0, 0), text2, font=font)
#     text2_width, text2_height = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    
#     total_height = text1_height + text2_height
#     y_position = (H - total_height) // 2
    
#     draw.text(((W - text1_width) // 2, y_position), text1, fill=255, font=font)
#     draw.text(((W - text2_width) // 2, y_position + text1_height + 200), text2, fill=255, font=font)
#     draw.text(((W - text2_width) // 2, y_position + text1_height - 150), text3, fill=255, font=font)

#     mask = transforms.ToTensor()(image)
    
#     return mask


def linear_interpolate(embed_dict, img_idx, sr_ratio, max_channel):
    if sr_ratio == 2:
        pre = embed_dict[f'{img_idx.item()-1}']
        next = embed_dict[f'{img_idx.item()+1}'] if img_idx<max_channel else pre
        img_embed = (pre + next)/2
    
    elif sr_ratio == 3:
        if f'{img_idx.item()-1}' in embed_dict.keys():
            pre = embed_dict[f'{img_idx.item()-1}']
            next = embed_dict[f'{img_idx.item()+2}'] if img_idx<max_channel else pre
            img_embed = (pre*2 + next*1)/3
        else:
            pre = embed_dict[f'{img_idx.item()-2}']
            next = embed_dict[f'{img_idx.item()+1}'] if img_idx<max_channel else pre
            img_embed = (pre*1 + next*2)/3
    
    elif sr_ratio == 4:
        if f'{img_idx.item()-1}' in embed_dict.keys():
            pre = embed_dict[f'{img_idx.item()-1}']
            next = embed_dict[f'{img_idx.item()+3}'] if img_idx<max_channel else pre
            img_embed = (pre*3 + next*1)/4
        elif f'{img_idx.item()-2}' in embed_dict.keys():
            pre = embed_dict[f'{img_idx.item()-2}']
            next = embed_dict[f'{img_idx.item()+2}'] if img_idx<max_channel else pre
            img_embed = (pre*2 + next*2)/4
        else:
            pre = embed_dict[f'{img_idx.item()-3}']
            next = embed_dict[f'{img_idx.item()+1}'] if img_idx<max_channel else pre
            img_embed = (pre*1 + next*3)/4
    return img_embed
