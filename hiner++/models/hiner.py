import time
import numpy as np
from math import pi, sqrt, ceil
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from hnerv_utils import quant_tensor
from models.layers import Mlp, NeRVBlock, OutImg, PositionEncoding, ConvNeXt


class HINER(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed = args.embed
        self.arch = args.arch
        ks_enc, ks_dec1, ks_dec2 = [int(x) for x in args.ks.split('_')]
        enc_blks, dec_blks = [int(x) for x in args.num_blks.split('_')]

        # BUILD Encoder LAYERS
        if 'hiner' in args.arch:
            enc_dim1, enc_dim2 = [int(x) for x in args.enc_dim.split('_')]
            hnerv_hw = np.prod(args.enc_strds) // np.prod(args.dec_strds)
            self.fc_h, self.fc_w = hnerv_hw, hnerv_hw
            self.fc_h_arg, self.fc_w_arg = [int(x) for x in args.fc_hw.split('_')]
            ch_in = enc_dim2
            self.pe_embed = PositionEncoding('pe_1.25_80') 
            self.encoder = Mlp(160, 320, 16*self.fc_h_arg*self.fc_w_arg)
        elif 'hnerv' in args.arch:        #HNeRV
            enc_dim1, enc_dim2 = [int(x) for x in args.enc_dim.split('_')]
            c_in_list, c_out_list = [enc_dim1] * len(args.enc_strds), [enc_dim1] * len(args.enc_strds)
            c_out_list[-1] = enc_dim2
            if args.conv_type[0] == 'convnext':
                self.encoder = ConvNeXt(stage_blocks=enc_blks, strds=args.enc_strds, dims=c_out_list,
                    in_chans=3 if 'video' in args.data_type else 1, drop_path_rate=0)
            else:
                c_in_list[0] = 3
                encoder_layers = []
                for c_in, c_out, strd in zip(c_in_list, c_out_list, args.enc_strds):
                    encoder_layers.append(NeRVBlock(dec_block=False, conv_type=args.conv_type[0], ngf=c_in,
                     new_ngf=c_out, ks=ks_enc, strd=strd, bias=True, norm=args.norm, act=args.act))
                self.encoder = nn.Sequential(*encoder_layers)
            hnerv_hw = np.prod(args.enc_strds) // np.prod(args.dec_strds)
            self.fc_h, self.fc_w = hnerv_hw, hnerv_hw
            ch_in = enc_dim2
        else:
            ch_in = 2 * int(args.embed.split('_')[-1])
            self.pe_embed = PositionEncoding(args.embed)  
            self.encoder = nn.Identity()
            self.fc_h, self.fc_w = [int(x) for x in args.fc_hw.split('_')]

        # BUILD Decoder LAYERS  
        decoder_layers = []        
        ngf = args.fc_dim
        out_f = int(ngf * self.fc_h * self.fc_w)
        decoder_layer1 = NeRVBlock(dec_block=False, conv_type='conv', ngf=ch_in, new_ngf=out_f, ks=0, strd=1, 
            bias=True, norm=args.norm, act=args.act)
        decoder_layers.append(decoder_layer1)
        for i, strd in enumerate(args.dec_strds):                         
            reduction = sqrt(strd) if args.reduce==-1 else args.reduce
            new_ngf = int(max(round(ngf / reduction), args.lower_width))
            for j in range(dec_blks):
                cur_blk = nn.Sequential(
                    NeRVBlock(dec_block=True, conv_type=args.conv_type[1], ngf=ngf, new_ngf=new_ngf, 
                        ks=min(ks_dec1+2*i, ks_dec2), strd=1 if j else strd, bias=True, norm=args.norm, act=args.act),
                )
                decoder_layers.append(cur_blk)
                ngf = new_ngf
        
        self.decoder = nn.ModuleList(decoder_layers)
        self.head_layer = nn.Conv2d(ngf, 3, 3, 1, 1) if 'video' in args.data_type else nn.Conv2d(ngf, 1, 3, 1, 1)
        self.out_bias = args.out_bias
        self.quant_bit = args.quant_embed_bit
        self.batch = args.batchSize
        self.max_channel = args.max_channel
        self.embed_dict = {}

    def forward(self, img, norm_id, img_idx, input_embed=None, spectraSR=-1, spatialSR=-1):
        
        if spectraSR == -1:
            if 'hnerv' in self.arch:
                img_embed = self.encoder(img)
            elif 'hiner' in self.arch:
                img = self.pe_embed(norm_id[:,None]).float()
                img_embed = self.encoder(img).contiguous().view(img.shape[0],16,self.fc_h_arg,self.fc_w_arg)
            else:
                img = self.pe_embed(norm_id[:,None]).view(norm_id[:,None].size(0), -1, 1, 1).float()
                img_embed = self.encoder(img)

            self.embed_dict[f'{img_idx.item()}'] = img_embed.detach()
        else:
            img_embed = linear_interpolate(self.embed_dict, img_idx, spectraSR, self.max_channel)
        
        
        if input_embed != None:
            _, img_embed = quant_tensor(img_embed, self.quant_bit)

        embed_list = [img_embed]
        dec_start = time.time()
        output = self.decoder[0](img_embed)
        n, c, h, w = output.shape
        output = output.view(n, -1, self.fc_h, self.fc_w, h, w).permute(0,1,4,2,5,3).reshape(n,-1,self.fc_h * h, self.fc_w * w)
        embed_list.append(output)
        for id, layer in enumerate(self.decoder[1:]):
            output = layer(output) 
            if spatialSR!=-1 and id==4:
                output = torch.nn.functional.interpolate(output, scale_factor=spatialSR, mode='bicubic')
            embed_list.append(output)

        img_out = OutImg(self.head_layer(output), self.out_bias)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start

        return img_out, embed_list, dec_time


class HINERDecoder(nn.Module):
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
