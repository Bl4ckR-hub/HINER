import time
import torch
import torch.nn as nn
from math import pi, sqrt, ceil
import torch.nn.functional as F
import numpy as np
import decord
from hiner_utils import quant_tensor

decord.bridge.set_bridge("torch")


class Mlp(nn.Module):
    def __init__(
        self,
        in_features=160,
        hidden_features=256,
        out_features=128,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(int(in_features), int(hidden_features))
        self.act = act_layer()
        self.fc2 = nn.Linear(int(hidden_features), int(out_features))
        # self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        # x = self.drop(x)
        return x


class NeRVBlock(nn.Module):
    def __init__(self, **kargs):
        super().__init__()
        conv = UpConv if kargs["dec_block"] else DownConv
        self.conv = conv(
            ngf=kargs["ngf"],
            new_ngf=kargs["new_ngf"],
            strd=kargs["strd"],
            ks=kargs["ks"],
            conv_type=kargs["conv_type"],
            bias=kargs["bias"],
        )
        self.norm = NormLayer(kargs["norm"], kargs["new_ngf"])
        self.act = ActivationLayer(kargs["act"])

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


def Quantize_tensor(img_embed, quant_bit):
    out_min = img_embed.min(dim=1, keepdim=True)[0]
    out_max = img_embed.max(dim=1, keepdim=True)[0]
    scale = (out_max - out_min) / 2**quant_bit
    img_embed = ((img_embed - out_min) / scale).round()
    img_embed = out_min + scale * img_embed
    return img_embed


def OutImg(x, out_bias="tanh"):
    if out_bias == "sigmoid":
        return torch.sigmoid(x)
    elif out_bias == "tanh":
        return (torch.tanh(x) * 0.5) + 0.5
    else:
        return x + float(out_bias)


class HINER(nn.Module):
    def __init__(self, args):
        super().__init__()
        _, ks_dec1, ks_dec2 = [int(x) for x in args.ks.split("_")]
        _, dec_blks = [int(x) for x in args.num_blks.split("_")]

        # BUILD Encoder LAYERS
        _, enc_dim2 = [int(x) for x in args.enc_dim.split("_")]
        hiner_hw = np.prod(args.enc_strds) // np.prod(args.dec_strds)
        self.fc_h, self.fc_w = hiner_hw, hiner_hw
        self.fc_h_arg, self.fc_w_arg = [int(x) for x in args.fc_hw.split("_")]
        ch_in = enc_dim2
        self.pe_embed = PositionEncoding("pe_1.25_80")
        self.encoder = Mlp(160, 320, 16 * self.fc_h_arg * self.fc_w_arg)
        # BUILD Decoder LAYERS
        decoder_layers = []
        ngf = args.fc_dim
        out_f = int(ngf * self.fc_h * self.fc_w)
        decoder_layer1 = NeRVBlock(
            dec_block=False,
            conv_type="conv",
            ngf=ch_in,
            new_ngf=out_f,
            ks=0,
            strd=1,
            bias=True,
            norm=args.norm,
            act=args.act,
        )
        decoder_layers.append(decoder_layer1)
        for i, strd in enumerate(args.dec_strds):
            reduction = sqrt(strd) if args.reduce == -1 else args.reduce
            new_ngf = int(max(round(ngf / reduction), args.lower_width))
            for j in range(dec_blks):
                cur_blk = NeRVBlock(
                    dec_block=True,
                    conv_type="pshuffel",
                    ngf=ngf,
                    new_ngf=new_ngf,
                    ks=min(ks_dec1 + 2 * i, ks_dec2),
                    strd=1 if j else strd,
                    bias=True,
                    norm=args.norm,
                    act=args.act,
                )
                decoder_layers.append(cur_blk)
                ngf = new_ngf

        self.decoder = nn.ModuleList(decoder_layers)
        self.head_layer = nn.Conv2d(ngf, 1, 3, 1, 1)
        self.out_bias = args.out_bias
        self.quant_bit = args.quant_embed_bit

    def forward(self, img, norm_id, input_embed=None, encode_only=False):

        img = self.pe_embed(norm_id[:, None]).float()
        img_embed = (
            self.encoder(img).contiguous().view(1, 16, self.fc_h_arg, self.fc_w_arg)
        )

        if input_embed != None:
            _, img_embed = quant_tensor(img_embed, self.quant_bit)

        embed_list = [img_embed]
        dec_start = time.time()
        output = self.decoder[0](img_embed)
        n, _, h, w = output.shape
        output = (
            output.view(n, -1, self.fc_h, self.fc_w, h, w)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(n, -1, self.fc_h * h, self.fc_w * w)
        )
        embed_list.append(output)
        for layer in self.decoder[1:]:
            output = layer(output)
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
        n, _, h, w = output.shape
        output = (
            output.view(n, -1, self.fc_h, self.fc_w, h, w)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(n, -1, self.fc_h * h, self.fc_w * w)
        )
        for layer in self.decoder[1:]:
            output = layer(output)
        output = self.head_layer(output)

        return OutImg(output, self.out_bias)


###################################  Basic layers like position encoding/ downsample layers/ upscale blocks   ###################################
class PositionEncoding(nn.Module):
    def __init__(self, pe_embed):
        super(PositionEncoding, self).__init__()
        self.pe_embed = pe_embed
        if "pe" in pe_embed:
            lbase, levels = [float(x) for x in pe_embed.split("_")[-2:]]
            self.pe_bases = lbase ** torch.arange(int(levels)) * pi

    def forward(self, pos):
        if "pe" in self.pe_embed:
            value_list = pos * self.pe_bases.to(pos.device)
            pe_embed = torch.cat([torch.sin(value_list), torch.cos(value_list)], dim=-1)
            return pe_embed.view(pos.size(0), -1)
            # return pe_embed.view(pos.size(0), -1, 1, 1)
        else:
            return pos


class Sin(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Sin, self).__init__()

    def forward(self, input):
        return torch.sin(input)


def ActivationLayer(act_type):
    if act_type == "relu":
        act_layer = nn.ReLU(True)
    elif act_type == "leaky":
        act_layer = nn.LeakyReLU(inplace=True)
    elif act_type == "leaky01":
        act_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act_type == "relu6":
        act_layer = nn.ReLU6(inplace=True)
    elif act_type == "gelu":
        act_layer = nn.GELU()
    elif act_type == "sin":
        act_layer = Sin
    elif act_type == "swish":
        act_layer = nn.SiLU(inplace=True)
    elif act_type == "softplus":
        act_layer = nn.Softplus()
    elif act_type == "hardswish":
        act_layer = nn.Hardswish(inplace=True)
    else:
        raise KeyError(f"Unknown activation function {act_type}.")

    return act_layer


def NormLayer(norm_type, ch_width):
    if norm_type == "none":
        norm_layer = nn.Identity()
    elif norm_type == "bn":
        norm_layer = nn.BatchNorm2d(num_features=ch_width)
    elif norm_type == "in":
        norm_layer = nn.InstanceNorm2d(num_features=ch_width)
    else:
        raise NotImplementedError

    return norm_layer


class DownConv(nn.Module):
    def __init__(self, **kargs):
        super(DownConv, self).__init__()
        ks, ngf, new_ngf, strd = (
            kargs["ks"],
            kargs["ngf"],
            kargs["new_ngf"],
            kargs["strd"],
        )
        if kargs["conv_type"] == "pshuffel":
            self.downconv = nn.Sequential(
                nn.PixelUnshuffle(strd) if strd != 1 else nn.Identity(),
                nn.Conv2d(
                    ngf * strd**2,
                    new_ngf,
                    ks,
                    1,
                    ceil((ks - 1) // 2),
                    bias=kargs["bias"],
                ),
            )
        elif kargs["conv_type"] == "conv":
            self.downconv = nn.Conv2d(
                ngf, new_ngf, ks + strd, strd, ceil(ks / 2), bias=kargs["bias"]
            )
        elif kargs["conv_type"] == "interpolate":
            self.downconv = nn.Sequential(
                nn.Upsample(
                    scale_factor=1.0 / strd,
                    mode="bilinear",
                ),
                nn.Conv2d(
                    ngf,
                    new_ngf,
                    ks + strd,
                    1,
                    ceil((ks + strd - 1) / 2),
                    bias=kargs["bias"],
                ),
            )

    def forward(self, x):
        return self.downconv(x)


class UpConv(nn.Module):
    def __init__(self, **kargs):
        super(UpConv, self).__init__()
        ks, ngf, new_ngf, strd = (
            kargs["ks"],
            kargs["ngf"],
            kargs["new_ngf"],
            kargs["strd"],
        )
        if kargs["conv_type"] == "pshuffel":
            self.upconv = nn.Sequential(
                nn.Conv2d(
                    ngf,
                    new_ngf * strd * strd,
                    ks,
                    1,
                    ceil((ks - 1) // 2),
                    bias=kargs["bias"],
                ),
                nn.PixelShuffle(strd) if strd != 1 else nn.Identity(),
            )
        elif kargs["conv_type"] == "conv":
            self.upconv = nn.ConvTranspose2d(
                ngf, new_ngf, ks + strd, strd, ceil(ks / 2)
            )
        elif kargs["conv_type"] == "interpolate":
            self.upconv = nn.Sequential(
                nn.Upsample(
                    scale_factor=strd,
                    mode="bilinear",
                ),
                nn.Conv2d(
                    ngf,
                    new_ngf,
                    strd + ks,
                    1,
                    ceil((ks + strd - 1) / 2),
                    bias=kargs["bias"],
                ),
            )

    def forward(self, x):
        return self.upconv(x)


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
