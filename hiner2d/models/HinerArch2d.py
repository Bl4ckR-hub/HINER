"""HINER2D — a 2D-coordinate INR for hyperspectral cubes.

Where classical HINER (1D) maps a normalized band index -> a full band image
(2D conv + 2D pixelshuffle), HINER2D maps a normalized pixel coordinate (i, j)
-> that pixel's full spectral signature (1D conv + 1D pixelshuffle).

    coord (N, 2) in [0,1]^2
        -> 2D Fourier features (N, 4L)
        -> MLP                 (N, seed_ch * fc_len) -> reshape (N, seed_ch, fc_len)
        -> Conv1d 1x1          (N, fc_dim, fc_len)
        -> 1D pixelshuffle decoder, length fc_len -> B
        -> head Conv1d(.,1)    (N, 1, B) -> (N, B)   spectral signature

Output length contract (same spirit as classical HINER's fc_hw * prod(dec_strds)):
    fc_len * prod(dec_strds) == B (number of bands).
"""

import time
from math import pi, sqrt, ceil

import torch
import torch.nn as nn

from hiner_utils import quant_tensor


# ----------------------------------------------------------------------------- helpers
def OutImg(x, out_bias="tanh"):
    if out_bias == "sigmoid":
        return torch.sigmoid(x)
    elif out_bias == "tanh":
        return (torch.tanh(x) * 0.5) + 0.5
    else:
        return x + float(out_bias)


def ActivationLayer(act_type):
    table = {
        "relu": lambda: nn.ReLU(True),
        "leaky": lambda: nn.LeakyReLU(inplace=True),
        "leaky01": lambda: nn.LeakyReLU(negative_slope=0.1, inplace=True),
        "relu6": lambda: nn.ReLU6(inplace=True),
        "gelu": lambda: nn.GELU(),
        "swish": lambda: nn.SiLU(inplace=True),
        "softplus": lambda: nn.Softplus(),
        "hardswish": lambda: nn.Hardswish(inplace=True),
    }
    if act_type not in table:
        raise KeyError(f"Unknown activation function {act_type}.")
    return table[act_type]()


def NormLayer1d(norm_type, ch_width):
    if norm_type == "none":
        return nn.Identity()
    elif norm_type == "bn":
        return nn.BatchNorm1d(num_features=ch_width)
    elif norm_type == "in":
        return nn.InstanceNorm1d(num_features=ch_width)
    raise NotImplementedError


class Mlp(nn.Module):
    """Same 2-layer MLP as classical HINER (in -> hidden -> out)."""

    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(int(in_features), int(hidden_features))
        self.act = act_layer()
        self.fc2 = nn.Linear(int(hidden_features), int(out_features))

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


# ----------------------------------------------------------------------------- 2D positional encoding
class PositionEncoding2D(nn.Module):
    """(N, 2) coords in [0,1] -> (N, 4L) Fourier features.

    For each coordinate c and base b_k = lbase**k * pi (k = 0..L-1):
        [sin(i*b), cos(i*b), sin(j*b), cos(j*b)]  -> 4L features.
    """

    def __init__(self, pe_embed="pe_1.25_80"):
        super().__init__()
        lbase, levels = [float(x) for x in pe_embed.split("_")[-2:]]
        self.levels = int(levels)
        # plain tensor attribute (not a buffer) -> stays out of state_dict, like HINER
        self.pe_bases = lbase ** torch.arange(self.levels) * pi

    @property
    def out_dim(self):
        return 4 * self.levels

    def forward(self, coords):
        bases = self.pe_bases.to(coords.device)
        i = coords[..., 0:1] * bases  # (N, L)
        j = coords[..., 1:2] * bases  # (N, L)
        return torch.cat(
            [torch.sin(i), torch.cos(i), torch.sin(j), torch.cos(j)], dim=-1
        )  # (N, 4L)


# ----------------------------------------------------------------------------- 1D pixelshuffle decoder
class PixelShuffle1D(nn.Module):
    """(N, C*r, L) -> (N, C, L*r)."""

    def __init__(self, upscale):
        super().__init__()
        self.r = int(upscale)

    def forward(self, x):
        n, cr, l = x.shape
        c = cr // self.r
        return x.view(n, c, self.r, l).permute(0, 1, 3, 2).reshape(n, c, l * self.r)


class UpConv1D(nn.Module):
    def __init__(self, ngf, new_ngf, strd, ks, conv_type, bias):
        super().__init__()
        if conv_type == "pshuffel":
            self.upconv = nn.Sequential(
                nn.Conv1d(ngf, new_ngf * strd, ks, 1, ceil((ks - 1) // 2), bias=bias),
                PixelShuffle1D(strd) if strd != 1 else nn.Identity(),
            )
        elif conv_type == "conv":
            self.upconv = nn.ConvTranspose1d(ngf, new_ngf, ks + strd, strd, ceil(ks / 2))
        else:
            raise NotImplementedError(conv_type)

    def forward(self, x):
        return self.upconv(x)


class NeRVBlock1D(nn.Module):
    def __init__(self, ngf, new_ngf, strd, ks, conv_type, norm, act, bias=True):
        super().__init__()
        self.conv = UpConv1D(ngf, new_ngf, strd, ks, conv_type, bias)
        self.norm = NormLayer1d(norm, new_ngf)
        self.act = ActivationLayer(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# ----------------------------------------------------------------------------- model
class HINER2D(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc_len = int(args.fc_len)
        self.out_bias = args.out_bias

        # decoder kernels (same "_a_b_c" format as HINER; first entry is unused)
        _, ks_dec1, ks_dec2 = [int(x) for x in args.ks.split("_")]
        _, dec_blks = [int(x) for x in args.num_blks.split("_")]
        seed_ch = int(args.enc_dim.split("_")[1])  # channels of the MLP seed map
        self.seed_ch = seed_ch

        # --- fixed encoder: 2D Fourier PE + MLP -> (seed_ch, fc_len) seed ---
        self.pe_embed = PositionEncoding2D("pe_1.25_80")
        pe_dim = self.pe_embed.out_dim  # 4L
        self.encoder = Mlp(pe_dim, 2 * pe_dim, seed_ch * self.fc_len)

        # --- decoder: 1x1 conv to fc_dim, then 1D pixelshuffle upsampling ---
        ngf = int(args.fc_dim)
        decoder_layers = [nn.Conv1d(seed_ch, ngf, 1, 1, 0)]  # analog of HINER decoder[0]
        for i, strd in enumerate(args.dec_strds):
            reduction = sqrt(strd) if args.reduce == -1 else args.reduce
            new_ngf = int(max(round(ngf / reduction), args.lower_width))
            for j in range(dec_blks):
                decoder_layers.append(
                    NeRVBlock1D(
                        ngf=ngf,
                        new_ngf=new_ngf,
                        strd=1 if j else strd,
                        ks=min(ks_dec1 + 2 * i, ks_dec2),
                        conv_type="pshuffel",
                        norm=args.norm,
                        act=args.act,
                    )
                )
                ngf = new_ngf
        self.decoder = nn.ModuleList(decoder_layers)
        self.head_layer = nn.Conv1d(ngf, 1, 3, 1, 1)

    def forward(self, coords):
        # coords: (N, 2) in [0,1]
        feats = self.pe_embed(coords).float()
        x = self.encoder(feats).view(-1, self.seed_ch, self.fc_len)  # (N, seed_ch, fc_len)
        dec_start = time.time()
        x = self.decoder[0](x)
        for layer in self.decoder[1:]:
            x = layer(x)
        out = OutImg(self.head_layer(x), self.out_bias)  # (N, 1, B)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dec_time = time.time() - dec_start
        return out.squeeze(1), dec_time  # (N, B)


class HINER2DDecoder(nn.Module):
    """Decoder-only module (for tracing / standalone decode from a seed map)."""

    def __init__(self, model):
        super().__init__()
        self.out_bias = model.out_bias
        self.decoder = model.decoder
        self.head_layer = model.head_layer

    def forward(self, seed):
        x = self.decoder[0](seed)
        for layer in self.decoder[1:]:
            x = layer(x)
        return OutImg(self.head_layer(x), self.out_bias).squeeze(1)
