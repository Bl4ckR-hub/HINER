import time
import torch
import logging

import torch.nn as nn
import torch.nn.functional as F

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()
        
    def forward(self, input):
        return input


class round_noise_ste(torch.autograd.Function):
    """
    Implement Straight-Through Estimator for rounding noise.
    """
    @staticmethod
    def forward(ctx, x):
        half = float(0.5)
        noise = torch.empty_like(x).uniform_(-half, half)
        return torch.round(x + noise)

    @staticmethod
    def backward(ctx, g):
        return g

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def floor_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.floor() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()
        # return torch.mean((pred-tgt)**p)

        
class UniformAffineQuantizer(nn.Module):
    r"""
        PyTorch Function that can be used for asymmetric quantization (also called uniform affine
        quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
        through' on the backward pass, ignoring the quantization that occurred.
        Based on https://arxiv.org/abs/1806.08342.
    Args:
        n_bits: number of bit for quantization
        symmetric: if True, the zero_point should always be 0
        channel_wise: if True, compute scale and zero_point in each channel
        scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, 
                    symmetric: bool = False, 
                    channel_wise: bool = False, 
                    scale_method: str = 'max', 
                    prob: float = 1.0):
        super(UniformAffineQuantizer, self).__init__()
        
        self.sym = symmetric
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None #scaling factors
        self.zero_point = None
        self.eps = torch.tensor(1e-8, dtype=torch.float32)
        self.inited = False
        self.channel_wise = channel_wise
        self.scale_method = scale_method
        
        
        '''do like dropout'''
        self.prob = prob
        self.is_training = False

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.delta = nn.Parameter(self.delta)
            self.inited = True

        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        
        if self.is_training and self.prob < 1.0:
            x_ans = torch.where(torch.rand_like(x) < self.prob, x_dequant, x)
        else:
            x_ans = x_dequant
        return x_ans
    
    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0] 
            
            if len(x.shape) == 4: # conv.weight
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                
                delta = x_max.clone()
                zero_point = x_max.clone()
                
                for c in range(n_channels):
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
                    
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2: # linear.weight
                x_max = x_clone.abs().max(dim=-1)[0]
                
                delta = x_max.clone()
                zero_point = x_max.clone()
                
                for c in range(n_channels):
                    delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
                    
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
            elif len(x.shape) == 1: # bias
                x_max = x_clone.abs().max()
                
                delta = x_max.clone()
                zero_point = x_max.clone()
                
                delta, zero_point = self.init_quantization_scale(x_clone, channel_wise=False)
                delta = delta.view(-1)
                zero_point = zero_point.view(-1)
            else: 
                raise ValueError
        else:
            x_min = min(x.min().item(), 0)
            x_max = max(x.max().item(), 0)
            if 'scale' in self.scale_method:
                x_min = x_min * (self.n_bits + 2) / 8
                x_max = x_max * (self.n_bits + 2) / 8
                
            x_absmax = max(abs(x_min), x_max)
            if self.sym:
                x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax
                
            delta = torch.tensor((x_max - x_min) / (self.n_levels - 1)).type_as(x)
            delta = torch.max(delta, self.eps)
            
            # zero_point = round(-x_min / delta)
            zero_point = (- x_min / delta).round().type_as(x)
            # delta = torch.tensor(delta).type_as(x)
            # zero_point = torch.tensor(zero_point).type_as(x)

        return delta, zero_point
    
    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        delta = torch.max(delta, self.eps)
        zero_point = (- min / delta).round()
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q
    
    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits
        
    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},'
        return s.format(**self.__dict__)



class AdaRoundQuantizer(nn.Module):
    r"""
        Adaptive Rounding Quantizer, used to optimize the rounding policy
        by reconstructing the intermediate output.
        Based on
        Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568
    Args:
        uaq: UniformAffineQuantizer, used to initialize quantization parameters in this quantizer
        round_mode: controls the forward pass in this quantizer
        weight_tensor: initialize alpha
    """
    
    def __init__(self, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor, round_mode='learned_round_sigmoid'):
        super(AdaRoundQuantizer, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta.half().float()
        self.zero_point = uaq.zero_point.half().float()
        self.n_levels = uaq.n_levels
        
        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False
        self.x_quant = None
        
        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.init_alpha(x=weight_tensor.clone())
        
    def forward(self, x):
        x_floor = torch.floor(x / self.delta)
        # x_floor = floor_ste(x / self.delta)
        if self.soft_targets:
            x_int = x_floor + self.get_soft_targets()
        else:
            x_int = x_floor + (self.alpha >= 0).float()
        
        self.x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        x_float_q = (self.x_quant - self.zero_point) * self.delta
        
        return x_float_q
    
    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
    
    def init_alpha(self, x: torch.Tensor):
        # each element has its alpha
        x_floor = torch.floor(x / self.delta)
        if self.round_mode == 'learned_hard_sigmoid':
            logging.info('Init alpha to be FP32')
            init_start = time.time()
            rest = (x / self.delta) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = nn.Parameter(alpha)
            
            # self.delta = nn.Parameter(self.delta)
            
            init_time = time.time() - init_start
            logging.info('init time: {}'.format(init_time))
        else:
            raise NotImplementedError
    
    @torch.jit.export
    def extra_repr(self):
        return 'bit={}'.format(self.n_bits)