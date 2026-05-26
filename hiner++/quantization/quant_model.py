import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union

from models import HINER
from quantization.quantizer import StraightThrough, UniformAffineQuantizer


class QuantModule(nn.Module):
    r"""
        Convert module to quantmodule.
    """

    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {}):
        super(QuantModule, self).__init__()
        
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Linear):
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        else:
            raise ValueError('Not supported modules: {}'.format(org_module))
        
        
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        self.bias = org_module.bias
        self.org_bias = org_module.bias.data.clone()
            
        # de-activate the quantized forward default
        self.use_weight_quant = False
        
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.bias_quantizer = UniformAffineQuantizer(**weight_quant_params)
        
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False
        
        self.extra_repr = org_module.extra_repr
    
    def forward(self, input: torch.Tensor):
        
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias_quantizer(self.bias)
        else:
            weight = self.org_weight
            bias = self.org_bias
        
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        out = self.activation_function(out)
        return out
    
    def set_quant_state(self, weight_quant: bool = False):
        self.use_weight_quant = weight_quant

class QuantModel(nn.Module):

    def __init__(self, model: HINER, weight_quant_params: dict = {}):
        super().__init__()
        self.model = model
        self.quant_module_refactor(self.model, weight_quant_params)

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}):
        r"""
            Recursively replace the module to QuantModule.
        Args:
            module: nn.Module with children modules
            weight_quant_params: quantization parameters for weight quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            
            if 'encoder' in name:
                continue
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(module, name, QuantModule(child_module, weight_quant_params))
                prev_quantmodule = getattr(module, name)
            elif isinstance(child_module, (nn.GELU,)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue
            elif isinstance(child_module, StraightThrough):
                continue                   
            else:
                self.quant_module_refactor(child_module, weight_quant_params)
                
    def set_quant_state(self, weight_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule)):
                m.set_quant_state(weight_quant)

    def forward(self, img_data, norm_idx, img_idx):
        return self.model.forward(img_data, norm_idx, img_idx)
    
    def compute_model_bits(self):
        bits = 0.
        for m in self.model.modules():
            if isinstance(m, QuantModule):
                cur_bit = m.weight_quantizer.n_bits * m.weight.numel() + m.bias_quantizer.n_bits * m.bias.numel()
                bits += cur_bit
        return bits
