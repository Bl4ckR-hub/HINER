import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from hnerv_utils import loss_fn
from quantization.quantizer import AdaRoundQuantizer
from quantization.quant_model import QuantModel, QuantModule



class LossFunction:
    def __init__(self,
                    model: nn.Module,
                    round_loss: str = 'relaxation',
                    rec_loss: str = 'SAM',
                    weight: float = 1.,
                    max_count: int = 2000,
                    b_range: tuple = (10, 2),
                    decay_start: float = 0.0,
                    warmup: float = 0.0):
        
        self.model = model
        self.round = round_loss
        self.rec = rec_loss
        self.weight = weight
        self.loss_start = max_count * warmup
        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                            start_b=b_range[0], end_b=b_range[1])
        self.count = 0
    
    def collect_round_loss(self, module, b):
        for name, module in module.named_children():
            if 'encoder' in name:
                continue
            elif isinstance(module, QuantModule):
                round_vals = module.weight_quantizer.get_soft_targets()
                self.round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
            else:
                self.collect_round_loss(module, b)
                
        
    def __call__(self, pred, tgt, grad=None, outf=None):
        r"""
            Compute the total loss for optimization:  
                rec_loss is the output reconstruction loss of current layer, 
                round_loss is a regularization term to optimize the rounding policy,
                task_loss is the output reconstruction loss of current coder.
                
        Args:
            pred (tensor): output from current quantized layer
            tgt (tensor): the floating-point output of current layer
            quant_net_out (tensor): output from current quantized coder
            cali_data (tensor): the floating-point output of current coder
            grad (tensor): gradients to compute fisher information
            return: total loss function
        """
        self.count += 1
        
        rec_loss = loss_fn(pred, tgt, self.rec)
        
        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round == 'none':
            b = self.round_loss = 0
        elif self.round == 'relaxation':
            self.round_loss = 0
            self.collect_round_loss(self.model, b)
        else:
            raise NotImplementedError
        
        total_loss = self.round_loss + rec_loss
        if self.count % 500 == 0:
            print_str = 'Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                            float(total_loss), float(rec_loss), float(self.round_loss), b, self.count)
            print(print_str, flush=True)
            with open('{}/eval.txt'.format(outf), 'a') as f:
                f.write(print_str + '\n\n')   
        return total_loss


def model_reconstruction(model: QuantModel, gt: DataLoader, outf: str, loss: str,
                            iters: int = 20000, weight: float = 0.01, 
                            b_range: tuple = (20, 2), 
                            warmup: float = 0.0, p: float = 2.0, lr: float = 0.0015):
    r"""
    Args:
            model: QuantModel
            iters: optimization iterations
            weight: the weight of rounding regularization term
            b_range: temperature range
            warmup: proportion of iterations that no scheduling for temperature
            p: L_p norm minimization
    """
    
    model.set_quant_state(True)
    round_mode = 'learned_hard_sigmoid'    
    
    opt_params = []
    def set_optimizer(model: nn.Module, opt_params):
        for name, module in model.named_children():
            if 'encoder' in name:
                # opt_params += list(module.parameters())
                continue
            elif isinstance(module, QuantModule):
                module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                            weight_tensor=module.org_weight.data)
                module.weight_quantizer.soft_targets = True
                opt_params += [module.weight_quantizer.alpha]
                
                module.bias_quantizer = AdaRoundQuantizer(uaq=module.bias_quantizer, round_mode=round_mode,
                                                            weight_tensor=module.bias.data)
                module.bias_quantizer.soft_targets = True
                opt_params += [module.bias_quantizer.alpha]
            else:
                set_optimizer(module, opt_params)
    
    
    set_optimizer(model, opt_params)
    optimizer = torch.optim.Adam(opt_params, lr=lr)
    # param_optimizer = torch.optim.Adam(param, lr=0.0005)
    scheduler = None

    loss_mode = 'relaxation'
    
    loss_func = LossFunction(model, round_loss=loss_mode, rec_loss=loss, weight=weight, 
                                max_count=iters, b_range=b_range, 
                                decay_start=0, warmup=warmup)
    # opt alpha
    epochs = int(iters / len(gt))
    print(epochs)
    for epoch in range(epochs):
        model.train()
        device = next(model.parameters()).device
        for i, sample in enumerate(gt):
            
            img_data, norm_idx, img_idx = sample['img'].to(device), sample['norm_idx'].to(device), sample['idx'].to(device)
            img_out, _, _ = model(img_data, norm_idx, img_idx)

            optimizer.zero_grad()
            err = loss_func(pred=img_out, tgt=img_data, grad=None, outf=outf)
            err.backward()
            
            optimizer.step()
            if scheduler:
                scheduler.step()
            
    torch.cuda.empty_cache()
    # model.eval()
    
    def set_quantizer(model: nn.Module):
        for name, module in model.named_children():
            if 'encoder' in name:
                continue
            elif isinstance(module, QuantModule):
                module.weight_quantizer.soft_targets = False
            else:
                set_quantizer(module)
    
    set_quantizer(model)



class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b
        
    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))

