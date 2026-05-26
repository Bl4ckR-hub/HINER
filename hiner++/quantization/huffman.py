import torch
import numpy as np
from dahuffman import HuffmanCodec
from quantization.quant_model import QuantModule

def huffman_coding(args, embed, qnn):
    embed_huffman_coding(args, embed)
    model_huffman_coding(args, qnn)
    args.full_bpp = args.embed_bpp + args.model_bpp

    return args.full_bpp

def embed_huffman_coding(args, embed):
    
    quant_embed = quant_tensor(embed, 6)
    quant_v_list = quant_embed['quant'].flatten().tolist()
    tmin_scale_len = quant_embed['min'].nelement() + quant_embed['scale'].nelement()
    # get the element name and its frequency
    unique, counts = np.unique(quant_v_list, return_counts=True)
    num_freq = dict(zip(unique, counts))
    
    # generating HuffmanCoding table
    codec = HuffmanCodec.from_data(quant_v_list)
    # encode = codec.encode(quant_v_list)
    sym_bit_dict = {}
    for k, v in codec.get_code_table().items():
        sym_bit_dict[k] = v[0]
        
    # total bits for quantized embed + model weights
    total_bits = 0
    for num, freq in num_freq.items():
        total_bits += freq * sym_bit_dict[num]
    
    # including the overhead for min and scale storage, 
    total_bits += tmin_scale_len * 16               #(16bits for float16)
    args.embed_bits_per_param = total_bits / len(quant_v_list)
    
    # bits per pixel
    args.embed_bpp = total_bits / args.final_size / args.full_data_length


def model_huffman_coding(args, qnn):
    
    quant_v_list = []
    tmin_scale_len = 0
    for m in qnn.modules():
        if isinstance(m, QuantModule):
            quant_v_list.extend(m.weight_quantizer.x_quant.flatten().tolist())
            quant_v_list.extend(m.bias_quantizer.x_quant.flatten().tolist())
            tmin_scale_len += m.bias_quantizer.zero_point.nelement() + m.bias_quantizer.delta.nelement()
            
    # get the element name and its frequency
    unique, counts = np.unique(quant_v_list, return_counts=True)
    num_freq = dict(zip(unique, counts))
    
    # generating HuffmanCoding table
    codec = HuffmanCodec.from_data(quant_v_list)
    # encode = codec.encode(quant_v_list)
    sym_bit_dict = {}
    for k, v in codec.get_code_table().items():
        sym_bit_dict[k] = v[0]
        
    # total bits for quantized embed + model weights
    total_bits = 0
    for num, freq in num_freq.items():
        total_bits += freq * sym_bit_dict[num]
    
    # including the overhead for min and scale storage, 
    total_bits += tmin_scale_len * 16               #(16bits for float16)
    args.model_bits_per_param = total_bits / len(quant_v_list)
    
    # bits per pixel
    args.model_bpp = total_bits / args.final_size / args.full_data_length

def quant_tensor(t, bits=8):
    tmin_scale_list = []
    # quantize over the whole tensor, or along each dimenstion
    t_min, t_max = t.min(), t.max()
    scale = (t_max - t_min) / (2**bits-1)
    tmin_scale_list.append([t_min, scale])
    for axis in range(t.dim()):
        t_min, t_max = t.min(axis, keepdim=True)[0], t.max(axis, keepdim=True)[0]
        if t_min.nelement() / t.nelement() < 0.02:
            scale = (t_max - t_min) / (2**bits-1)
            # tmin_scale_list.append([t_min, scale]) 
            tmin_scale_list.append([t_min.to(torch.float16), scale.to(torch.float16)]) 
    
    quant_t_list, new_t_list, err_t_list = [], [], []
    for t_min, scale in tmin_scale_list:
        t_min, scale = t_min.expand_as(t), scale.expand_as(t)
        quant_t = ((t - t_min) / (scale)).round().clamp(0, 2**bits-1)
        new_t = t_min + scale * quant_t
        err_t = (t - new_t).abs().mean()
        quant_t_list.append(quant_t)
        new_t_list.append(new_t)
        err_t_list.append(err_t)   
        
    best_err_t = min(err_t_list)
    best_quant_idx = err_t_list.index(best_err_t)
    best_new_t = new_t_list[best_quant_idx]
    best_quant_t = quant_t_list[best_quant_idx].to(torch.uint8)
    best_tmin = tmin_scale_list[best_quant_idx][0]
    best_scale = tmin_scale_list[best_quant_idx][1]
    quant_t = {'quant': best_quant_t, 'min': best_tmin, 'scale': best_scale}
    
    return quant_t      
