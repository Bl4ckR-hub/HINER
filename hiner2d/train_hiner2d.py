"""Train a HINER2D coordinate-INR on a single hyperspectral cube.

Bitstream = the QUANTIZED NETWORK WEIGHTS (encoder MLP + 1D decoder + head).
Unlike 1D HINER there are no stored per-pixel latents (there would be H*W of
them, larger than the cube); embeddings are recomputed from the coordinate grid
at decode time, so the whole net is the compressed artifact and
    bpp = total_weight_bits / (H * W * B).
"""

import os
import json
import random
import shutil
import argparse
from datetime import datetime

import numpy as np
from copy import deepcopy
from dahuffman import HuffmanCodec

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from hiner_utils import adjust_lr, quant_tensor, RoundTensor, worker_init_fn
from data import HSIPixelDataSet
from models import HINER2D


# ----------------------------------------------------------------------------- losses / metrics
def spectral_loss(pred, gt, weight, loss_type):
    # pred, gt: (N, B); weight: (N,)
    if loss_type == "L2":
        err = ((pred - gt) ** 2).mean(dim=1)
    elif loss_type == "L1":
        err = (pred - gt).abs().mean(dim=1)
    elif loss_type == "L1_L2":
        err = 0.5 * ((pred - gt) ** 2).mean(1) + 0.5 * (pred - gt).abs().mean(1)
    else:
        raise ValueError(f"unknown loss {loss_type}")
    return (err * weight).sum() / weight.sum().clamp_min(1e-8)


def psnr_db(mse):
    return -10.0 * torch.log10(mse + 1e-9)


def sam_mean(pred, gt):
    # mean spectral angle (radians) over pixels; pred, gt: (N, B)
    num = (pred * gt).sum(dim=1)
    den = pred.norm(dim=1) * gt.norm(dim=1) + 1e-8
    return torch.acos((num / den).clamp(-1 + 1e-6, 1 - 1e-6)).mean()


# ----------------------------------------------------------------------------- reconstruction
@torch.no_grad()
def reconstruct(model, coords, chunk, device):
    out = []
    model.eval()
    for s in range(0, coords.shape[0], chunk):
        c = coords[s : s + chunk].to(device)
        pred, _ = model(c)
        out.append(pred.cpu())
    model.train()
    return torch.cat(out, 0)  # (H*W, B)


def metrics_over(pred, gt, roi):
    res = {}
    mse = ((pred - gt) ** 2).mean()
    res["psnr_db"] = float(psnr_db(mse))
    res["sam_rad"] = float(sam_mean(pred, gt))
    res["rmse"] = float(torch.sqrt(mse))
    if roi is not None and roi.any() and not roi.all():
        p, g = pred[roi], gt[roi]
        res["psnr_roi_db"] = float(psnr_db(((p - g) ** 2).mean()))
        res["sam_roi_rad"] = float(sam_mean(p, g))
    return res


# ----------------------------------------------------------------------------- quantization / bpp
@torch.no_grad()
def quantize_model(model, bits):
    """Quantize ALL weights (encoder included). Returns (quant_model, quant_ckt)."""
    qmodel = deepcopy(model)
    ckt = qmodel.state_dict()
    quant_ckt = {}
    for k, v in ckt.items():
        if not v.dtype.is_floating_point or v.nelement() <= 1:
            continue  # leave tiny / non-float buffers alone
        qd, deq = quant_tensor(v, bits)
        quant_ckt[k] = qd
        ckt[k] = deq
    qmodel.load_state_dict(ckt)
    return qmodel, quant_ckt


def bitstream_stats(quant_ckt, num_pixels_times_bands):
    vals, tmin_scale_len = [], 0
    for layer in quant_ckt.values():
        vals.extend(layer["quant"].flatten().tolist())
        tmin_scale_len += layer["min"].nelement() + layer["scale"].nelement()
    unique, counts = np.unique(vals, return_counts=True)
    freq = dict(zip(unique, counts))
    codec = HuffmanCodec.from_data(vals)
    sym_bits = {k: v[0] for k, v in codec.get_code_table().items()}
    total_bits = sum(freq[n] * sym_bits[n] for n in freq)
    bits_per_param = total_bits / len(vals)
    total_bits += tmin_scale_len * 16  # float16 min/scale overhead
    return {
        "bits_per_param": bits_per_param,
        "bits_per_param_with_overhead": total_bits / len(vals),
        "bits_per_pixel": total_bits / num_pixels_times_bands,
        "num_quant_params": len(vals),
    }


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--data_path", type=str, required=True, help="(B,H,W) .npy cube")
    p.add_argument("--vid", type=str, default="hsi", help="id for output naming")
    # masking
    p.add_argument("--mask_path", type=str, default="", help="(H,W) bool .npy ROI mask")
    p.add_argument("--mask_mode", type=str, default="none", choices=["none", "hard", "soft"])
    p.add_argument("--soft_weight", type=float, default=0.1, help="loss weight outside ROI (soft)")
    # architecture
    p.add_argument("--fc_len", type=int, default=101, help="seed length; fc_len*prod(dec_strds) must == B")
    p.add_argument("--fc_dim", type=int, default=128, help="decoder base width")
    p.add_argument("--enc_dim", type=str, default="64_16", help="(unused_64)_(seed channels)")
    p.add_argument("--dec_strds", type=int, nargs="+", default=[2], help="1D upsample strides")
    p.add_argument("--num_blks", type=str, default="1_1", help="(enc)_(dec) blocks per stage")
    p.add_argument("--ks", type=str, default="0_3_3", help="(enc)_(dec1)_(dec2) kernel sizes")
    p.add_argument("--reduce", type=float, default=1.2, help="channel reduction per stage")
    p.add_argument("--lower_width", type=int, default=32)
    p.add_argument("--norm", type=str, default="none", choices=["none", "bn", "in"])
    p.add_argument("--act", type=str, default="gelu")
    p.add_argument("--out_bias", type=str, default="tanh")
    # training
    p.add_argument("-b", "--batchSize", type=int, default=4096, help="pixels per batch")
    p.add_argument("-j", "--workers", type=int, default=4)
    p.add_argument("-e", "--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_type", type=str, default="cosine_0.1_1_0.1")
    p.add_argument("--loss", type=str, default="L2", choices=["L2", "L1", "L1_L2"])
    # eval / quant
    p.add_argument("--eval_freq", type=int, default=50)
    p.add_argument("--quant_model_bit", type=int, default=8)
    p.add_argument("--eval_chunk", type=int, default=8192, help="pixels per reconstruction chunk")
    # logging
    p.add_argument("--manualSeed", type=int, default=1)
    p.add_argument("--print_freq", type=int, default=1)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--outf", type=str, default="hiner2d")
    p.add_argument("--suffix", type=str, default="")
    args = p.parse_args()

    cudnn.benchmark = True
    torch.manual_seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    random.seed(args.manualSeed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- data ---
    dataset = HSIPixelDataSet(args)
    B, H, W = dataset.B, dataset.H, dataset.W
    prod = int(np.prod(args.dec_strds))
    assert args.fc_len * prod == B, (
        f"fc_len*prod(dec_strds) = {args.fc_len}*{prod} = {args.fc_len*prod} != B={B}. "
        f"Pick fc_len and --dec_strds so they multiply to {B}."
    )

    exp_id = (
        f"{args.vid}/fclen{args.fc_len}_fcdim{args.fc_dim}_dec{'-'.join(map(str,args.dec_strds))}"
        f"_blk{args.num_blks}_e{args.epochs}_b{args.batchSize}_{args.loss}"
        f"_q{args.quant_model_bit}_mask-{args.mask_mode}{args.suffix}"
    )
    outf = os.path.join("output", "debug" if args.debug else args.outf, exp_id)
    if args.overwrite and os.path.isdir(outf):
        shutil.rmtree(outf)
    os.makedirs(outf, exist_ok=True)
    print(f"[HINER2D] cube=({B},{H},{W}) train_pixels={len(dataset)} mask={args.mask_mode} -> {outf}")

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batchSize, shuffle=True, num_workers=args.workers,
        pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn,
    )

    # full grid + gt for reconstruction / metrics
    full_coords = dataset.coords
    gt_full = dataset.spectra  # (H*W, B)
    roi = dataset.roi if args.mask_mode != "none" else None

    # --- model ---
    model = HINER2D(args).to(device)
    n_params = sum(q.nelement() for q in model.parameters())
    print(f"[HINER2D] params: {n_params/1e6:.4f} M  (encoder+decoder+head, all stored)")
    optimizer = optim.Adam(model.parameters(), weight_decay=0.0)

    # --- train ---
    start = datetime.now()
    for epoch in range(args.epochs):
        model.train()
        psnrs = []
        for i, sample in enumerate(loader):
            coord = sample["coord"].to(device)
            gt = sample["spectrum"].to(device)
            wt = sample["weight"].to(device)
            if args.debug and i > 5:
                break
            cur = (epoch + i / len(loader)) / args.epochs
            lr = adjust_lr(optimizer, cur, args)
            pred, _ = model(coord)
            loss = spectral_loss(pred, gt, wt, args.loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            psnrs.append(psnr_db(((pred.detach() - gt) ** 2).mean()))
        if (epoch % args.print_freq == 0) or (epoch == args.epochs - 1):
            print(f"[{datetime.now():%H:%M:%S}] epoch {epoch+1}/{args.epochs} "
                  f"lr {lr:.2e} train_PSNR {torch.stack(psnrs).mean():.2f}", flush=True)
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            pred_full = reconstruct(model, full_coords, args.eval_chunk, device)
            m = metrics_over(pred_full, gt_full, roi)
            print(f"           eval: " + " | ".join(f"{k} {v:.4f}" for k, v in m.items()), flush=True)

    train_time = str(datetime.now() - start)

    # --- final eval (full precision + quantized) + bitstream ---
    pred_full = reconstruct(model, full_coords, args.eval_chunk, device)
    res_fp = metrics_over(pred_full, gt_full, roi)

    qmodel, quant_ckt = quantize_model(model, args.quant_model_bit)
    pred_q = reconstruct(qmodel, full_coords, args.eval_chunk, device)
    res_q = metrics_over(pred_q, gt_full, roi)
    bits = bitstream_stats(quant_ckt, H * W * B)

    torch.save({"state_dict": model.state_dict(), "args": vars(args)}, f"{outf}/model.pth")

    results = {
        "sample": {"data_path": args.data_path, "vid": args.vid,
                   "num_bands": B, "image_hw": [H, W],
                   "train_pixels": len(dataset), "mask_mode": args.mask_mode},
        "model": {"params_M": n_params / 1e6, "fc_len": args.fc_len,
                  "fc_dim": args.fc_dim, "dec_strds": args.dec_strds,
                  "quant_bit": args.quant_model_bit},
        "metrics_full_precision": res_fp,
        "metrics_quantized": res_q,
        "bitstream": bits,
        "epochs": args.epochs,
        "train_time": train_time,
    }
    out_json = f"{outf}/results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[HINER2D] done in {train_time}")
    print(f"[HINER2D] results -> {out_json}")
    print(json.dumps({**res_q, **bits}, indent=2, default=str))


if __name__ == "__main__":
    main()
