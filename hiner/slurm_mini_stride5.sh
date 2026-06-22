#!/bin/bash
#SBATCH --job-name=hiner_mini5
#SBATCH --output=logs/hiner_mini5_%j.out
#SBATCH --error=logs/hiner_mini5_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#
# Train HINER on EVERY Nth patch of the 'mini' / 'train' split of HySpecNet-11k
# and dump per-sample evaluation results (PSNR, compression time, SAM, bpppb)
# as one JSON per sample under hiner/results/, plus a combined summary CSV.
#
# Usage:
#   sbatch slurm_mini_stride5.sh [EPOCHS] [STRIDE] [SPLIT]
#     EPOCHS  number of training epochs per patch   (default: 300)
#     STRIDE  take every STRIDEth patch              (default: 5)
#     SPLIT   HySpecNet split name under splits/     (default: mini)
#
# HySpecNet patches are (202, 128, 128) (B,H,W). The decoder geometry
# (fc_hw=8_8, dec_strds=2 2 2 2 -> 8*16 = 128) is tied to that 128x128 size.

set -eo pipefail

# ---- config ----------------------------------------------------------------
EPOCHS=${1:-300}
STRIDE=${2:-5}
SPLIT=${3:-mini}
MODELSIZE=1.5
OUTF="mini_${SPLIT}_stride${STRIDE}"
PROJECT_DIR="/faststorage/viktor/project_context/hsi-toolkit/hiner_forked/HINER/hiner"
ENV_FILE="/faststorage/viktor/project_context/hsi-toolkit/.env"
RESULTS_DIR="$PROJECT_DIR/results"
SUMMARY_CSV="$RESULTS_DIR/${OUTF}_summary.csv"

cd "$PROJECT_DIR"
mkdir -p logs results "$RESULTS_DIR"

# ---- environment -----------------------------------------------------------
eval "$(micromamba shell hook --shell bash)"
micromamba activate hiner_forked
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# ---- dataset root from .env ------------------------------------------------
set -a; source "$ENV_FILE"; set +a
: "${HSI_DATASET_ROOT:?HSI_DATASET_ROOT not set in $ENV_FILE}"

TRAIN_CSV="$HSI_DATASET_ROOT/splits/$SPLIT/train.csv"
[ -f "$TRAIN_CSV" ] || { echo "ERROR: split csv not found: $TRAIN_CSV"; exit 1; }

# ---- select every STRIDEth patch (lines 1, 1+STRIDE, 1+2*STRIDE, ...) ------
mapfile -t SAMPLES < <(awk -v s="$STRIDE" 'NR % s == 1' "$TRAIN_CSV")
echo "Split          : $SPLIT/train.csv"
echo "Total patches  : $(wc -l < "$TRAIN_CSV")"
echo "Stride         : $STRIDE  ->  selected ${#SAMPLES[@]} patches"
echo "Epochs/patch   : $EPOCHS"
echo "Output dir     : output/$OUTF"
echo "Results dir    : $RESULTS_DIR"
echo "----------------------------------------------------------------------"

# Reset the summary CSV header (per-sample JSONs are written incrementally).
echo "vid,data_path,num_bands,height,width,epochs,quant_seen_psnr_db,pred_seen_psnr_db,sam_deg,sam_rad,bits_per_pixel_bpppb,compression_time" > "$SUMMARY_CSV"

# ---- loop over selected patches --------------------------------------------
idx=0
for SAMPLE_REL in "${SAMPLES[@]}"; do
    idx=$((idx + 1))
    DATA_PATH="$HSI_DATASET_ROOT/patches/$SAMPLE_REL"
    if [ ! -f "$DATA_PATH" ]; then
        echo "[$idx/${#SAMPLES[@]}] SKIP (missing): $DATA_PATH"
        continue
    fi
    VID=$(basename "$(dirname "$DATA_PATH")")     # patch id, used for naming
    echo "======================================================================"
    echo "[$idx/${#SAMPLES[@]}] VID=$VID"
    echo "  data_path : $DATA_PATH"

    # ---- train (--dump_image saves the reconstructed cube for SAM) ---------
    python -u train_hiner.py \
        --data_path "$DATA_PATH" \
        --vid "$VID" \
        --outf "$OUTF" \
        --data_split 1_1_1 \
        --ori_shape 128_128_202 \
        --modelsize "$MODELSIZE" \
        --fc_hw 8_8 \
        --dec_strds 2 2 2 2 \
        --enc_dim 64_16 \
        --num_blks 1_1 \
        --loss Fusion6 \
        --lr 0.001 \
        --epochs "$EPOCHS" \
        --eval_freq 10 \
        --quant_model_bit 8 \
        --quant_embed_bit 6 \
        --dump_image \
        --overwrite

    # ---- collect metrics for this sample into results/<vid>.json -----------
    python - "$PROJECT_DIR/output/$OUTF/$VID" "$DATA_PATH" "$VID" "$EPOCHS" \
             "$RESULTS_DIR/${VID}.json" "$SUMMARY_CSV" <<'PY'
import sys, glob, os, json
import numpy as np
import pandas as pd
import scipy.io as sio

out_dir, data_path, vid, epochs, out_json, summary_csv = sys.argv[1:7]

# --- metrics from the last epoch CSV (Time, bits/pixel, PSNR) ---------------
csvs = sorted(glob.glob(os.path.join(out_dir, "**", "epoch*.csv"), recursive=True),
              key=os.path.getmtime)
if not csvs:
    raise SystemExit(f"No epoch*.csv found under {out_dir}")
row = pd.read_csv(csvs[-1]).iloc[0].to_dict()

# --- SAM from the quantized reconstruction (final_mat) vs normalized GT -----
# Reproduce the per-band min-max normalization used by data/datasets.py so the
# GT lives in the same space as the saved reconstruction.
def per_band_normalize(cube_bhw):
    b = cube_bhw.reshape(cube_bhw.shape[0], -1).astype(np.float64)
    mn = b.min(axis=1, keepdims=True)
    mx = b.max(axis=1, keepdims=True)
    rng = np.where((mx - mn) == 0, 1.0, (mx - mn))
    norm = (b - mn) / rng
    return norm.reshape(cube_bhw.shape).transpose(1, 2, 0)  # (H, W, B)

def sam_mean(gt_hwb, rec_hwb, eps=1e-8):
    g = gt_hwb.reshape(-1, gt_hwb.shape[-1])
    r = rec_hwb.reshape(-1, rec_hwb.shape[-1])
    dot = np.sum(g * r, axis=1)
    ng = np.linalg.norm(g, axis=1)
    nr = np.linalg.norm(r, axis=1)
    valid = (ng > eps) & (nr > eps)
    cos = np.clip(dot[valid] / (ng[valid] * nr[valid]), -1.0, 1.0)
    ang = np.arccos(cos)
    return float(np.mean(ang)) if ang.size else float("nan")

sam_rad = sam_deg = None
# prefer the quantized reconstruction (consistent with bits/pixel + quant PSNR)
mats = (sorted(glob.glob(os.path.join(out_dir, "**", "visualize_model_quant",
                                      "final_mat", "*_final.mat"), recursive=True))
        or sorted(glob.glob(os.path.join(out_dir, "**", "visualize_model_orig",
                                         "final_mat", "*_final.mat"), recursive=True)))
if mats:
    rec = sio.loadmat(mats[-1])["hsi"].astype(np.float64)      # (H, W, B)
    gt = per_band_normalize(np.load(data_path).astype(np.float64))  # (H, W, B)
    if gt.shape == rec.shape:
        sam_rad = sam_mean(gt, rec)
        sam_deg = float(np.degrees(sam_rad))
    else:
        print(f"WARN: shape mismatch gt{gt.shape} vs rec{rec.shape}; SAM skipped")
else:
    print(f"WARN: no final_mat under {out_dir}; SAM skipped")

shp = np.load(data_path, mmap_mode="r").shape  # (B, H, W)
quant_psnr = row.get("best_quant_seen_psnr")
pred_psnr = row.get("pred_seen_psnr", row.get("best_pred_seen_psnr"))
bpppb = row.get("bits/pixel")
comp_time = row.get("Time")

result = {
    "sample": {
        "data_path": data_path, "vid": vid,
        "num_bands": int(shp[0]),
        "image_hw": [int(shp[1]), int(shp[2])],
        "epochs": int(epochs),
    },
    "key_metrics": {
        "quant_seen_psnr_db": quant_psnr,
        "pred_seen_psnr_db": pred_psnr,
        "sam_deg": sam_deg,
        "sam_rad": sam_rad,
        "bits_per_pixel_bpppb": bpppb,
        "compression_time": comp_time,
        "bits_per_param_with_overhead": row.get("bits/param w/ overhead"),
        "model_size_M_enc_dec_total": row.get("Size (M)"),
    },
    "source_csv": csvs[-1],
    "source_final_mat": mats[-1] if mats else None,
    "all_metrics": row,
}
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, "w") as f:
    json.dump(result, f, indent=2, default=str)

# append one row to the combined summary CSV
with open(summary_csv, "a") as f:
    f.write(",".join(str(x) for x in [
        vid, data_path, int(shp[0]), int(shp[1]), int(shp[2]), int(epochs),
        quant_psnr, pred_psnr, sam_deg, sam_rad, bpppb, comp_time,
    ]) + "\n")

print(f"  -> {out_json}")
print(json.dumps(result["key_metrics"], indent=2, default=str))
PY
done

echo "======================================================================"
echo "DONE. Per-sample JSONs in $RESULTS_DIR/ ; summary: $SUMMARY_CSV"
