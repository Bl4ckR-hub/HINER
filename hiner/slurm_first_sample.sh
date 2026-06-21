#!/bin/bash
#SBATCH --job-name=hiner_first
#SBATCH --output=logs/hiner_first_%j.out
#SBATCH --error=logs/hiner_first_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#
# Train HINER on the FIRST HySpecNet sample and dump the results to JSON.
#
# Usage:
#   sbatch slurm_first_sample.sh [EPOCHS] [SPLIT]
#     EPOCHS  number of training epochs        (default: 300)
#     SPLIT   HySpecNet split under splits/     (default: single)
#
# Assumes HySpecNet patches: (202, 128, 128) (B,H,W). The decoder geometry
# (fc_hw=8_8, dec_strds=2 2 2 2 -> 8*16 = 128) is tied to that 128x128 size.

set -eo pipefail

# ---- config ----------------------------------------------------------------
EPOCHS=${1:-300}
SPLIT=${2:-single}
MODELSIZE=1.5
OUTF="hiner_first_sample"
PROJECT_DIR="/faststorage/viktor/project_context/hsi-toolkit/hiner_forked/HINER/hiner"
ENV_FILE="/faststorage/viktor/project_context/hsi-toolkit/.env"

cd "$PROJECT_DIR"
mkdir -p logs results

# ---- environment -----------------------------------------------------------
eval "$(micromamba shell hook --shell bash)"
micromamba activate hiner_forked
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

# ---- dataset root from .env ------------------------------------------------
set -a; source "$ENV_FILE"; set +a
: "${HSI_DATASET_ROOT:?HSI_DATASET_ROOT not set in $ENV_FILE}"

# ---- pick the FIRST data sample of the split -------------------------------
TRAIN_CSV="$HSI_DATASET_ROOT/splits/$SPLIT/train.csv"
[ -f "$TRAIN_CSV" ] || { echo "ERROR: split csv not found: $TRAIN_CSV"; exit 1; }
SAMPLE_REL=$(head -n 1 "$TRAIN_CSV")
DATA_PATH="$HSI_DATASET_ROOT/patches/$SAMPLE_REL"
[ -f "$DATA_PATH" ] || { echo "ERROR: sample not found: $DATA_PATH"; exit 1; }
VID=$(basename "$(dirname "$DATA_PATH")")          # patch id, used for naming
echo "First sample : $DATA_PATH"
echo "VID          : $VID"
echo "Epochs       : $EPOCHS"

# ---- train -----------------------------------------------------------------
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
    --overwrite

# ---- collect the results into JSON -----------------------------------------
RESULTS_JSON="$PROJECT_DIR/results/first_sample_${SLURM_JOB_ID:-local}.json"
python - "$PROJECT_DIR/output/$OUTF" "$DATA_PATH" "$VID" "$EPOCHS" "$RESULTS_JSON" <<'PY'
import sys, glob, os, json
import numpy as np
import pandas as pd

out_dir, data_path, vid, epochs, out_json = sys.argv[1:6]

csvs = sorted(glob.glob(os.path.join(out_dir, "**", "epoch*.csv"), recursive=True),
              key=os.path.getmtime)
if not csvs:
    raise SystemExit(f"No epoch*.csv found under {out_dir}")
csv = csvs[-1]
row = pd.read_csv(csv).iloc[0].to_dict()

shp = np.load(data_path, mmap_mode="r").shape  # (B, H, W)
result = {
    "sample": {
        "data_path": data_path,
        "vid": vid,
        "num_bands": int(shp[0]),
        "image_hw": [int(shp[1]), int(shp[2])],
        "epochs": int(epochs),
    },
    "key_metrics": {
        "pred_seen_psnr_db": row.get("pred_seen_psnr"),
        "pred_seen_ssim": row.get("pred_seen_ssim"),
        "quant_seen_psnr_db": row.get("best_quant_seen_psnr"),
        "bits_per_pixel": row.get("bits/pixel"),
        "bits_per_param_with_overhead": row.get("bits/param w/ overhead"),
        "model_size_M_enc_dec_total": row.get("Size (M)"),
    },
    "source_csv": csv,
    "all_metrics": row,
}
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"Results written to {out_json}")
print(json.dumps(result["key_metrics"], indent=2, default=str))
PY

echo "DONE."
