#!/bin/bash
#SBATCH --job-name=hiner2d_s5
#SBATCH --output=logs/hiner2d_s5_%A_%a.out
#SBATCH --error=logs/hiner2d_s5_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-170:5%4
#
# Apply hiner2d to every 5th patch of the mini/train split (35 patches:
# indices 0,5,10,...,170), storing one results.json per patch.
# Mirrors the baseline runs (mini/train, stride 5) for a fair dataset-average.
#
# Submit:  sbatch slurm_hiner2d_stride5.sh
# Config:  enc_dim 64_16 (~3.0 bpppb) | 2000 epochs (~16k steps) | 8-bit quant

set -eo pipefail

export HSI_DATASET_ROOT=/faststorage/hyspecnet-11k
HINER2D_DIR=/faststorage/viktor/project_context/hsi-toolkit/hiner_forked/HINER/hiner2d
cd "$HINER2D_DIR"
mkdir -p logs

eval "$(micromamba shell hook --shell bash)"
micromamba activate hsi_toolkit
export PYTHONUNBUFFERED=1

IDX=${SLURM_ARRAY_TASK_ID}
# Resolve the patch path for this array index from the split (no hardcoding).
PATCH=$(python -c "from hsi_toolkit.data import load_split; print(load_split('mini','train')[${IDX}])")
VID=$(basename "$PATCH" | sed 's/-DATA.npy$//')

echo "[hiner2d_s5] task=${IDX} vid=${VID}"
echo "[hiner2d_s5] patch=${PATCH}"

python -u train_hiner2d.py --data_path "$PATCH" \
    --vid "$VID" --outf hiner2d_stride5 \
    --enc_dim 64_16 --fc_len 101 --dec_strds 2 --fc_dim 128 \
    --epochs 2000 --batchSize 2048 --lr 1e-3 --loss L2 \
    --quant_model_bit 8 --eval_freq 2000 --print_freq 500 \
    --manualSeed 42 --overwrite