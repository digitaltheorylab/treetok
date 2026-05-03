#!/usr/bin/env zsh

set -euo pipefail
trap 'echo -e "\nInterrupted. Exiting..."; exit 130' INT

OUTDIR="data/"
CLF="model.json"

# Preferred training dataset composition
N_POS=2500
N_HARD=24000
N_EASY=2000
SEED=0

# Preferred training hyperparameters
VAL_SIZE=0.5
TARGET_PRECISION=0.99
THRESHOLD_FLOOR=0.8
MERGE_TARGET_PRECISION=0.999
MERGE_THRESHOLD_FLOOR=0.95

typeset -A MODELS=(
  answerdotai/ModernBERT-base             bert.parquet
  allenai/Olmo-3-1025-7B                  olmo.parquet
  google/gemma-4-E4B                      gemma.parquet
  mistralai/Ministral-3-8B-Base-2512      mistral.parquet
  Qwen/Qwen3.5-9B                         qwen.parquet
)

for model filename in "${(@kv)MODELS}"; do
  python -m treetok build-dataset "$model" -o "${OUTDIR}${filename}" \
    --n-positives "$N_POS" \
    --n-hard-negatives "$N_HARD" \
    --n-easy-negatives "$N_EASY" \
    --seed "$SEED"
done

python -m treetok train ${OUTDIR}*.parquet -o "${OUTDIR}${CLF}" \
  --val-size "$VAL_SIZE" \
  --seed "$SEED" \
  --target-precision "$TARGET_PRECISION" \
  --threshold-floor "$THRESHOLD_FLOOR" \
  --merge-target-precision "$MERGE_TARGET_PRECISION" \
  --merge-threshold-floor "$MERGE_THRESHOLD_FLOOR"
