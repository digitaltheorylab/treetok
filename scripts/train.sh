#!/usr/bin/env zsh

set -euo pipefail
trap 'echo -e "\nInterrupted. Exiting..."; exit 130' INT

OUTDIR="data/"
CLF="model.json"

# Preferred training dataset composition
N_POS=5000
N_HARD=24000
N_EASY=2000
SEED=0

# Marker-variant semantics: "merge" (marker toggles are positives) or
# "separate" (marker toggles are hard negatives). Recorded in the datasets
# and picked up automatically by `treetok train`
MARKER_POLICY="merge"

# Preferred training hyperparameters
VAL_SIZE=0.5
TARGET_PRECISION=0.95
THRESHOLD_FLOOR=0.5
MERGE_TARGET_PRECISION=0.999
MERGE_THRESHOLD_FLOOR=0.5

typeset -A MODELS=(
  answerdotai/ModernBERT-base             modernbert.parquet
  allenai/Olmo-3-1025-7B                  olmo.parquet
  google/gemma-4-E4B                      gemma.parquet
  mistralai/Ministral-3-8B-Base-2512      mistral.parquet
  Qwen/Qwen3.5-9B                         qwen.parquet
  google-bert/bert-base-uncased           bert-wordpiece.parquet
  facebookAI/xlm-roberta-base             xlmr.parquet
)

for model filename in "${(@kv)MODELS}"; do
  python -m treetok build-dataset "$model" -o "${OUTDIR}${filename}" \
    --n-positives "$N_POS" \
    --n-hard-negatives "$N_HARD" \
    --n-easy-negatives "$N_EASY" \
    --seed "$SEED" \
    --marker-policy "$MARKER_POLICY"
done

python -m treetok train ${OUTDIR}*.parquet -o "${OUTDIR}${CLF}" \
  --val-size "$VAL_SIZE" \
  --seed "$SEED" \
  --target-precision "$TARGET_PRECISION" \
  --threshold-floor "$THRESHOLD_FLOOR" \
  --merge-target-precision "$MERGE_TARGET_PRECISION" \
  --merge-threshold-floor "$MERGE_THRESHOLD_FLOOR"
