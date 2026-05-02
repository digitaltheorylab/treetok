#!/usr/bin/env zsh

set -euo pipefail
trap 'echo -e "\nInterrupted. Exiting..."; exit 130' INT

OUTDIR="data/"
CLF="model.json"

typeset -A MODELS=(
  answerdotai/ModernBERT-base    bert.parquet
  allenai/OLMo-2-7B-1124         olmo.parquet
  google/gemma-4-E4B             gemma.parquet
  mistralai/Mistral-7B-v0.3      mistral.parquet
  Qwen/Qwen2.5-7B                qwen.parquet
)

for model filename in "${(@kv)MODELS}"; do
  python -m treetok build-dataset "$model" -o "${OUTDIR}${filename}" \
    --n-positives 2500 --n-hard-negatives 10000 --n-easy-negatives 2500
done

python -m treetok train ${OUTDIR}*.parquet -o "${OUTDIR}${CLF}" \
  --val-size 0.5 --merge-threshold-floor 0.92

