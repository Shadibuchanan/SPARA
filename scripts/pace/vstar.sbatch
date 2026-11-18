#!/bin/bash
# V* (spara.vstar): greedy answers to the 191 released questions, one folder per prompt format and
# image-token limit. Without SPARA_CHECKPOINT it reads the base model as released, with no latent tokens.
# One GPU; resources are given at submission. Fetch the questions first, on the login node:
#   .venv/bin/python -m spara.cli vstar --stage fetch --data-root data/vstar
# Submit only within the agreed GPU budget; this script does not submit itself.
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
PYTHON="$ROOT/.venv/bin/python"
DATA="${SPARA_VSTAR_DATA:-$ROOT/data/vstar}"
BASE="$ROOT/models/Qwen2.5-VL-3B-Instruct"
FORMATS="${SPARA_VSTAR_FORMATS:-chat,livr}"
TOKENS="${SPARA_VSTAR_TOKENS:-16384,1280}"
[ -f "$DATA/manifest.json" ] || { echo "Missing V* questions: $DATA" >&2; exit 1; }
[ -f "$BASE/download-manifest.json" ] || { echo "Missing base model: $BASE" >&2; exit 1; }
SOURCE=()
if [ -n "${SPARA_CHECKPOINT:-}" ]; then
  [ -f "$SPARA_CHECKPOINT/complete.json" ] || { echo "Missing weights: $SPARA_CHECKPOINT" >&2; exit 1; }
  SOURCE=(--checkpoint "$SPARA_CHECKPOINT")
fi
mkdir -p "$ROOT/results"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -m spara.cli vstar --config configs/livr.json --data-root "$DATA" --base-model "$BASE" \
  --formats "$FORMATS" --max-image-tokens "$TOKENS" ${SPARA_VSTAR_LIMIT:+--limit "$SPARA_VSTAR_LIMIT"} \
  "${SOURCE[@]}" --output "$ROOT/results/vstar-${SPARA_VSTAR_NAME:-base}-$SLURM_JOB_ID"
