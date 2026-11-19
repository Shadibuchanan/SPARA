#!/bin/bash
# Perception benchmarks (docs/PERCEPTION_BENCHMARKS_SPEC.md): one arm (SPARA_ARM: base, livr, patch or
# scaffold) trained on the 8k Zebra-CoT rows and scored on MMVP, V* and BLINK val. SPARA_LIMIT=N runs a
# pilot on the first N rows of every set. One H200; resources and the time limit are given at submission.
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
ARM="${SPARA_ARM:?set SPARA_ARM to base, livr, patch or scaffold}"
DATA="$ROOT/data/perception"
BASE="${SPARA_BASE_MODEL:-$ROOT/models/Qwen2.5-VL-3B-Instruct}"
for split in zebra8k mmvp vstar blink; do
  [ -f "$DATA/$split/rows.jsonl" ] || { echo "Missing perception split: $split" >&2; exit 1; }
done
[ -f "$BASE/download-manifest.json" ] || { echo "Missing base model: $BASE" >&2; exit 1; }
LIMIT=()
NAME="perception-$ARM"
if [ -n "${SPARA_LIMIT:-}" ]; then LIMIT=(--limit "$SPARA_LIMIT"); NAME="perception-pilot-$ARM"; fi
mkdir -p "$ROOT/results"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$ROOT/.venv/bin/python" -m spara.perception --arm "$ARM" --data-root "$DATA" --base-model "$BASE" \
  "${LIMIT[@]}" --output "$ROOT/results/$NAME-$SLURM_JOB_ID"
