#!/bin/bash
# Detour scaling check (docs/DETOUR_SCALING_SPEC.md), one arm per job: SPARA_ARM is livr, n98,
# n500 or n2000. Fine-tunes the LIVR rerun's final Stage 2 weights (except livr), then runs the
# greedy evaluation and, for fine-tuned arms, the reference benchmark. Submit only within the
# agreed GPU budget; this script does not submit itself. Requires SPARA_LIVR_CHECKPOINT.
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
PYTHON="$ROOT/.venv/bin/python"
CHECKPOINT="${SPARA_LIVR_CHECKPOINT:?set SPARA_LIVR_CHECKPOINT to the final Stage 2 checkpoint of the LIVR rerun}"
ARM="${SPARA_ARM:?set SPARA_ARM to livr, n98, n500 or n2000}"
DATA="$ROOT/data/vsp"
for split in detour-heldout detour-pool probe; do
  [ -f "$DATA/fresh/$split/maps.jsonl" ] || { echo "Missing fresh set: $split" >&2; exit 1; }
done
RUN="$ROOT/results/detour-scaling-$ARM-$SLURM_JOB_ID"
mkdir -p "$ROOT/results"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -m spara.cli detour-scaling --arm "$ARM" --recipe configs/detour_scaling.json \
  --config configs/livr.json --data-root "$DATA" --checkpoint "$CHECKPOINT" --output "$RUN"
