#!/bin/bash
# Phase 0 latent battery on a final LIVR Stage 2 checkpoint (SPARA_CHECKPOINT):
# probe features for every train and test map, then greedy answers for each test
# map under unmasked/bottleneck attention with own, swapped, mean and removed
# latents. This script does not submit itself; pass resources and the log path
# to sbatch. SPARA_LIMIT runs a pilot on the first rows of each grid size.
#   -> ${SPARA_OUTPUT:-$ROOT/results/latent-battery-$SLURM_JOB_ID}
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
PYTHON="$ROOT/.venv/bin/python"
CHECKPOINT="${SPARA_CHECKPOINT:?set SPARA_CHECKPOINT to a final LIVR Stage 2 checkpoint}"
[[ -f "$CHECKPOINT/complete.json" ]] || { echo "error: no completed checkpoint at $CHECKPOINT" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -m spara.cli battery --config configs/livr.json --data-root "$ROOT/data/vsp" \
  --checkpoint "$CHECKPOINT" --output "${SPARA_OUTPUT:-$ROOT/results/latent-battery-$SLURM_JOB_ID}" \
  ${SPARA_LIMIT:+--limit "$SPARA_LIMIT"}
