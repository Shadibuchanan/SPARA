#!/bin/bash
# Diagnostic multi-sample evaluation; submit only within an agreed budget.
# Requires SPARA_EVAL_CHECKPOINTS, e.g. "base=/path/stage2/final rl=/path/text-rl/final".
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
mkdir -p "$ROOT/results"
args=()
for spec in $SPARA_EVAL_CHECKPOINTS; do args+=(--checkpoint "$spec"); done
"$ROOT/.venv/bin/python" scripts/pace/multisample_eval.py --data-root "$ROOT/data/vsp" \
  --samples "${SPARA_EVAL_SAMPLES:-16}" --output "$ROOT/results/multisample-$SLURM_JOB_ID" "${args[@]}"
