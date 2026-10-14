#!/bin/bash
# Submit only after the full-run resource budget is agreed. This script does not submit itself.
# Set SPARA_RUN to an earlier run directory to continue it; completed phases are skipped.
# SPARA_CONFIG, SPARA_PHASES, SPARA_SCAFFOLD_FROM and SPARA_RL_FLOOR also pass through to
# run_scaffolding.sh.
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
bash scripts/run_scaffolding.sh \
  --run "${SPARA_RUN:-$ROOT/results/scaffolding-$SLURM_JOB_ID}" \
  --python "$ROOT/.venv/bin/python" --data "$ROOT/data/vsp" \
  --model "$ROOT/models/Qwen2.5-VL-3B-Instruct"
