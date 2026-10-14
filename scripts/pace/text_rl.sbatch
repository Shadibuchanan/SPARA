#!/bin/bash
# Text-only RL (condition B) from the final LIVR checkpoint. Submit only after its
# resource budget is agreed; this script does not submit itself.
# Requires SPARA_LIVR_CHECKPOINT, e.g. $ROOT/results/baseline-6189457/stage2/final.
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
RUN="$ROOT/results/text-rl${SPARA_RL_MAX_STEPS:+-pilot}-$SLURM_JOB_ID"
mkdir -p "$ROOT/results"
mkdir "$RUN"
PYTHON="$ROOT/.venv/bin/python"
# Optional, e.g. --final-weights-only when scratch space is limited.
SPARA_TRAIN_FLAGS="${SPARA_TRAIN_FLAGS:-}"
nvidia-smi
"$PYTHON" -m pip freeze > "$RUN/environment.txt"
"$PYTHON" -m spara.cli train-rl --config configs/livr.json --rl-config configs/text_rl.json \
  --data-root "$ROOT/data/vsp" --checkpoint "$SPARA_LIVR_CHECKPOINT" \
  --output "$RUN/text-rl" $SPARA_TRAIN_FLAGS ${SPARA_RL_MAX_STEPS:+--max-steps "$SPARA_RL_MAX_STEPS"}
# A step-limited pilot measures cost and reward signal only; it is not evaluated.
[[ -n "${SPARA_RL_MAX_STEPS:-}" ]] && exit 0
"$PYTHON" -m spara.cli evaluate --config configs/livr.json --rl-config configs/text_rl.json \
  --data-root "$ROOT/data/vsp" --checkpoint "$RUN/text-rl/final" \
  --output "$RUN/evaluation"
