#!/bin/bash
# Submit only after the full-run resource budget is agreed. This script does not submit itself.
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
RUN="$ROOT/results/baseline-$SLURM_JOB_ID"
mkdir -p "$ROOT/results"
mkdir "$RUN"
PYTHON="$ROOT/.venv/bin/python"
# Optional, e.g. --final-weights-only when scratch space is limited.
SPARA_TRAIN_FLAGS="${SPARA_TRAIN_FLAGS:-}"
nvidia-smi
"$PYTHON" -m pip freeze > "$RUN/environment.txt"
"$PYTHON" -m spara.cli train --stage 1 --config configs/livr.json \
  --data-root "$ROOT/data/vsp" --base-model "$ROOT/models/Qwen2.5-VL-3B-Instruct" \
  --output "$RUN/stage1" $SPARA_TRAIN_FLAGS
"$PYTHON" -m spara.cli train --stage 2 --config configs/livr.json \
  --data-root "$ROOT/data/vsp" --checkpoint "$RUN/stage1/final" \
  --output "$RUN/stage2" $SPARA_TRAIN_FLAGS
"$PYTHON" -m spara.cli evaluate --config configs/livr.json \
  --data-root "$ROOT/data/vsp" --checkpoint "$RUN/stage2/final" \
  --output "$RUN/evaluation"
