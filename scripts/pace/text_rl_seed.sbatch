#!/bin/bash
# One text-only RL seed from a fixed LIVR checkpoint, evaluated in memory without
# saving weights. Requires SPARA_LIVR_CHECKPOINT and SPARA_RL_SEED. Submit only
# within an agreed budget; this script does not submit itself.
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
RUN="$ROOT/results/text-rl-seed$SPARA_RL_SEED-$SLURM_JOB_ID"
mkdir -p "$ROOT/results"
mkdir "$RUN"
PYTHON="$ROOT/.venv/bin/python"
nvidia-smi
"$PYTHON" -m pip freeze > "$RUN/environment.txt"
"$PYTHON" -m spara.cli train-rl --config configs/livr.json --rl-config configs/text_rl.json \
  --data-root "$ROOT/data/vsp" --checkpoint "$SPARA_LIVR_CHECKPOINT" \
  --rl-seed "$SPARA_RL_SEED" --evaluate-in-memory "${SPARA_EVAL_SAMPLES:-16}" \
  --output "$RUN/text-rl"
