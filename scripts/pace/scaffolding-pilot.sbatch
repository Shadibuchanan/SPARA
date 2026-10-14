#!/bin/bash
# Submit with scripts/pace/submit-scaffolding.sh pilot; resource/account flags live there.
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export SPARA_TEST_DATA="$ROOT/data/vsp"
export SPARA_TEST_PROCESSOR="$ROOT/models/Qwen2.5-VL-3B-Instruct"
nvidia-smi
"$ROOT/.venv/bin/python" -m pip freeze > "$ROOT/logs/environment-$SLURM_JOB_ID.txt"
"$ROOT/.venv/bin/python" -m pytest -q
"$ROOT/.venv/bin/python" -m spara.scaffolding.cli pilot \
  --config configs/scaffolding.json --data-root "$ROOT/data/vsp" \
  --base-model "$ROOT/models/Qwen2.5-VL-3B-Instruct" \
  --output "$ROOT/results/scaffolding-pilot-$SLURM_JOB_ID"
