#!/bin/bash
# Pilot and full baseline back to back inside ONE allocation, so the baseline never
# waits in the queue a second time. Submit with scripts/pace/run-all-scaffolding.sh;
# resource and account flags live there.
#
# Set SPARA_PILOT_ONLY=1 to stop after the pilot. Set SPARA_RUN to an earlier
# baseline run directory to continue it; completed phases are skipped.
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
PYTHON="$ROOT/.venv/bin/python"
PILOT="$ROOT/results/scaffolding-pilot-$SLURM_JOB_ID"

nvidia-smi
"$PYTHON" -m pip freeze > "$ROOT/logs/environment-$SLURM_JOB_ID.txt"
"$PYTHON" -m pytest -q

echo "=== pilot ==="
"$PYTHON" -m spara.scaffolding.cli pilot \
  --config configs/scaffolding.json --data-root "$ROOT/data/vsp" \
  --base-model "$ROOT/models/Qwen2.5-VL-3B-Instruct" --output "$PILOT"

if [[ "${SPARA_PILOT_ONLY:-0}" == "1" ]]; then
  echo "SPARA_PILOT_ONLY=1; stopping before the baseline."
  exit 0
fi

echo "=== baseline ==="
bash scripts/run_scaffolding.sh \
  --run "${SPARA_RUN:-$ROOT/results/scaffolding-$SLURM_JOB_ID}" \
  --python "$PYTHON" --data "$ROOT/data/vsp" \
  --model "$ROOT/models/Qwen2.5-VL-3B-Instruct"
