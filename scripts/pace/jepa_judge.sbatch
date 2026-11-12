#!/bin/bash
# JEPA judge (docs/JEPA_JUDGE_SPEC.md): states from the frozen LIVR rerun on generated
# maps, the JEPA and its comparisons, and three search arms on the 97 unseen detour maps.
# Submit only within the agreed GPU budget; this script does not submit itself.
# Requires SPARA_LIVR_CHECKPOINT; SPARA_PROBE_POINTS defaults to step 0's points.
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
POINTS="${SPARA_PROBE_POINTS:-$ROOT/results/detour-probe-6937009/points.jsonl}"
DATA="$ROOT/data/vsp"
for split in jepa-train detour-heldout probe; do
  [ -f "$DATA/fresh/$split/maps.jsonl" ] || { echo "Missing fresh set: $split" >&2; exit 1; }
done
[ -f "$POINTS" ] || { echo "Missing probe points: $POINTS" >&2; exit 1; }
RUN="$ROOT/results/jepa-judge-$SLURM_JOB_ID"
mkdir -p "$ROOT/results"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -m spara.cli jepa-judge --config configs/livr.json --settings configs/jepa_judge.json \
  --data-root "$DATA" --checkpoint "$CHECKPOINT" --probe-points "$POINTS" --output "$RUN"
