#!/bin/bash
# Detour reach check (docs/DETOUR_LEARNING_SPEC.md, step 0) on the LIVR rerun's final Stage 2
# checkpoint, reading LIVR's recorded greedy routes (planning probe 6887458 for test maps,
# exploration probe 6884608 for fresh/probe). Submit only within the agreed GPU budget; this
# script does not submit itself. Requires SPARA_LIVR_CHECKPOINT.
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
DATA="$ROOT/data/vsp"
TEST_ROUTES="${SPARA_TEST_ROUTES:-$ROOT/results/planning-probe-6887458/probe/routes.jsonl}"
FRESH_ROUTES="${SPARA_FRESH_ROUTES:-$ROOT/results/explore-probe-6884608/probe/sweep.jsonl}"
for file in "$TEST_ROUTES" "$FRESH_ROUTES"; do
  [ -f "$file" ] || { echo "Missing greedy routes: $file" >&2; exit 1; }
done
RUN="$ROOT/results/detour-probe-$SLURM_JOB_ID"
mkdir -p "$ROOT/results"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -m spara.cli detour-probe --config configs/livr.json --rl-config configs/text_rl.json \
  --data-root "$DATA" --checkpoint "$CHECKPOINT" --test-routes "$TEST_ROUTES" \
  --fresh-routes "$FRESH_ROUTES" --fresh-split fresh/probe --output "$RUN"
