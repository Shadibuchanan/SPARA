#!/bin/bash
# Guided search with LIVR's policy (docs/GUIDED_SEARCH_SPEC.md) on the 97 unseen detour maps
# (test and fresh/probe), with the uniform prior alongside. It measures what step 1a left for
# the model: the adapter's agreement with detour probe 6937009 and the departures from LIVR's
# choices. Not scheduled; submit only after approval. Requires SPARA_LIVR_CHECKPOINT.
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
POINTS="${SPARA_PROBE_POINTS:-$ROOT/results/detour-probe-6937009/points.jsonl}"
[ -f "$POINTS" ] || { echo "Missing detour-probe points: $POINTS" >&2; exit 1; }
RUN="$ROOT/results/guided-search-$SLURM_JOB_ID"
mkdir -p "$ROOT/results"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -m spara.cli guided-search --config configs/livr.json --data-root "$DATA" \
  --checkpoint "$CHECKPOINT" --probe-points "$POINTS" --fresh-split fresh/probe --budget 64 --output "$RUN"
