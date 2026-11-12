#!/bin/bash
# Planning probe (docs/PLANNING_PROBE_SPEC.md) on the final LIVR checkpoint: greedy routes
# and per-move hidden states, the state-tracking probes, and the latent headroom test, in
# one model load. Submit only within the agreed GPU budget; this script does not submit
# itself. Requires SPARA_LIVR_CHECKPOINT, e.g. $ROOT/results/baseline-6253300/stage2/final.
# SPARA_BASIS defaults to the exploration probe's basis (job 6884608); SPARA_FRESH_SPLIT
# defaults to fresh/probe.
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
PYTHON="$ROOT/.venv/bin/python"
CHECKPOINT="${SPARA_LIVR_CHECKPOINT:?set SPARA_LIVR_CHECKPOINT to the final Stage 2 LIVR checkpoint}"
BASIS="${SPARA_BASIS:-$ROOT/results/explore-probe-6884608/probe/basis.pt}"
SPLIT="${SPARA_FRESH_SPLIT:-fresh/probe}"
RUN="$ROOT/results/planning-probe-$SLURM_JOB_ID"
mkdir -p "$RUN"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" -m pip freeze > "$RUN/environment.txt"
"$PYTHON" -m spara.cli planning-probe --config configs/livr.json --rl-config configs/latent_rl.json \
  --data-root "$ROOT/data/vsp" --checkpoint "$CHECKPOINT" --basis "$BASIS" --fresh-split "$SPLIT" \
  --output "$RUN/probe"
