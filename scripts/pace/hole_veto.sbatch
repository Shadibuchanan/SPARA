#!/bin/bash
#SBATCH --job-name=spara-veto
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.log

set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
source "$ROOT/.venv/bin/activate"

CHECKPOINT="${1:-$ROOT/models/Qwen2.5-VL-3B-Instruct}"
OUTDIR="$ROOT/results/hole_veto"
mkdir -p "$OUTDIR"

echo "=== Running Phase 1B: Baseline Greedy Decode (No Veto) ==="
python scripts/hole_veto_decode.py \
  --checkpoint "$CHECKPOINT" \
  --data-root "$ROOT/data/vsp" \
  --output "$OUTDIR/baseline_greedy" \
  --no-veto \
  --split test

echo "=== Running Phase 1B: Hole-Veto Greedy Decode (Holes Blocked) ==="
python scripts/hole_veto_decode.py \
  --checkpoint "$CHECKPOINT" \
  --data-root "$ROOT/data/vsp" \
  --output "$OUTDIR/veto_greedy" \
  --split test

echo "Hole-veto evaluations completed!"
