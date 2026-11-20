#!/bin/bash
#SBATCH --job-name=spara-probe
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.log

set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
source "$ROOT/.venv/bin/activate"

mkdir -p "$ROOT/results/probe_holes"

echo "=== Running Phase 1A: Hole Probing on Train + Test maps ==="
python scripts/probe_holes_from_image_tokens.py \
  --data-root "$ROOT/data/vsp" \
  --model-path "$ROOT/models/Qwen2.5-VL-3B-Instruct" \
  --save-features "$ROOT/results/probe_holes/cell_features.npz" \
  --output "$ROOT/results/probe_holes/results.json" \
  --split both \
  --device cuda

echo "Probe completed successfully!"
