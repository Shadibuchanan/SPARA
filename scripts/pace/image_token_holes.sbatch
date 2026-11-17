#!/bin/bash
# Gate 1 of docs/PERCEPTION_JEPA_SPEC.md, on CPU: do the image tokens show the holes?
# Reads the latent battery run's saved features; no model and no GPU.
#   sbatch -p ice-cpu scripts/pace/image_token_holes.sbatch
#SBATCH --job-name=spara-image-holes
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
RUN="${SPARA_BATTERY_RUN:-$ROOT/results/latent-battery-6848878}"
[ -f "$RUN/features-train.pt" ] || { echo "Missing battery features: $RUN" >&2; exit 1; }
OUT="$ROOT/results/image-token-holes-$SLURM_JOB_ID"
mkdir -p "$OUT"
"$ROOT/.venv/bin/python" scripts/probe_image_token_holes.py "$RUN" --output "$OUT/image-token-holes.json"
