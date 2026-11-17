#!/bin/bash
# Gate 3 of the fixed-target retry (docs/PERCEPTION_JEPA_SPEC.md), on CPU: the fixed targets, the
# heads' warm start and the latent-dependence check, from the features job's saved files.
#   SPARA_FEATURES_RUN=<features output> sbatch -p ice-cpu scripts/pace/jepa_fixed_gate.sbatch
#SBATCH --job-name=spara-jepa-fixed-gate
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
FEATURES="${SPARA_FEATURES_RUN:?set SPARA_FEATURES_RUN to the features job output directory}"
[ -f "$FEATURES/features.pt" ] || { echo "Missing features: $FEATURES/features.pt" >&2; exit 1; }
"$ROOT/.venv/bin/python" scripts/fixed_target_gate.py "$FEATURES" "$ROOT/results/jepa-fixed-gate-$SLURM_JOB_ID" \
  --settings configs/perception_jepa_fixed.json
