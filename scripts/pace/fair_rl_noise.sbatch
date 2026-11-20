#!/bin/bash
# C″ (docs/LATENT_DIAGNOSTICS_SPEC.md, C″ amendment) in one allocation: seeds 42 then 43 on one GPU,
# each a scripts/pace/fair_rl.sbatch train stage (run, then evaluation), from the B′/C′ pilot, so
# the outputs and checks are those of B′'s runs, which are reused as the baseline.
# Resources are given at submission: one GPU (a 48 GB Blackwell slice with SPARA_LOW_MEMORY=1).
#   SPARA_LIVR_CHECKPOINT  the LIVR rerun's final Stage 2 weights (required)
#   SPARA_PILOT_RUN        the B′/C′ pilot (required; fair-pilot-7142592)
#   SPARA_NOISE_BASIS      the confirmed basis: latent-diag-sweep-L16-24-7148362 (required)
#   SPARA_PYTHON, SPARA_LOW_MEMORY  as for fair_rl.sbatch
set -euo pipefail
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
STAGE_SCRIPT="$ROOT/code/scripts/pace/fair_rl.sbatch"
: "${SPARA_LIVR_CHECKPOINT:?set SPARA_LIVR_CHECKPOINT}" "${SPARA_PILOT_RUN:?set SPARA_PILOT_RUN}"
: "${SPARA_NOISE_BASIS:?set SPARA_NOISE_BASIS to the confirmed latent-diagnostics basis}"
mkdir -p "$ROOT/logs"
nvidia-smi -L
status=0
for seed in 42 43; do
  log="$ROOT/logs/fair-n-seed$seed-$SLURM_JOB_ID.log"
  echo "== start: n-seed$seed ($(date)), log $log"
  if SPARA_STAGE=train SPARA_ARM=n SPARA_SEED=$seed bash "$STAGE_SCRIPT" > "$log" 2>&1; then
    echo "== done: n-seed$seed ($(date))"
  else
    echo "== FAILED: n-seed$seed ($(date)); see $log" >&2; status=1; break
  fi
done
echo "== finished ($(date))"
exit $status
