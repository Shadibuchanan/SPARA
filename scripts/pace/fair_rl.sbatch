#!/bin/bash
# B′ and C′ under fair conditions (docs/STRUCTURED_EXPLORATION_SPEC.md, 2026-11-17 amendment).
# Resources are given at submission (one H200 per job). This script does not submit itself.
#   SPARA_STAGE=pilot  on the LIVR rerun: set checks, the basis, the sampler check, the share of
#                      groups with mixed rewards (it sets the maps per step), C′'s scale, and L's
#                      greedy answers. About 30 minutes. Read its summary.json before any run.
#   SPARA_STAGE=train  SPARA_ARM=b, c or n (C″), SPARA_SEED (42 or 43), SPARA_PILOT_RUN: one run, then
#                      its evaluation in the same job. About 1.5 hours. Final weights only (15 GB).
#                      C″ also needs SPARA_NOISE_BASIS (docs/LATENT_DIAGNOSTICS_SPEC.md, C″ amendment).
# Both need SPARA_LIVR_CHECKPOINT, e.g. $ROOT/results/baseline-6253300/stage2/final.
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SPARA_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export PYTHONUNBUFFERED=1
PYTHON="${SPARA_PYTHON:-$ROOT/.venv/bin/python}"
DATA="$ROOT/data/vsp"
CHECKPOINT="${SPARA_LIVR_CHECKPOINT:?set SPARA_LIVR_CHECKPOINT to the final Stage 2 checkpoint of the LIVR rerun}"
STAGE="${SPARA_STAGE:?set SPARA_STAGE to pilot or train}"
SETTINGS="${SPARA_SETTINGS:-configs/fair_rl.json}"
for split in detour-pool detour-heldout detour-heldout-78; do
  [ -f "$DATA/fresh/$split/maps.jsonl" ] || { echo "Missing fresh set: $split" >&2; exit 1; }
done
mkdir -p "$ROOT/results"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case "$STAGE" in
  pilot)
    RUN="$ROOT/results/fair-pilot-$SLURM_JOB_ID"
    "$PYTHON" -m spara.cli fair-rl --stage pilot --settings "$SETTINGS" --config configs/livr.json \
      --data-root "$DATA" --checkpoint "$CHECKPOINT" --output "$RUN"
    ;;
  train)
    ARM="${SPARA_ARM:?set SPARA_ARM to b, c or n}"
    [ "$ARM" != n ] || [ -n "${SPARA_NOISE_BASIS:-}" ] || { echo "C″ needs SPARA_NOISE_BASIS" >&2; exit 1; }
    SEED="${SPARA_SEED:?set SPARA_SEED to 42 or 43}"
    PILOT="${SPARA_PILOT_RUN:?set SPARA_PILOT_RUN to the output directory of the pilot job}"
    [ -f "$PILOT/summary.json" ] || { echo "Missing pilot summary: $PILOT/summary.json" >&2; exit 1; }
    "$PYTHON" -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s['proceed'] else 'The pilot stopped: ' + '; '.join(s['rules']['stop']))" "$PILOT/summary.json"
    # One run per arm and seed per pilot: the first job to claim it runs it, any other skips it, so
    # separate jobs can take runs off fair_rl_all.sbatch's sequence and run them in parallel.
    mkdir -p "$PILOT/claims"
    if ! mkdir "$PILOT/claims/$ARM-seed$SEED" 2>/dev/null; then
      echo "skip: $ARM-seed$SEED already claimed by job $(cat "$PILOT/claims/$ARM-seed$SEED/job" 2>/dev/null)"; exit 0
    fi
    echo "$SLURM_JOB_ID" > "$PILOT/claims/$ARM-seed$SEED/job"
    RUN="$ROOT/results/fair-$ARM-seed$SEED-$SLURM_JOB_ID"
    "$PYTHON" -m spara.cli fair-rl --stage train --arm "$ARM" --seed "$SEED" --pilot-run "$PILOT" \
      --settings "$SETTINGS" --config configs/livr.json --data-root "$DATA" --checkpoint "$CHECKPOINT" --output "$RUN" \
      ${SPARA_LOW_MEMORY:+--low-memory} ${SPARA_NOISE_BASIS:+--noise-basis "$SPARA_NOISE_BASIS"}
    ;;
  *)
    echo "SPARA_STAGE must be pilot or train" >&2
    exit 1
    ;;
esac
