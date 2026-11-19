#!/bin/bash
# Latent-exploration diagnostics after C′ (docs/LATENT_DIAGNOSTICS_SPEC.md). No training.
# Resources are given at submission (one GPU per GPU part). This script does not submit itself.
#   SPARA_PART=sweep        SPARA_LAYERS (e.g. 16,24); SPARA_TEXT_BASELINE=1; SPARA_ANSWER_NOISE=1;
#                           SPARA_BASES_FROM=<earlier sweep> reuses its bases.pt (the confirmation)
# Any part: SPARA_SETTINGS=<settings file> (default configs/latent_diagnostics.json).
#   SPARA_PART=gradients    the gradient split on C′'s first batches, then final-verdict and latent search
#   SPARA_PART=checkpoints  SPARA_RUNS='{"livr": ..., "b42": ..., ...}' (final checkpoint directories)
#   SPARA_PART=weights      SPARA_RUNS as above; CPU only
# The GPU parts need SPARA_PILOT_RUN (the B′/C′ pilot) and, except checkpoints, SPARA_LIVR_CHECKPOINT.
# Each GPU part first runs a smoke pass (a few maps and draws) into <run>-smoke and stops if it fails.
# The analysis runs afterwards on the CPU: spara.cli latent-diagnostics --part analyse (see the spec).
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
PART="${SPARA_PART:?set SPARA_PART to sweep, gradients, checkpoints or weights}"
RUN="$ROOT/results/latent-diag-$PART${SPARA_LAYERS:+-L${SPARA_LAYERS//,/-}}-$SLURM_JOB_ID"
mkdir -p "$ROOT/results"
ARGS=(--part "$PART" --config configs/livr.json --data-root "$DATA" --settings "${SPARA_SETTINGS:-configs/latent_diagnostics.json}")
case "$PART" in
  sweep|gradients)
    ARGS+=(--checkpoint "${SPARA_LIVR_CHECKPOINT:?set SPARA_LIVR_CHECKPOINT}" --pilot-run "${SPARA_PILOT_RUN:?set SPARA_PILOT_RUN}")
    [ -n "${SPARA_LAYERS:-}" ] && ARGS+=(--layers "$SPARA_LAYERS")
    [ "${SPARA_TEXT_BASELINE:-}" = 1 ] && ARGS+=(--text-baseline)
    [ "${SPARA_ANSWER_NOISE:-}" = 1 ] && ARGS+=(--answer-noise)
    [ -n "${SPARA_BASES_FROM:-}" ] && ARGS+=(--bases-from "$SPARA_BASES_FROM")
    ;;
  checkpoints)
    ARGS+=(--pilot-run "${SPARA_PILOT_RUN:?set SPARA_PILOT_RUN}" --runs "${SPARA_RUNS:?set SPARA_RUNS}")
    ;;
  weights)
    ARGS+=(--runs "${SPARA_RUNS:?set SPARA_RUNS}")
    ;;
  *)
    echo "SPARA_PART must be sweep, gradients, checkpoints or weights" >&2; exit 1 ;;
esac
if [ "$PART" != weights ]; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  nvidia-smi -L
  echo "== smoke ($(date))"
  "$PYTHON" -m spara.cli latent-diagnostics "${ARGS[@]}" --output "$RUN-smoke" --smoke
  echo "== smoke passed ($(date))"
fi
echo "== full run: $RUN ($(date))"
"$PYTHON" -m spara.cli latent-diagnostics "${ARGS[@]}" --output "$RUN"
echo "== finished ($(date))"
