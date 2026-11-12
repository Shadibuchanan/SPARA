#!/bin/bash
# Latent RL (condition C) from the final LIVR checkpoint. Submit only within the agreed
# GPU budget; this script does not submit itself. Requires SPARA_LIVR_CHECKPOINT,
# e.g. $ROOT/results/baseline-6253300/stage2/final (condition B's starting point).
#   SPARA_STEP=probe-pilot  latent-probe on 100 training maps, then a step-limited pilot
#                           (SPARA_RL_MAX_STEPS, default 10) at the sigma the probe's rule
#                           chooses; stops after the probe if no sigma qualifies.
#                           SPARA_PROBE_SIGMAS / SPARA_PROBE_DRAWS override the probe grid.
#   SPARA_STEP=full         one epoch with the recipe's sigma (or SPARA_SIGMA_INIT), final
#                           weights only, then the benchmark with sampled and zero adjustment.
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
STEP="${SPARA_STEP:?set SPARA_STEP to probe-pilot or full}"
COMMON=(--config configs/livr.json --rl-config configs/latent_rl.json --data-root "$ROOT/data/vsp"
        --checkpoint "$CHECKPOINT")
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case "$STEP" in
  probe-pilot)
    RUN="$ROOT/results/latent-rl-pilot-$SLURM_JOB_ID"
    mkdir -p "$RUN"
    "$PYTHON" -m pip freeze > "$RUN/environment.txt"
    "$PYTHON" -m spara.cli latent-probe "${COMMON[@]}" --output "$RUN/probe" \
      ${SPARA_PROBE_SIGMAS:+--sigmas "$SPARA_PROBE_SIGMAS"} ${SPARA_PROBE_DRAWS:+--draws "$SPARA_PROBE_DRAWS"}
    SIGMA=$("$PYTHON" -c "import json,sys; s=json.load(open(sys.argv[1]))['chosen_sigma']; print('' if s is None else s)" "$RUN/probe/summary.json")
    [[ -n "$SIGMA" ]] || { echo "No probed sigma met the rule; pilot skipped"; exit 0; }
    echo "chosen sigma $SIGMA"
    "$PYTHON" -m spara.cli train-rl "${COMMON[@]}" --output "$RUN/latent-rl" --sigma-init "$SIGMA" \
      --max-steps "${SPARA_RL_MAX_STEPS:-10}"
    ;;
  full)
    RUN="$ROOT/results/latent-rl-$SLURM_JOB_ID"
    mkdir -p "$RUN"
    "$PYTHON" -m pip freeze > "$RUN/environment.txt"
    "$PYTHON" -m spara.cli train-rl "${COMMON[@]}" --output "$RUN/latent-rl" --final-weights-only \
      --evaluate-latent-modes ${SPARA_SIGMA_INIT:+--sigma-init "$SPARA_SIGMA_INIT"}
    ;;
  *) echo "error: unknown SPARA_STEP $STEP" >&2; exit 1 ;;
esac
