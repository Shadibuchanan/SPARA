#!/bin/bash
# Latent JEPA (docs/PERCEPTION_JEPA_SPEC.md). One H200 per stage; resources are given at submission.
#   SPARA_STAGE=gate: E's features of the training, pool and held-out maps; the image-token probe and
#     the target layer; centring means; the heads' warm start; the loss weight; the swap checks on E
#     and E2. Writes summary.json and jepa_heads.pt.
#   SPARA_STAGE=train: EJ2 from E with the gate's heads (SPARA_GATE_RUN), then round 2's evaluation
#     and the benchmark.
#   The fixed-target retry (EJF):
#   SPARA_STAGE=features: E's latents and layer-4 cells on the same maps, saved for gate 3, which runs
#     on CPU (scripts/pace/jepa_fixed_gate.sbatch).
#   SPARA_STAGE=train-fixed: EJF from E with gate 3's heads and targets (SPARA_GATE_RUN): the loss
#     weight's pilot rows, then as train.
# Submit only within the agreed GPU budget; this script does not submit itself.
set -euo pipefail
module load python/3.11.9
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
cd "$ROOT/code"
export HF_HOME="$ROOT/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
PYTHON="$ROOT/.venv/bin/python"
STAGE="${SPARA_STAGE:?set SPARA_STAGE to gate, train, features or train-fixed}"
E="${SPARA_E_CHECKPOINT:-$ROOT/results/expert-train-search-6961875/stage2/final}"
E2="${SPARA_E2_CHECKPOINT:-$ROOT/results/round2-e2-6992799/phase2/final}"
ROUND1_ROUTES="${SPARA_ROUND1_ROUTES:-$ROOT/results/expert-routes-6961874/routes.jsonl}"
SEARCH_RUN="${SPARA_SEARCH_RUN:-$ROOT/results/round2-search-6992798}"
DATA="$ROOT/data/vsp"
[ -f "$E/complete.json" ] || { echo "Missing E's checkpoint: $E" >&2; exit 1; }
[ -f "$ROUND1_ROUTES" ] || { echo "Missing round 1 routes: $ROUND1_ROUTES" >&2; exit 1; }
[ -f "$SEARCH_RUN/routes.jsonl" ] || { echo "Missing round 2 routes: $SEARCH_RUN/routes.jsonl" >&2; exit 1; }
for split in detour-heldout-78 detour-pool-78 detour-heldout detour-pool probe jepa-train; do
  [ -f "$DATA/fresh/$split/maps.jsonl" ] || { echo "Missing fresh set: $split" >&2; exit 1; }
done
mkdir -p "$ROOT/results"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
case "$STAGE" in
  gate)
    [ -f "$E2/complete.json" ] || { echo "Missing E2's checkpoint: $E2" >&2; exit 1; }
    "$PYTHON" -m spara.cli perception-jepa --stage gate --config configs/livr.json --data-root "$DATA" \
      --checkpoint "$E" --e2-checkpoint "$E2" --round1-routes "$ROUND1_ROUTES" --search-run "$SEARCH_RUN" \
      --output "$ROOT/results/jepa-gate-$SLURM_JOB_ID"
    ;;
  train)
    GATE_RUN="${SPARA_GATE_RUN:?set SPARA_GATE_RUN to the gate job output directory}"
    [ -f "$GATE_RUN/jepa_heads.pt" ] || { echo "Missing the gate's heads: $GATE_RUN/jepa_heads.pt" >&2; exit 1; }
    "$PYTHON" -m spara.cli perception-jepa --stage train --config configs/livr.json --data-root "$DATA" \
      --checkpoint "$E" --gate-run "$GATE_RUN" --round1-routes "$ROUND1_ROUTES" --search-run "$SEARCH_RUN" \
      --output "$ROOT/results/jepa-ej2-$SLURM_JOB_ID"
    ;;
  features)
    "$PYTHON" -m spara.cli perception-jepa --stage features --config configs/livr.json --data-root "$DATA" \
      --settings configs/perception_jepa_fixed.json --checkpoint "$E" --round1-routes "$ROUND1_ROUTES" \
      --search-run "$SEARCH_RUN" --output "$ROOT/results/jepa-features-$SLURM_JOB_ID"
    ;;
  train-fixed)
    GATE_RUN="${SPARA_GATE_RUN:?set SPARA_GATE_RUN to gate 3 output directory}"
    [ -f "$GATE_RUN/jepa_heads.pt" ] || { echo "Missing gate 3 heads: $GATE_RUN/jepa_heads.pt" >&2; exit 1; }
    grep -q '"reading": "gates passed: run EJF"' "$GATE_RUN/summary.json" \
      || { echo "Gate 3 did not pass: $GATE_RUN/summary.json" >&2; exit 1; }
    "$PYTHON" -m spara.cli perception-jepa --stage train --config configs/livr.json --data-root "$DATA" \
      --checkpoint "$E" --gate-run "$GATE_RUN" --round1-routes "$ROUND1_ROUTES" --search-run "$SEARCH_RUN" \
      --output "$ROOT/results/jepa-ejf-$SLURM_JOB_ID"
    ;;
  *)
    echo "Unknown SPARA_STAGE: $STAGE" >&2; exit 1
    ;;
esac
