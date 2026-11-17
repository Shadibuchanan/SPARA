#!/bin/bash
# Region latents (docs/PERCEPTION_JEPA_SPEC.md): EJR ("Region latents: EJR", the default) or, with
# SPARA_REGION_SETTINGS=configs/region_latents_masked.json and SPARA_ARM=egr, EGR ("Global + region
# latents: EGR"). One H200; resources are given at submission. From E, with gate 3's fixed targets
# (SPARA_GATE_RUN, job 7118222's output): the start block, EB2's two phases with the region readout,
# round 2's evaluation, and the benchmark unless the settings drop it.
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
if [ "${SPARA_STAGE:-train}" = tracking ]; then
  # The planning probe's state probe on a region-latent run's weights (SPARA_REGION_RUN, e.g. EGD), unmasked
  # and masked with the globals hidden, on round 2's held-out and pool maps. Needs no gate run.
  RUN="${SPARA_REGION_RUN:?set SPARA_REGION_RUN to the run directory whose weights the probe reads}"
  [ -f "$RUN/phase2/final/complete.json" ] || { echo "Missing weights: $RUN/phase2/final" >&2; exit 1; }
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  "$PYTHON" -m spara.cli region-latents --stage tracking --config configs/livr.json --data-root "$ROOT/data/vsp" \
    --checkpoint "$RUN/phase2/final" --recorded-run "$RUN" --output "$ROOT/results/region-tracking-$SLURM_JOB_ID"
  exit 0
fi
E="${SPARA_E_CHECKPOINT:-$ROOT/results/expert-train-search-6961875/stage2/final}"
ROUND1_ROUTES="${SPARA_ROUND1_ROUTES:-$ROOT/results/expert-routes-6961874/routes.jsonl}"
SEARCH_RUN="${SPARA_SEARCH_RUN:-$ROOT/results/round2-search-6992798}"
GATE_RUN="${SPARA_GATE_RUN:?set SPARA_GATE_RUN to gate 3 output directory}"
SETTINGS="${SPARA_REGION_SETTINGS:-configs/region_latents.json}"
ARM="${SPARA_ARM:-ejr}"
[ -f "$SETTINGS" ] || { echo "Missing settings: $SETTINGS" >&2; exit 1; }
grep -q "\"arm\": \"$ARM\"" "$SETTINGS" || [ "$ARM" = ejr ] || { echo "Settings $SETTINGS are not for $ARM" >&2; exit 1; }
DATA="$ROOT/data/vsp"
[ -f "$E/complete.json" ] || { echo "Missing E's checkpoint: $E" >&2; exit 1; }
[ -f "$ROUND1_ROUTES" ] || { echo "Missing round 1 routes: $ROUND1_ROUTES" >&2; exit 1; }
[ -f "$SEARCH_RUN/routes.jsonl" ] || { echo "Missing round 2 routes: $SEARCH_RUN/routes.jsonl" >&2; exit 1; }
[ -f "$GATE_RUN/jepa_heads.pt" ] || { echo "Missing gate 3 heads: $GATE_RUN/jepa_heads.pt" >&2; exit 1; }
grep -q '"reading": "gates passed: run EJF"' "$GATE_RUN/summary.json" \
  || { echo "Gate 3 did not pass: $GATE_RUN/summary.json" >&2; exit 1; }
for split in detour-heldout-78 detour-pool-78 detour-heldout detour-pool; do
  [ -f "$DATA/fresh/$split/maps.jsonl" ] || { echo "Missing fresh set: $split" >&2; exit 1; }
done
mkdir -p "$ROOT/results"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "${SPARA_STAGE:-train}" = dropout ]; then
  # EGD: global dropout from EGR's weights (SPARA_EGR_RUN), then its evaluation and swap test.
  EGR_RUN="${SPARA_EGR_RUN:?set SPARA_EGR_RUN to the EGR run directory}"
  [ -f "$EGR_RUN/phase2/final/complete.json" ] || { echo "Missing EGR weights: $EGR_RUN/phase2/final" >&2; exit 1; }
  "$PYTHON" -m spara.cli region-latents --stage dropout --config configs/livr.json --data-root "$DATA" \
    --checkpoint "$EGR_RUN/phase2/final" --settings configs/region_dropout.json --round1-routes "$ROUND1_ROUTES" \
    --search-run "$SEARCH_RUN" --output "$ROOT/results/region-egd-$SLURM_JOB_ID"
  exit 0
fi
if [ "${SPARA_STAGE:-train}" = swap ]; then
  # The swap test on a run's weights (SPARA_EGR_RUN): masked greedy with block latents swapped or averaged.
  # For EGD: SPARA_SWAP_SETTINGS=configs/region_swap_hidden.json SPARA_SWAP_RECORDED=<run>/masked_hidden.jsonl.
  EGR_RUN="${SPARA_EGR_RUN:?set SPARA_EGR_RUN to the run directory whose weights the test reads}"
  SWAP_SETTINGS="${SPARA_SWAP_SETTINGS:-configs/region_swap.json}"
  RECORDED="${SPARA_SWAP_RECORDED:-$EGR_RUN/masked.jsonl}"
  [ -f "$EGR_RUN/phase2/final/complete.json" ] || { echo "Missing weights: $EGR_RUN/phase2/final" >&2; exit 1; }
  [ -f "$RECORDED" ] || { echo "Missing recorded routes: $RECORDED" >&2; exit 1; }
  "$PYTHON" -m spara.cli region-latents --stage swap --config configs/livr.json --data-root "$DATA" \
    --checkpoint "$EGR_RUN/phase2/final" --settings "$SWAP_SETTINGS" --recorded-masked "$RECORDED" \
    --output "$ROOT/results/region-swap-$SLURM_JOB_ID"
  exit 0
fi
"$PYTHON" -m spara.cli region-latents --config configs/livr.json --data-root "$DATA" --checkpoint "$E" \
  --settings "$SETTINGS" --gate-run "$GATE_RUN" --round1-routes "$ROUND1_ROUTES" \
  --search-run "$SEARCH_RUN" --output "$ROOT/results/region-$ARM-$SLURM_JOB_ID"
