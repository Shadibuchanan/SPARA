#!/bin/bash
# B′ and C′ in one allocation (docs/STRUCTURED_EXPLORATION_SPEC.md, 2026-11-17 amendment):
# the pilot on GPU 0, then, if its summary says proceed, the four runs (b and c x seeds 42 and 43)
# on the GPUs given, in waves: all four at once on 4 GPUs, b/c pairs on 2, one after another on 1.
# Each stage is scripts/pace/fair_rl.sbatch, run as a plain script, so the outputs and checks are
# the same as separate jobs. One submission instead of five avoids queueing again after the pilot.
# Resources are given at submission: 1-4 GPUs on one node, about 3 CPUs and 80 GB of memory per GPU.
#   SPARA_LIVR_CHECKPOINT  the LIVR rerun's final Stage 2 weights (required)
#   SPARA_PYTHON           interpreter (default $ROOT/.venv/bin/python)
#   SPARA_PILOT_RUN        an existing pilot's output directory, to skip the pilot
set -euo pipefail
ROOT="${SPARA_REMOTE_ROOT:-$HOME/scratch/SPARA}"
STAGE_SCRIPT="$ROOT/code/scripts/pace/fair_rl.sbatch"
: "${SPARA_LIVR_CHECKPOINT:?set SPARA_LIVR_CHECKPOINT to the final Stage 2 checkpoint of the LIVR rerun}"
mkdir -p "$ROOT/logs"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
# Guard for submissions without a GPU type: training peaked at about 70 GiB (B and C), so stop on a
# smaller GPU or a MIG slice (whose card reports its full memory). SPARA_MIN_GPU_MIB overrides.
# With SPARA_LOW_MEMORY=1 (AdamW state on the CPU, bfloat16 KL reference; about 35 GiB) a 44 GB+
# slice or GPU is enough.
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
SLICE=$(nvidia-smi -L | grep -oE 'MIG [0-9]+g\.[0-9]+gb' | grep -oE '[0-9]+gb' | tr -d gb | head -1)
[ -n "$SLICE" ] && MEM=$(( SLICE * 1000 ))
MIN="${SPARA_MIN_GPU_MIB:-79000}"
[ "${SPARA_LOW_MEMORY:-}" = 1 ] && MIN="${SPARA_MIN_GPU_MIB:-44000}"
if [ "${MEM:-0}" -lt "$MIN" ]; then
  echo "== guard failed: ${MEM:-no} MiB (MIG slice: ${SLICE:-none}) below $MIN" >&2; exit 3
fi
echo "== guard passed (${MEM} MiB)"
# Count GPUs, not lines: on a MIG slice `nvidia-smi -L` also prints the slice under its card.
GPUS="${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | grep -c '^GPU ')}"
[ "$GPUS" -ge 1 ] || { echo "No GPU visible" >&2; exit 1; }
[ "$GPUS" -gt 4 ] && GPUS=4
export SPARA_THREADS=$(( ${SLURM_CPUS_PER_TASK:-4} / GPUS ))
[ "$SPARA_THREADS" -ge 1 ] || SPARA_THREADS=1

PILOT="${SPARA_PILOT_RUN:-}"
if [ -z "$PILOT" ]; then
  echo "== pilot on GPU 0 ($(date))"
  CUDA_VISIBLE_DEVICES=0 SPARA_STAGE=pilot bash "$STAGE_SCRIPT"
  PILOT="$ROOT/results/fair-pilot-$SLURM_JOB_ID"
fi
"${SPARA_PYTHON:-$ROOT/.venv/bin/python}" -c "import json,sys; s=json.load(open(sys.argv[1])); sys.exit(0 if s['proceed'] else 'The pilot stopped: ' + '; '.join(s['rules']['stop']))" "$PILOT/summary.json"

echo "== four runs from $PILOT on $GPUS GPU(s), in waves of $GPUS ($(date))"
RUNS=("b 42" "c 42" "b 43" "c 43")
status=0
for ((start = 0; start < ${#RUNS[@]}; start += GPUS)); do
  pids=(); names=()
  for ((gpu = 0; gpu < GPUS && start + gpu < ${#RUNS[@]}; gpu++)); do
    read -r arm seed <<< "${RUNS[$((start + gpu))]}"
    log="$ROOT/logs/fair-$arm-seed$seed-$SLURM_JOB_ID.log"
    echo "start: $arm-seed$seed on GPU $gpu ($(date)), log $log"
    CUDA_VISIBLE_DEVICES=$gpu SPARA_STAGE=train SPARA_ARM=$arm SPARA_SEED=$seed SPARA_PILOT_RUN="$PILOT" \
      bash "$STAGE_SCRIPT" > "$log" 2>&1 &
    pids+=($!); names+=("$arm-seed$seed")
  done
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then echo "done: ${names[$i]} ($(date))"; else echo "FAILED: ${names[$i]}" >&2; status=1; fi
  done
done
echo "== finished ($(date))"
exit $status
