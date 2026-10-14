# SPARA

Spatial Planning with Adaptive Representation Adjustment. The first phase
reimplements LIVR's Qwen2.5-VL-3B planning baseline. See the accepted
[implementation specification](docs/LIVR_IMPLEMENTATION_SPEC.md) and
[project plan](docs/SPARA_PROJECT_PLAN.md).

The [PACE pilot report](docs/LIVR_PILOT_REPORT.md) records 21 passing tests,
successful actual-model checks and measured resources. The completed baseline
scored **323/400 (80.75%)** after 10+10 epochs, using 1 hour 34 minutes of H200
time. Saved responses were verified by local reference rescoring; the final
Stage 2 backup is still transferring. See [the full-run results](docs/LIVR_BASELINE_RESULTS.md)
for the comparison, assumptions and evidence.

![LIVR evaluation summary and per-grid results](docs/figures/livr-evaluation.png)

Training curves, reproducible chart commands and the per-example evidence are
in the [results report](docs/LIVR_BASELINE_RESULTS.md).

## Baseline commands

Python 3.11 is the tested environment. Install an appropriate PyTorch 2.6.0 /
torchvision 0.21.0 build, then `python -m pip install -e '.[model,test]'`.
On Windows, run these commands with `.venv/Scripts/python`; on Linux use
`.venv/bin/python`. The actual model runs belong on an allocated PACE GPU.

```bash
python -m spara.cli prepare-data --data-root data/vsp
python -m spara.cli fetch-model --output .cache/processor --processor-only
python -m pytest -q
```

Tests use a tiny random Qwen model for architectural checks and the real released
records/tokenizer for data and serialization checks. They do not download 3B
weights or establish a reproduced benchmark result. Integration tests skip if
the dataset/processor is absent; prepare both for a complete test run.

`prepare-data` verifies both archive hashes, preserves the original 1,000/400
records, and downloads the hash-pinned reference simulator. Scoring calls those
upstream functions and `mathruler==0.1.0`; there is no separate SPARA scorer.
The original split has seven shared layouts affecting eight test rows. Helper
images remain in the source archive but never enter model inputs.

## PACE pilot

From this workstation:

```bash
python scripts/pace/sync.py --target pace-ice
ssh pace-ice
```

On the cluster, run `bash -l ~/scratch/SPARA/code/scripts/pace/setup.sh` once.
It prepares the environment, data and model under scratch. `SPARA_REMOTE_ROOT`
changes that root; `--target` / `SPARA_SSH_TARGET` changes the personal SSH alias.
Do not run training on the login node.

Select compatible hardware and your account settings after inspecting Slurm:

```bash
export SPARA_ACCOUNT=<your-account> SPARA_QOS=<your-qos>
export SPARA_PARTITION=ice-gpu SPARA_GPU=gpu:h200:1
export SPARA_PILOT_MINUTES=40
bash ~/scratch/SPARA/code/scripts/pace/submit-pilot.sh
```

The H200 request is an example, not a team requirement. The pinned CUDA 12.4
stack targets Ampere/Hopper/Ada; Blackwell requires a separately verified stack.
Check actual VRAM and availability. The approved pilot ceiling is **one GPU-hour
total across attempts**, including tests and checkpoint I/O in the allocation.
Before retrying, use `sacct -j JOBID --format=JobID,State,ElapsedRaw,AllocTRES`
and subtract actual allocated time; do not submit overlapping pilots.

Pilot logs are `logs/pilot-JOBID.log`; artifacts are `results/pilot-JOBID/`.
The pilot verifies gradients, learning on one training example, checkpoint
round-trip, optimizer restoration/reset, and cached/uncached logits; it profiles
typical/longest examples in both stages. Its diagnostic checkpoint and greedy
outputs are **not** a completed baseline or benchmark evaluation.

## Full training and reference evaluation

The first full run is approved for one H200, four CPUs, 96 GiB RAM and at most
four allocated GPU-hours. For another run, settle its resource budget first. Run the
following inside the GPU job (paths are illustrative):

```bash
python -m spara.cli train --stage 1 --data-root /path/to/data/vsp \
  --base-model /path/to/models/Qwen2.5-VL-3B-Instruct --output /path/to/stage1
python -m spara.cli train --stage 2 --data-root /path/to/data/vsp \
  --checkpoint /path/to/stage1/final --output /path/to/stage2
python -m spara.cli evaluate --data-root /path/to/data/vsp \
  --checkpoint /path/to/stage2/final --output /path/to/evaluation
```

The recipe is [configs/livr.json](configs/livr.json): full language fine-tuning,
frozen vision/projector, four random learned tokens, 10 epochs per stage,
microbatch 1 / accumulation 8, AdamW 1e-5 and separate warmup/cosine schedules.
Stage 2 loads weights with a fresh optimizer. Within a stage, `--resume` restores
the saved optimizer, schedule, iteration position and random state. Only load
trusted training-state checkpoints. Optional `checkpoint_every_steps` writes
additional resumable checkpoints; it defaults to zero because full checkpoints
are large. Provision storage before enabling periodic saves.

Evaluation samples 400 records in original order with seed 42, temperature 0.7,
top-p 0.9 and max 1,024 new tokens, preserving inherited generation defaults.
It writes raw responses, extracted routes and reference outcomes, plus overall
and per-grid results. No test-based checkpoint selection is implemented.

## Text-only RL

`python -m spara.cli train-rl --checkpoint /path/to/stage2/final ...` runs GRPO
on route text from the final LIVR checkpoint, without latent adjustments. It is
condition B of the planned comparisons. Settings, sources, assumptions and open
issues are in the [text-only RL specification](docs/TEXT_RL_SPEC.md). One epoch
scored 326/400 (81.5%) from a rerun LIVR baseline at 328/400 (82.0%); see
[the results](docs/TEXT_RL_RESULTS.md).

## Reproduction assumptions

The public sources do not recover LIVR's exact model/package revisions. This
implementation pins an available Qwen revision and stock Transformers 4.51.3.
The four latents occur after the user-end marker and before the assistant header.
The header has no trailing newline, following Mirage evaluation; the released
answer and assistant-end token are supervised. Tokenization keeps the inference
prefix intact at the answer boundary. All old language embeddings/output-head
weights remain trainable. New rows use the model's normal initialization scale.

SDPA supports the explicit Stage 1 mask. Multimodal positions are computed from
the original sequence before applying that mask. FP32 parameters/AdamW states
with BF16 autocast, gradient checkpointing, max gradient norm 1.0 and standard
AdamW bias/norm decay exclusions are recorded engineering choices. Tiny-model
cache tests use tighter tolerances than actual-model tests. Actual-model cache
validation checks FP32 logits at `atol=1e-3, rtol=1e-4`. BF16 logit errors, KL
divergence and argmax agreement are reported as numerical diagnostics; Stage 2
must match stock Qwen's FP32 and BF16 cached and uncached paths exactly. The initial BF16
elementwise/distribution thresholds proved inappropriate for stock Qwen's kernel
roundoff. Retained diagnostics are reproducible with `scripts/pace/check_cache.py`.

Run manifests record source/data hashes, model revision, package versions,
trainable parameters, token IDs, preprocessing, hardware, decoding and lineage.
Keep checkpoints and results on scratch for execution, then export completed
artifacts to a durable destination before relying on scratch retention.

From the workstation, monitor an already submitted job and collect both final
checkpoints, logs and evaluation evidence with:

```bash
python scripts/pace/collect_run.py --job JOBID --output results/pace-baseline-JOBID
```

The collector reads cluster files, verifies local copies with SHA-256 and never
submits a GPU job. Keep it running until collection finishes. It accepts the
same `--target` and `--root` overrides as source deployment; restarting skips
files whose size and hash already match. `latest-status.json` reports progress
and `backup-verification.json` records verified files under the output directory.
After a successful job, it independently rescores the saved responses against
`--data-root` (default `data/vsp` in this project), checks complete training and
budget accounting, and writes `run-validation.json` and `RUN_REPORT.md`.
The local collector requires this workstation to remain running.
