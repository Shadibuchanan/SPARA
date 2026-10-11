# LIVR implementation and PACE pilot

Date: 2026-10-11. The implementation and pilot checks are complete. At the end
of this pilot, full training and benchmarking had not run. The subsequently
completed 10+10-epoch baseline scored 323/400 (80.75%); see the separate
[full-run results](LIVR_BASELINE_RESULTS.md). The measurements below remain
pilot evidence, not the full baseline result.

## Implemented and checked

- Hash-pinned Mirage training/test archives: 1,000/400 original records, original
  images and text, with seven shared layouts affecting eight test rows disclosed.
- The exact downloaded, hash-verified reference simulator and pinned
  `mathruler==0.1.0` extractor. Malformed/nested boxes, off-grid moves, holes,
  character filtering and walking beyond the goal have fixtures.
- Four distinct random latent tokens, frozen vision encoder/projector,
  trainable language backbone, answer/end-token loss, Stage 1 visual bottleneck
  and ordinary Stage 2 attention. Explicit multimodal positions precede masking.
- Final-checkpoint rules, fresh Stage 2 optimizer, and within-stage restoration
  of optimizer, schedule, iteration position and RNG. A CPU integration test
  resumes mid-epoch and matches uninterrupted weights exactly.
- Model/tokenizer persistence, intended gradients, frozen vision weights,
  tiny-subset learning, both stages' backward/optimizer steps, and generation.
- **21 tests pass locally and on PACE**, including real-data/tokenizer fixtures
  and tiny random Qwen architecture tests. Pilot checkpoints cannot be passed off
  as completed baseline checkpoints through the training/evaluation CLI.

The model revision is `66285546d2b821cf421d4f5eb2576359d3770cd3` for
`Qwen/Qwen2.5-VL-3B-Instruct`. The PACE environment uses Python 3.11.9,
PyTorch 2.6.0+cu124, Transformers 4.51.3, SDPA, FP32 parameters/AdamW states and
BF16 autocast. Its package freeze is [locked here](../requirements/pace-cu124.lock.txt).
The parameter manifest records 3,085,938,688 trainable language parameters and
668,684,288 frozen vision/projector parameters.

## Execution evidence

All jobs used one NVIDIA H200. Allocated GPU time, including tests, loading,
checkpoint I/O and failed checks, was **818 seconds (13 minutes 38 seconds)**,
within the approved 3,600-second ceiling. Environment/model/data preparation
used no GPU allocation. The first parallel Hugging Face download process was
killed; a streaming downloader completed and verified both weight-shard hashes.

| Slurm job | Outcome | Allocated seconds |
| --- | --- | ---: |
| 6188935 | Training/roundtrip passed; initial BF16 logit tolerance failed | 249 |
| 6189026 | Single-prediction reproduction of the same tolerance failure | 16 |
| 6189039 | FP32/BF16 comparison completed | 17 |
| 6189075 | Both stages ran; provisional BF16 KL threshold failed | 242 |
| 6189162 | Stock BF16 parity passed; provisional FP32 bound failed after Stage 2 updates | 149 |
| 6189260 | Continued from verified Stage 1 checkpoint; final checks passed | 145 |

The successful continuation rechecked Stage 1 cache behavior and repeated Stage 2
with a fresh optimizer. Its summary explicitly links the earlier checkpoint and
Stage 1 measurements; it is not presented as a single uninterrupted successful
job. The final checkpoint roundtrip retained for diagnostics is a **Stage 1
tiny-subset pilot checkpoint**, not a trained baseline.

## Numerical validation findings

The first BF16 elementwise bound (`atol=0.15, rtol=0.02`) rejected 28 vocabulary
logits, despite matching next-token argmax. A later BF16 KL bound of 0.001 and
FP32 absolute bound of 0.0002 also proved too tight on subsequent Stage 2 updates.
These failed attempts and their raw outputs are retained; no attention/mask bug
was established by these failures.

Final validation uses two independent checks: FP32 cached/uncached logits must
agree at `atol=0.001, rtol=0.0001`, and Stage 2 must match **stock Qwen exactly**
for both cached and uncached execution in both FP32 and BF16. Tiny-model tests
retain their tighter 1e-6 absolute bound. BF16 cached/uncached differences are
reported as numerical diagnostics rather than assigned an uncalibrated pass/fail
threshold. These tolerances are implementation choices, not recovered LIVR settings.

| Final check | Stage 1 | Stage 2 |
| --- | ---: | ---: |
| FP32 max absolute logit difference | 0.0000570 | 0.0000668 |
| BF16 max absolute logit difference | 0.2500 | 0.2695 |
| BF16 KL divergence | 1.84e-7 | 7.07e-5 |
| Cached/uncached argmax agrees | Yes | Yes |
| Stock Qwen exact parity, FP32 and BF16 | Custom mask; not applicable | Passed |

The same-seed pilot is not a promise of bitwise training reproducibility across
GPU executions. The report preserves numerical differences and software/hardware
details rather than treating approximate floating-point results as exact.

## Measurements and proposed full run

All 1,000 training records were tokenized without truncation. The selected
median-length record has 443 tokens; the longest has 534. The pilot performs
eight accumulated microbatches per optimizer step. Typical examples are repeated
within a measured step, so these measurements omit per-example input preparation
and do not establish a full-epoch throughput benchmark.

Peak allocated GPU memory was about **55.2 GiB**, with about **57.0 GiB reserved**.
Use an 80 GB or larger compatible GPU; 48 GB cards are not supported by this
measured full-fine-tuning configuration. Typical steady steps were around two
seconds. Cold first-step costs produce a conservative **2.77 GPU-hour**
extrapolation for the 2,500 optimizer steps across both stages; warmed steps imply
less time. Neither estimate includes evaluation, queue delay or all I/O costs.

A resumable checkpoint occupies **39.72 GB (36.99 GiB)** and took **57.5 seconds**
to save in the retained run. Two final stage checkpoints need about 74 GiB.

**Subsequently approved by the user on 2026-10-11:** one H200, four CPUs, 96 GiB host RAM, with a
**four-hour allocated GPU ceiling** for the full 10+10 epochs and reference
evaluation. Use [baseline.sbatch](../scripts/pace/baseline.sbatch). Reassess the
time estimate if only slower hardware is available; do not silently substitute
LoRA, reduce images, change the split or shorten the recipe.

The two earlier failed attempts' model shards were SHA-256-identical to the
retained diagnostic checkpoint. Their duplicate model/optimizer payloads have
been removed; configurations, retirement records, logs and measurements remain.
One diagnostic checkpoint is retained. Recheck the 300 GiB scratch quota and reserve at least the roughly
74 GiB needed for both final checkpoints plus a working margin. Export final
checkpoints and results to this workstation's ignored `results/` or `checkpoints/`
directory and verify hashes; the workstation currently has about 209 GiB free.
This supplies a durable copy without assuming an unverified scratch-retention
policy. Full-run job `6189457` has now been submitted; its outcomes are tracked
in [the full-run record](LIVR_BASELINE_RESULTS.md).

## Artifacts and commands

Remote root: `~/scratch/SPARA`.

- `results/pilot-6189260/`: successful continuation summary, manifest, cache
  checks, Stage 2 generation and combined measurements.
- `results/pilot-6189162/`: Stage 1 learning/gradient/roundtrip evidence and the
  retained diagnostic checkpoint in `roundtrip/`.
- `logs/pilot-budget.json`: cumulative allocation accounting.
- `logs/pilot-*.log`, `logs/cache-*.log`: original attempts and numerical probes.

Small evidence files and logs are copied locally under
`results/pace-pilot-artifacts/` (ignored by Git); the source implementation,
configuration, tests and this report remain in the working tree, uncommitted.
See [README](../README.md) for installation, preparation, training, resuming and
reference-evaluation commands. The diagnostic probe is
`scripts/pace/check_cache.py`; the successful continuation uses
`spara pilot --continue-from <earlier-pilot-directory>` with the same recipe.
