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

## Where the project stands (2026-11-19)

The work since the baseline is summarised here; each row links to the document with the
design, the readings fixed before the run, and the evidence. Most results are from one
seed, and "no detectable difference" is not "equal". The sections after this one are the
original instructions for the baseline, text-only RL and Scaffolding Minds.

- **LIVR's latents add no measurable accuracy on the benchmark.** Direct fine-tuning
  without latents scores the same as LIVR.
- **RL from the LIVR checkpoint does not move the benchmark,** on text alone or with
  latent adjustments.
- **The limit is the route decision, not perception.** LIVR's failures are mostly walks
  into holes, but its image tokens show every hole, and giving the latents more map
  content did not change routes.
- **Training on unseen detour maps is what improved routes.** Expert iteration and
  text-only RL on those maps (B′) each more than double success on held-out detour maps,
  but lose no-detour maps, so the benchmark does not improve.
- **Latent exploration has not beaten text-only RL** on held-out detour maps in three
  attempts (worse, then no detectable difference twice).
- **On real images (V\*), training with latents lowers perception accuracy** against the
  untuned model, and LIVR's Stage 1 bottleneck is what causes the drop.
- **On real images the answer does not use the latents:** leaving them out of the prompt
  changes no score. See [Does the answer use the latents?](#does-the-answer-use-the-latents)

### Baselines and controls

| Study | Result | Document |
| --- | --- | --- |
| LIVR baseline, 10+10 epochs | 323/400 (80.75%); a rerun scored 328/400. LIVR's paper reports 66% | [results](docs/LIVR_BASELINE_RESULTS.md) |
| Audit of the gap to the paper | No scoring or answer-leakage error found; the cause of the gap is unknown | [audit](docs/LIVR_REPRODUCTION_AUDIT.md) |
| Direct SFT control (no latents) | 320/400 continuous, 323/400 staged: no difference from LIVR detected | [results](docs/SFT_CONTROL_RESULTS.md), [ADR-0004](docs/adr/0004-reinstate-vanilla-sft-control.md) |
| Scaffolding Minds (helper-supervised reference) | First run 0/400 (the paper-literal latent loss collapsed the model). The two fixed variants score 191 and 223/400 at Stage 1; RL raises the better one to 270/400, with the gain in the language-model weights, not the latent adjustment | [specification](docs/SCAFFOLDING_BASELINE_SPEC.md#results-2026-10-27) |

### RL from the LIVR checkpoint

| Study | Result | Document |
| --- | --- | --- |
| Text-only RL (condition B) | 326/400 against 328/400 for its starting checkpoint: no measurable change | [results](docs/TEXT_RL_RESULTS.md) |
| Latent RL (condition C), learned adjustment at layer 16 | 323/400; not different from B (p = 0.63). The learned adjustment has no effect at test time | [results](docs/LATENT_RL_RESULTS.md) |

### What the latents hold, and where routes fail

| Study | Result | Document |
| --- | --- | --- |
| Latent battery (no training) | Swapping in another map's latents swaps the answer (351 of 400). Goal and start decode almost perfectly; holes only partly | [results](docs/LATENT_BATTERY_RESULTS.md) |
| Planning probe | Natural-sized changes to the latents fix 7 of 73 failed maps. The agent's cell decodes weakly where moves are written | [specification](docs/PLANNING_PROBE_SPEC.md#result) |
| Step grounding | Position became decodable (89% against 39%) but routes did not change: 326 against 328 | [specification](docs/STEP_GROUNDING_SPEC.md#result) |
| Policy diagnostics | A linear probe on frozen vision tokens reads holes at AUROC 1.00. Masking moves into holes at decode time ends hole deaths but scores 181/400. 90% of benchmark maps need no detour; training on that natural mix scores 305 and 311 of 400 | [synthesis](docs/POLICY_ROUTING_AND_REDUNDANCY_SYNTHESIS.md) |

### Detour learning and expert iteration

Held-out detour maps are generated maps plus the benchmark's own detour maps (240 in total).

| Study | Result | Document |
| --- | --- | --- |
| Detour learning plan and reach check | The plan for the rows below; the reach check's reading was withheld | [specification](docs/DETOUR_LEARNING_SPEC.md) |
| Detour scaling | Variety helps: fine-tuning on 2,000 distinct detour layouts solves 107 of 240; the same number of rows from 98 layouts solves 64; LIVR 42. No arm improves the benchmark | [specification](docs/DETOUR_SCALING_SPEC.md#result-jobs-69414686941471) |
| Guided search | With LIVR as the policy and only the final accept or reject, search solves 55 of 97 unseen detour maps in 64 calls | [specification](docs/GUIDED_SEARCH_SPEC.md) |
| JEPA judge (learned transition model) | It hurts search and misses three in four holes | [specification](docs/JEPA_JUDGE_SPEC.md#result-job-6951675) |
| Expert iteration, round 1 (E) | 88 against LIVR's 42 of 240; no detectable difference between found routes and labels | [specification](docs/EXPERT_ITERATION_SPEC.md#result-jobs-69618746961876) |
| Expert iteration, round 2 on 7 x 7 and 8 x 8 (E2, EB2) | E2 84 against E's 54 of 400. LIVR's bottleneck did not change what the latents hold. On the benchmark E2 scores 322 against 328, from losses on no-detour maps | [specification](docs/EXPERT_ITERATION_ROUND2_SPEC.md#result-jobs-69927986992800) |

### Perception objectives on the latents

All in the [latent JEPA specification](docs/PERCEPTION_JEPA_SPEC.md); scores are of 400 maps of 7 x 7 and 8 x 8.

| Arm | Result |
| --- | --- |
| Gate 1 | LIVR's image tokens show every hole (balanced accuracy 1.00 at every layer from 4 to 36) |
| Latent JEPA (EJ2), and with fixed targets (EJF) | Neither took: the hole probe did not move. EJ2 68 and EJF 76, against EB2's 78 |
| Region latents (EJR) | Did not take; the sixteen tokens stayed near-copies. 80 against 78 |
| Global + region latents with a block mask (EGR) | Took (block probe 0.934 against 0.689), and routes did not change: 75 against 78. The answer ignored the block latents |
| Global dropout (EGD) | The answer now reads the block latents (98 of 100 routes follow a donor's), and routes still do not improve: 78 |
| Global + patch latents, for images without a grid | Built here, first run on V\* (below) |

### Latent exploration against text-only RL

Greedy successes on the 240 held-out detour maps; LIVR solves 42.

| Arm | Result | Document |
| --- | --- | --- |
| B′ (text-only RL) against C′ (learned Gaussian heads, layer 16) | B′ 92 and 96, C′ 77 and 86: C′ worse (p = 0.0016) | [specification](docs/STRUCTURED_EXPLORATION_SPEC.md) |
| Latent diagnostics | C′ was not a clean test: its latent loss swamped the gradient and its heads could not move | [specification](docs/LATENT_DIAGNOSTICS_SPEC.md) |
| C″ (fixed noise at layer 24, no heads) | 584 against B′'s 566 of 1,440 over six seeds: no detectable difference (p = 0.72) | [specification](docs/LATENT_DIAGNOSTICS_SPEC.md) |
| Latent distillation (D: latent target; S: routes only) | D 80 and 78, S 79 and 80: no detectable difference (p = 0.87) | [specification](docs/LATENT_DISTILLATION_SPEC.md) |

### Perception on real images (V\*)

Three arms trained from the base model on 10,000 Visual-CoT rows, scored on V\*'s 191
questions at native image size, in the project's prompt format.

| Model | V\* correct, of 191 |
| --- | --- |
| Untuned Qwen2.5-VL-3B | 133 |
| Direct SFT | 131 |
| LIVR | 121 |
| Global + patch latents | 119 |
| LIVR without the Stage 1 bottleneck (ablation) | 134 |

- Removing the latents from the prompt leaves LIVR at 121, so the loss is in the weights.
- After training with latents, the answer position attends about half as much to the image
  as after direct SFT, and reaches the target less.
- See the [results](docs/PERCEPTION_VSTAR_RESULTS.md) and the
  [attention report](docs/PERCEPTION_VSTAR_ATTENTION.md).

### Does the answer use the latents?

**On real images, no: the answer does not depend on them.** Evidence from the V\* and
Visual-CoT runs ([results](docs/PERCEPTION_VSTAR_RESULTS.md),
[attention report](docs/PERCEPTION_VSTAR_ATTENTION.md)):

| Test | LIVR | Global + patch latents |
| --- | --- | --- |
| V\* as trained, of 191 | 121 | 119 |
| V\* with the latent tokens left out of the prompt | 121 | 119 |
| V\* answers that change when they are left out | 13 | 8 |
| Held-out Visual-CoT rows containing the answer, with / without latents, of 354 | 223 / 225 | 231 / 232 |
| Patch model with its global latents shown to the answer | | 120 (3 answers change) |

- Accuracy is the same with and without the latents on both data sets, and both latent
  models stay below direct SFT (131) either way. Training with latents changed the
  language model's weights; the tokens themselves carry nothing the answer needs.
- **Attention is not the same as use.** The answer position puts more attention on the
  latents than on the whole image (LIVR: 10.3% on 4 latents, 4.6% on 4,472 image tokens),
  yet removing them changes nothing in the score.
- The latents do look at the target (3.0% of their image attention for LIVR, 6 to 9 times
  chance), so they pick up relevant content that the answer then does not need.

**On the FrozenLake benchmark the answer does read the latents, but they add no accuracy.**

- **Read:** given another map's latents, LIVR writes that map's route in 351 of 400 cases;
  with the latents dropped it solves 44 of 400 ([latent battery](docs/LATENT_BATTERY_RESULTS.md)).
- **No gain:** direct SFT without latents matches LIVR (323 against 323 of 400,
  [SFT control](docs/SFT_CONTROL_RESULTS.md)).
- **More content in the latents does not change routes.** With block latents that read
  their blocks at 0.934, the answer ignored them (93 of 100 routes unchanged when they
  were averaged). Once the answer was made to read them, routes still did not improve
  ([latent JEPA specification](docs/PERCEPTION_JEPA_SPEC.md)).
- **Learned latent adjustments have no effect at test time** (323 with, 324 without;
  [latent RL](docs/LATENT_RL_RESULTS.md)).

One observation bears on why the two cases differ, without settling it: after one masked
epoch on Visual-CoT, when the latents were the answer's only path to the image, LIVR
scored 68 of 191 on V\*, which is chance. Four latents did not carry what those questions
need, and the unmasked stage then learned to read the image directly. Whether a different
recipe would make the answer use the latents on real images is untested.

### Commands for the later studies

Each study has a `python -m spara.cli` command; its specification's "Running" section gives
the arguments, and `scripts/pace/` holds the matching job scripts.

| Command | Study |
| --- | --- |
| `battery`, `explore-probe`, `planning-probe` | Latent battery, exploration probe, planning probe |
| `detour-probe`, `detour-scaling`, `guided-search`, `jepa-judge` | Detour learning steps |
| `expert-iteration`, `expert-round2` | Expert iteration rounds 1 and 2 |
| `perception-jepa`, `region-latents` | Perception objectives on the latents |
| `fair-rl`, `latent-diagnostics`, `latent-distillation` | B′, C′ and C″; diagnostics; latent distillation |
| `perception`, `vstar` | Visual-CoT training arms; V\* evaluation and attention |

Terms are defined in [GLOSSARY.md](GLOSSARY.md). Run outputs are kept on PACE scratch, not
in git; each document names its run folders.

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

## Scaffolding Minds baseline (helper-supervised reference)

`src/spara/scaffolding/` adapts [Scaffolding Minds v2](docs/related_literature/Scaffolding_Minds_2608.19669v2.pdf)
to the LIVR baseline's data, backbone, evaluator and decoding settings. Unlike
LIVR, it trains with helper images: a value-function heatmap per map
([ADR-0003](docs/adr/0003-scaffolding-minds-helper-supervised-reference.md)).
Report it separately from LIVR and SPARA. The recipe is
[configs/scaffolding.json](configs/scaffolding.json). Method mapping, recorded
assumptions and checks are in [the specification](docs/SCAFFOLDING_BASELINE_SPEC.md).
The first PACE run (job 6257067, logs in `spara_results/`) scored 0/400 because
the paper-literal latent loss collapsed the language model. The default recipe is
now variant B (a zero-initialized linear latent projection plus MSE), and
`configs/scaffolding-mse.json` is variant A (MSE only). At Stage 1, B scored
223/400 and A 191/400; RL then raised B to 270/400. See
[the first run](docs/SCAFFOLDING_BASELINE_SPEC.md#first-run-job-6257067) and
[the results](docs/SCAFFOLDING_BASELINE_SPEC.md#results-2026-10-27).

On a GPU machine, one script runs the whole baseline. It trains all three
phases and evaluates both the Stage 1 and the Stage 2 checkpoints, writing logs
to `RUN/logs/`:

```bash
bash scripts/run_scaffolding.sh --model M --run R          # full baseline
bash scripts/run_scaffolding.sh --model M --run P --pilot  # bounded pilot
```

Rerun with the same `--run` to continue: completed phases are skipped. A phase
that stopped part-way is left untouched unless you pass `--retry-incomplete`.
`--phases` selects a subset, `--scaffold-from DIR` reuses another run's completed
Scaffolding Phase, and `--help` lists the remaining options. RL starts only if
Stage 1 scored at least `--rl-floor` (default 5%), and it stops itself after five
consecutive steps in which every rollout fails, recording why in `stage2-rl/stopped.json`. The script
wraps these CLI calls:

```bash
python -m spara.scaffolding.cli train --phase scaffold --base-model M --output S1A
python -m spara.scaffolding.cli train --phase predict --base-model M --encoder S1A/final --output S1B
python -m spara.scaffolding.cli evaluate --checkpoint S1B/final --output EVAL1
python -m spara.scaffolding.cli train --phase rl --checkpoint S1B/final --output S2
python -m spara.scaffolding.cli evaluate --checkpoint S2/final --output EVAL2
```

On PACE, submit `bash scripts/pace/submit-scaffolding.sh pilot` (or `baseline`)
with the same account and GPU variables as the LIVR pilot, plus `SPARA_MINUTES`.
Only do this after the budget is agreed. The baseline job runs the script above
and passes through `SPARA_RUN` (continue an earlier run), `SPARA_CONFIG`,
`SPARA_PHASES`, `SPARA_SCAFFOLD_FROM` and `SPARA_RL_FLOOR` from the submitting shell.

To do all of that in one step from a PACE login node — prepare the `$ROOT/code`
layout, run `setup.sh`, run the CPU tests, then submit a single job that runs the
pilot and the full baseline back to back — use:

```bash
bash scripts/pace/run-all-scaffolding.sh                 # pilot, then baseline
bash scripts/pace/run-all-scaffolding.sh --no-baseline   # stop after the pilot
bash scripts/pace/run-all-scaffolding.sh --continue DIR  # continue an earlier run
bash scripts/pace/run-all-scaffolding.sh --dry-run       # print the plan only
```

Both phases share one allocation (`scripts/pace/scaffolding-all.sbatch`), so the
baseline cannot lose the node to a second queue wait. Settings live in
[`scripts/pace/pace.conf`](scripts/pace/pace.conf): edit the GPU and walltime there
and they persist. Anything set in the environment overrides the file, and
`--config` points at a different one. The partition is always `ice-gpu`. Walltime is
`PILOT_HOURS + BASELINE_HOURS` whole hours, 16 by default, unless `HOURS`
overrides it. Account and QoS are left to your default Slurm
association, and `ROOT` defaults to `$HOME/scratch/spara` with `code/` linked to this checkout. A failing pilot stops the job before the baseline, but nobody sees the
pilot's numbers first, so the full-run budget review that
[the specification](docs/SCAFFOLDING_BASELINE_SPEC.md) asks for means running
`--no-baseline`, then rerunning with `--continue` once the numbers check out.
