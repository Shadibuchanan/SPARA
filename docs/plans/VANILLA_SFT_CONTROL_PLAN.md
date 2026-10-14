# Plan: vanilla SFT control (Run 1)

Train Qwen2.5-VL-3B-Instruct on the released VSP planning split with **no latent
tokens and no Stage-1 bottleneck**, score it with the pinned reference evaluator,
and report it beside LIVR job `6189457` (323/400 = 80.75%). The LIVR number is
currently uninterpretable: there is no run without the latent machinery to compare
against. This control is that comparison, and it also tests the simplest untested
explanation for the 14.75-point gap to LIVR's published 66% — that this split is
easy enough for plain SFT to reach a similar score.

Two arms, both at 2,500 optimizer updates:

- **A** — one stage, 20 epochs, one cosine schedule. The headline run.
- **C** — two 10-epoch segments with a fresh optimizer and cosine at the midpoint,
  matching LIVR's schedule shape. Isolates the LR restart from the latents.

## Fixed setup

Match LIVR `6189457` exactly: revision `66285546d2b821cf421d4f5eb2576359d3770cd3`,
seed 42, the same train/test split, `freeze_vision`, FP32 parameters with BF16
autocast and fused AdamW, SDPA, gradient checkpointing, the `configs/livr.json`
optimizer and schedule values, and `do_sample` with temperature 0.7, top-p 0.9,
1,024 new tokens — one sample per test record, all 400 in original order, no
test-based checkpoint selection.

Precision is held identical on purpose. The audit's BF16-parameter hypothesis for
the 66-vs-80.75 gap is a separate diagnostic; folding it in here would confound
this comparison.

## Resources

Ceiling 10 GPU-hours, ~4.2 h estimated. Run inside an interactive allocation; no
sbatch.

Request one H200 or H100 80 GB, 64 GB host RAM, 4–8 CPUs. LIVR peaked at 55.2 GiB
of GPU memory, leaving ~24 GiB headroom on an 80 GB card. Peak host use should be
20–25 GB — ~15 GB to materialize FP32 weights on load, ~15 GB plus a shard buffer
when `save_pretrained` returns the state dict to CPU, 3–5 GB of interpreter and
CUDA context, with load and save never overlapping. On an H100, budget ~10–30%
more wall-clock for the lower memory bandwidth, about 5 h for both arms. Record
the card in the run manifest and the report: LIVR ran on an H200, so an H100 is a
hardware difference, in the same noise class as seed variance. A and C share a
card, so the restart measurement is unaffected.

Checkpoints store weights only, ~15 GB each. A single-stage run never reloads
optimizer state, and LIVR's stage transition hands over weights with a fresh
optimizer, which is exactly C's segment boundary. `checkpoint_every_steps` stays
unset. Three checkpoints plus model and data need ~52 GB against 111 GB free.

## Why 2,500 updates

LIVR's claim is that latent tokens buy accuracy, so the control must not also
differ in how much supervised learning the backbone received. Equal optimizer
updates holds that fixed, and since the recipe and data are unchanged it
simultaneously equalises epochs (20 vs 20), examples seen, and — because dropping
four tokens from a ~1,900-token sequence changes length by under 0.3% — tokens
seen and wall-clock. `epochs_per_stage: 20` at batch 1 × 8 accumulation gives
125 steps/epoch × 20 = 2,500, exactly LIVR's 1,250 + 1,250.

Rejected alternatives: 1,250 updates / 10 epochs matches one LIVR stage but halves
the training, so a loss could not be attributed to the latents rather than to
undertraining. Matching wall-clock or tokens seen lands within ~0.3% of A anyway
while giving up the exact match on updates.

A and C are both kept because the risk from the schedule restart is asymmetric. If
vanilla SFT matches LIVR, the schedule is irrelevant — SFT got there with no
restart to help it. If it comes in below, the restart and the latents are
confounded, and C is the run that separates them. Running both makes a below-LIVR
result from A interpretable immediately.

## Shuffle order: `order_cycle`

`train_stage` seeds each epoch's permutation as `seed + epoch`, where `epoch` is
the **within-stage** index. LIVR therefore drew 10 permutations and used each
twice: P42…P51 in Stage 1, then P42…P51 again in Stage 2. A plain 20-epoch run
would draw 20 distinct permutations, P42…P61 — same records, same 20 passes, but
different groupings of eight into each optimizer step.

Seed instead on `seed + (epoch % order_cycle)` with `order_cycle: 10`, so a
20-epoch run replays P42…P51 twice and **optimizer step k sees exactly the same
eight examples as LIVR's step k**, for all 2,500 steps. Default `order_cycle` to
`epochs_per_stage` so the LIVR path is byte-identical.

Two reasons this is worth one line. LIVR's order reuse is an artifact of this
reimplementation restarting the epoch index per stage, not a paper setting — but
the comparison is against run `6189457`, so `6189457`'s actual behaviour is what
"matched" means. And C inherits that order sequence automatically from its
restarting epoch index, so without `order_cycle` an A-vs-C comparison would differ
in both the restart and the data order, and would no longer isolate the restart.

## Do not touch

- `configs/livr.json`, and LIVR's runtime behaviour through any code path. Every
  existing test must pass unchanged, and `configs/livr.json`'s SHA-256 must still
  match `docs/experiments/livr-6189457/provenance.json`.
- `docs/experiments/livr-6189457/` — the committed LIVR evidence. Do not refetch
  the LIVR checkpoints; this bundle is the comparison source.
- The reference evaluator, and any tuning against test results.

Keep new code and comments concise, in the style of the surrounding modules.
Log the work in `AGENT_USAGE.MD` per `CLAUDE.md`; no GitHub ticket for this control.

## Tasks

T1–T4 are CPU-only and can run in any order. T0 is CPU-only but gates T6 onward.
No GPU work starts until the user hands over an allocation, and T6 waits on the
T5 pilot being shown to the user.

**T0 — Environment.** This account has no venv, data or model weights. Create a
run root, link the checkout as `code/`, and run setup (~8 GB download):

```bash
export SPARA_REMOTE_ROOT=$HOME/scratch/sft-run
mkdir -p "$SPARA_REMOTE_ROOT" && ln -s /storage/ice1/5/6/psomu3/SPARA "$SPARA_REMOTE_ROOT/code"
scripts/pace/setup.sh
```

Check: `prepare-data` reports the 1,000/400 split with 7 shared layouts over 8
test rows; `fetch-model`'s `download-manifest.json` records the pinned revision.

**T1 — ADR-0004.** Reinstate this control. ADR-0001 accepted it on 2026-10-11 and
the user withdrew it the same day; ADR-0002 superseded it to prioritise the LIVR
reproduction. The project proposal's Oct 5–14 block still commits to "vanilla SFT
& pause tokens w/ images", so the repo and the proposal disagree — resolve it in
the ADR rather than quietly reinstating. ADR-0004 supersedes ADR-0002 on this
point only; leave the rest of ADR-0002 in force. Update the status lines of
ADR-0001 and ADR-0002 to point at it. Pause/filler tokens stay out of scope.

**T2 — Configs.** Copy `configs/livr.json` and add keys; do not edit the original.

- `configs/sft.json` (A): `"latent_tokens": 0, "epochs_per_stage": 20, "order_cycle": 10, "segments": 1`
- `configs/sft-staged.json` (C): `"latent_tokens": 0, "epochs_per_stage": 10, "segments": 2`

New keys read through `config.get` with LIVR-preserving defaults: `latent_tokens`
4, `order_cycle` `epochs_per_stage`, `segments` 2.

**T3 — Gate the latent machinery.** Five edits, all LIVR-neutral:

1. `serialization.encode` — add a `latents=LATENT_TOKENS` parameter and append
   `"".join(latents)` instead of the constant. Keep the existing guard that
   released text contains no latent tokens. Callers pass
   `model.config.livr_latent_tokens`.
2. `runtime.load_model` — when `latent_tokens == 0`, skip `initialize_latents` and
   instead set `livr_latent_ids = []`, `livr_latent_tokens = []`, `livr_stage = 2`,
   and `livr_serialization = "user-end,assistant-header-no-newline,answer,im-end"`.
   Skip the checkpoint branch's latent-ID comparison in that case.
3. `model.LivrQwen.forward` — make the bottleneck condition
   `livr_stage == 1 and self.config.livr_latent_ids`, so a segment labelled 1 with
   no latents uses the standard mask. With LIVR's non-empty IDs this is identical
   to today, which keeps every existing stage guard in `cli.py` working for C
   unchanged.
4. `training.train_stage` — seed with
   `config["seed"] + (epoch % config.get("order_cycle", config["epochs_per_stage"]))`,
   and pass `model.config.livr_latent_tokens` to `encode`.
5. `cli.py` — allow `--stage 2` with no source when `segments == 1`. A is labelled
   stage 2 so the evaluation gate and `require_final_checkpoint`'s 2,500-step
   expectation both work unchanged; with no latents there is no stage-1 mask to
   skip. No other guard changes.

`runtime.manifest` then records `latent_ids: []` on its own.

**T4 — Tests.** Match the existing `tests/` folder in kind and depth — a handful
of small, targeted tests on tiny models or real assets behind the `integration`
marker. Do not build out a broad suite; two tests cover this change. Full `pytest`
green with every existing LIVR and Scaffolding test unmodified, plus:

- `tests/test_sft_control.py`, integration-marked like `test_serialization.py`:
  the SFT prefix is the LIVR prefix **minus exactly the four latent IDs**
  `[151665, 151666, 151667, 151668]` — deleting those four positions from the LIVR
  IDs yields the SFT IDs, prefix length differs by exactly 4, the 121 image tokens
  and `image_grid_thw == [[1, 22, 22]]` are unchanged, the label boundary is
  unchanged, the answer still decodes to `text_output + "<|im_end|>"`, and the
  decoded prefix ends with `<|im_start|>assistant` and contains no latent token.
  Removing four tokens shifts sequence length and RoPE positions; this test is
  what establishes the control is "LIVR minus latents" and not also minus
  something else.
- A CPU test that `order_cycle: 10` over 20 epochs reproduces LIVR's permutation
  sequence, and that the LIVR config's effective seeding is unchanged.

**T5 — Pilot (~0.3 GPU-h).** Two optimizer steps and two evaluation records on the
real 3B. Check peak allocated GPU memory ≤ 60 GiB and that T4's prefix assertion
holds against the real processor and real weights. Record peak host RSS across a
load, a step and a checkpoint save. Show the user before continuing.

**T6 — Train A (~1.6 GPU-h).** `configs/sft.json`, `--stage 2`, from the fetched
base model. Check 2,500 sequential steps logged, all losses and grad norms finite,
final LR 0.

**T7 — Evaluate A (~0.25 GPU-h).** Final checkpoint, 400 records in original
order. Check 400 rows, each map ID once, and evaluator provenance hashes equal to
`6189457`'s.

**T8 — Train and evaluate C (~2.0 GPU-h).** `configs/sft-staged.json`, `--stage 1`
then `--stage 2 --checkpoint <segment1>/final`, then evaluate. Check 1,250 steps
per segment and that segment 2 starts from segment 1's weights with a fresh
optimizer.

**T9 — Report.** Fill the results table below. Report both the full 400 and the
392-row figure excluding the eight known train/test overlapping rows (test rows
38, 62, 109, 152, 269, 278, 344, 366; one-based — zero-based `row_index` 37, 61, 108, 151, 268, 277, 343, 365), as the LIVR baseline does. Commit evidence
bundles `docs/experiments/sft-a-<id>/` and `docs/experiments/sft-c-<id>/`
mirroring `docs/experiments/livr-6189457/`: all generated answers and outcomes,
aggregate counts, training logs, provenance, SHA-256 verified. Extend
`scripts/plot_livr_results.py` or add a sibling for the comparison chart.

## Stop conditions

- Stop before T6 if the pilot peaks above 70 GiB on an 80 GB card.
- Stop and report if the prefix assertion fails — the control would be "LIVR minus
  latents *and* something else", which is not the experiment.
- Training aborts on nonfinite loss (already enforced). Report; do not retry blind.
- Hard stop at 10 cumulative GPU-hours, counting retries. No recipe change or
  extra arm without the user's approval.
- **If vanilla SFT lands within ±2 points of 80.75%, that is the headline result.**
  Report it as: on this split, at one seed each, the latent tokens contribute no
  measurable accuracy, and the gap to LIVR's published 66% is not attributable to
  the latent machinery. Do not retune or add runs to rescue LIVR's margin. State
  the limit — n=1 seed per arm, and the binomial 95% interval on 400 trials at
  0.8075 is about ±3.9 points, so ±2 points is inside noise: the claim is "no
  difference detected", not "equal". A seed-variance study is the next ask and is
  out of scope here.
- If SFT is well below (under ~70%), that is consistent with the latents helping,
  at one seed, and still does not explain the paper gap.
- If SFT is well above 80.75%, the split and recipe rather than the method drive
  the score. Report it; do not change the evaluator.
- Three arms at n=1: read only differences larger than ±3.9 points as real, and
  commit to that before seeing any number.

## Results

| Method | Optimizer updates | 400 rows | 392 rows | Source |
| --- | ---: | ---: | ---: | --- |
| LIVR, K=4, 10+10 epochs | 2,500 | 323/400 = 80.75% | 315/392 = 80.36% | job `6189457` |
| **Vanilla SFT (A), no latents, 20 epochs** | **2,500** | 320/400 = 80.00% | 312/392 = 79.59% | job `6259811`, [results](../SFT_CONTROL_RESULTS.md) |
| **Vanilla SFT (C), no latents, 10+10 epochs** | **2,500** | 323/400 = 80.75% | 315/392 = 80.36% | job `6259811`, [results](../SFT_CONTROL_RESULTS.md) |
| LIVR, published 3B | — | 66% | — | LIVR v2 §4.2 |
| Mirage, published 3B | — | 46% | — | LIVR v2 §4.2 |
| Zero-shot 3B | — | 6% | — | LIVR v2 §4.2 |

## Reference

Steps: 125/epoch, 1,250/10 epochs, 2,500 total. Latent IDs 151665–151668. One
test image is 121 image tokens, `image_grid_thw == [[1, 22, 22]]`. LIVR serialization
marker: `user-end,4-latents,assistant-header-no-newline,answer,im-end`. `6189457`
took 5,664 GPU-s for 10+10 epochs plus evaluation on one H200 and peaked at
55.2 GiB. Background: `docs/LIVR_BASELINE_RESULTS.md`,
`docs/LIVR_REPRODUCTION_AUDIT.md`, `docs/research/LIVR_DATASET_COMPARABILITY.md`,
`docs/agents/pace-ice.md`.
