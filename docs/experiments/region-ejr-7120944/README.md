# Evidence for region latents, EJR (job 7120944)

This is the arm in [the latent JEPA spec](../../PERCEPTION_JEPA_SPEC.md#region-latents-ejr).
- **The arm:** EJR is round 2's EB2, from E, with one latent token per 2 x 2 block of cells (4, 9 or
  16 by grid size). A readout loss trains each token to predict its own block's fixed targets.
- **The targets:** EJF's, from [gate 3, job 7118222](../jepa-fixed-gate-7117800/README.md).

## The run

- **Compute:** one H200, 1,832 GPU-seconds by Slurm's elapsed time (30 min 32 s; 1,724 s in the
  run). The allowance was 2,400; the time limit was 39 minutes.
- **Training:** phase 1 took 405 s and phase 2 432 s.
- **Source:** the hashes in manifest.json match the local code for every file the run imports. The
  mismatches are files it does not import: the scaffolding files, the PACE config and the package's
  `SOURCES.txt`.
- **Weights:** EJR's final weights (15 GB) were deleted from PACE scratch on 2026-11-16, at the
  user's request. The latent states, the readout and the logs remain on PACE
  (`$SPARA_REMOTE_ROOT/results/region-ejr-7120944/`) and in the git-ignored local `results/`.

**The start block** (summary.json):
- **The identity check passed.** On 8 phase-1 rows at 3 x 3 and 4 x 4, the final-layer states after
  adding the tokens equal E's exactly (difference 0.0). The answer loss moved by 3e-7.
- **Insertion disturbance:** on 8 rows at 5 x 5 and above, adding the tokens raised the answer loss
  by 0.041 before any training.
- **Targets added:** the 88 training maps the features job had merged into others.
- **The readout's warm start:** a ridge fit on 350 training maps (4,312 token states). Tokens 9 to
  15 were seen on the 201 maps at 7 x 7 and 8 x 8.
- **β = 0.25,** inside 0.01 to 100.

## Pre-declared readings

The output of `region_latents.compare_region` is in comparison.json.

**Validity: passed.** The start check, round 2's set checks and gate 3 all passed.

**Manipulation check (count-matched): did not take.** Each block's holes are read by ridge from one
token's state: EJR's own token for the block, EB2's token 0. Fitted on pool maps and scored on
held-out maps:

| | 7 x 7 | 8 x 8 | Mean |
| --- | --- | --- | --- |
| EB2, token 0 for every block | 0.699 | 0.679 | 0.689 |
| EJR, each block's own token | 0.687 | 0.647 | 0.667 |

- **EJR − EB2:** −0.021, 95% interval −0.030 to −0.013. The margin was +0.05.

**Primary: no reading on perception.** EJR against EB2 on the 400 held-out maps (greedy, exact
McNemar), reported as descriptive because the check failed:

| | Maps | EJR | EB2 | Only EJR | Only EB2 | p |
| --- | --- | --- | --- | --- | --- | --- |
| All held-out | 400 | 80 | 78 | 24 | 22 | 0.88 |

**Secondary: no detectable difference from E2.**

| | Maps | EJR | E2 | Only EJR | Only E2 | p |
| --- | --- | --- | --- | --- | --- | --- |
| All held-out | 400 | 80 | 84 | 16 | 20 | 0.62 |

## The latents: still copies, carrying slightly more together

**Round 2's all-tokens probe** (descriptive; extra tokens that carry nothing lower it):

| | E | E2 | EB2 | EJR |
| --- | --- | --- | --- | --- |
| Hole balanced accuracy, mean of 7 x 7 and 8 x 8 | 0.649 | 0.653 | 0.657 | 0.677 |

- **EJR − EB2:** +0.020, 95% interval 0.013 to 0.028 (paired bootstrap).
- **Tokens 0 to 3 alone:** 0.660.

**Do the tokens divide the map?** Post hoc, CPU (token-diagnosis.json). For each token, a ridge from
its state to every cell's hole, then per-cell correlation on held-out maps:

| | Tokens | Between-token similarity, 7 x 7 / 8 x 8 | Own block / other blocks, 7 x 7 | Own / other, 8 x 8 |
| --- | --- | --- | --- | --- |
| EB2 | 4 | 0.86 / 0.85 | | |
| EJR | 16 | 0.83 / 0.83 | 0.44 / 0.38 | 0.36 / 0.33 |

**The linear ceiling** to the fixed targets, from all of a model's latents (post hoc):

| Held-out, 7 x 7 / 8 x 8 | Share unexplained | Hole-direction share |
| --- | --- | --- |
| EB2 | 0.824 / 0.917 | 0.161 / 0.094 |
| EJR | 0.784 / 0.881 | 0.200 / 0.140 |

**The readout during training** (monitored, `training.jepa_error`):

| | Start | After phase 1 | After phase 2 |
| --- | --- | --- | --- |
| 32 validation maps: share unexplained | 0.887 | 0.918 | 0.895 |
| 32 validation maps: shuffled gap | 0.108 | 0.124 | 0.148 |

**Reading (interpretation):**
- **The tokens barely specialised.** The sixteen tokens remain near-copies. Each one leans only a
  little toward its own block.
- **Together, the latents hold slightly more of the map than EB2's.** The all-tokens probe rose
  0.020 and the linear ceiling improved, but no single token reads its block better than EB2's token
  0 does.
- **Routes did not change.**

## Descriptive

**First mistakes on failed held-out maps:**

| | E | E2 | EB2 | EJR |
| --- | --- | --- | --- | --- |
| Into an adjacent hole, a safe shortest move toward the goal existed | 96 | 79 | 80 | 92 |
| Into an adjacent hole, at a forced point | 130 | 124 | 128 | 114 |
| Safe, but off every shortest route | 91 | 91 | 91 | 93 |
| Stops short, wall bump or overrun | 29 | 22 | 23 | 21 |

The perception count rose, as for EJ2 (90) and EJF (91).

**Outcomes on the held-out maps:** EJR reached the goal by a shortest route on 76 maps and by a longer
one on 4. It fell into a hole on 303, and stopped short or passed the goal on 17.

**Participation ratio** of the latents at layer 12 (7 x 7, 8 x 8): EJR 29.0 and 27.3 (16 tokens),
against EB2's 22.6 and 21.4.

**Masked greedy** (only the latents see the image): EJR 65 of 400, against EB2's 73, E2's 66, EJF's 70
and EJ2's 59.

**Harm check on round 1's 240 held-out maps:** EJR 95 against E's 88 (p = 0.19). EB2 100, E2 102,
EJF 103.

**Benchmark** (one sampled answer per test map), against LIVR's 328:

| | All 400 | p | Detour (40) | p | No-detour (360) | p |
| --- | --- | --- | --- | --- | --- | --- |
| EJR | 317 | 0.13 | 14 against 9 | 0.13 | 303 against 319 | 0.011 |

## Files

| File | Contents |
| --- | --- |
| comparison.json | `region_latents.compare_region`: validity, the count-matched check with the all-tokens probe beside it, primary and secondary, descriptive readings, tokens 0 to 3 alone |
| token-diagnosis.json | Post hoc: between-token similarity, own-block against other-block information, the linear ceiling (EB2 and EJR) |
| summary.json | The start check, targets added, the warm start, the loss weight's pilot, training and monitored readout errors, evaluation counts, time |
| manifest.json | Run manifest; the PACE root is written as `$SPARA_REMOTE_ROOT/` |
| greedy.jsonl, masked.jsonl | One row per greedy answer |
| phase1/, phase2/ | Each phase's summary and per-step log: answer loss, gradient norms, the readout loss, its unexplained share, the shuffled gap |
| benchmark-outcomes.jsonl, benchmark-summary.json | Per test map: row, grid size, success and extracted answer; accuracy overall and by size |
