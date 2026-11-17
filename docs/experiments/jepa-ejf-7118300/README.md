# Evidence for the fixed-target arm, EJF (job 7118300)

This is the arm in [the latent JEPA spec](../../PERCEPTION_JEPA_SPEC.md#retry-with-fixed-targets-ejf),
as amended ([amendment](../../PERCEPTION_JEPA_SPEC.md#amendment-after-gate-3-2026-11-16)). EJF is
round 2's EB2 plus the fixed-target term in both phases, starting from E. It uses the heads, the
latent standardisation and the targets of [gate 3, job 7118222](../jepa-fixed-gate-7117800/README.md).

## The run

- **Compute:** one H200, 1,994 GPU-seconds by Slurm's elapsed time (33 min 14 s; 1,730 s in the
  run; peak 55.4 GiB).
- **Slurm reports TIMEOUT.** The run had finished and written summary.json. The job passed its
  32 min 56 s limit while shutting down, and Slurm killed it 18 s later.
- **Training:** phase 1 took 429 s and phase 2 437 s.
- **The approved allowance:** 2,600 GPU-seconds for the retry. Used: 2,618, **18 over**.
  - Features (job 7116225): 489.
  - A first EJF job that failed on the map key (7117889): 135.
  - This job: 1,994.
- **Targets added at the start:** 88 training maps the features job had merged into others. They got
  targets before any update, from the same prompt pass, centring means and projection.
- **The loss weight:** β = 0.027 (bounds 0.01 to 100). On the 16 pilot rows, the standardised term's
  median gradient into the language model was about 37 times the answer loss's.
- **Source:** the hashes in manifest.json match the local code for every file the run imports. The
  mismatches are files it does not import: the scaffolding files, the PACE config, the package's
  `SOURCES.txt` and scripts/analyze_jepa_predictions.py (edited after the run).
- **Weights:** EJF's final weights (15 GB) were deleted from PACE scratch on 2026-11-16, at the
  user's request. The latent states, the heads and the logs remain on PACE
  (`$SPARA_REMOTE_ROOT/results/jepa-ejf-7118300/`) and in the git-ignored local `results/`.

## Pre-declared readings

The output of `perception_jepa.compare(..., arm="ejf")` is in comparison.json.

**Validity: passed.** Gate 3 passed and the set checks passed. EJF's first optimiser step's answer
loss equals EB2's (0.10009 both).

**Manipulation check: did not take.** Round 2's latent hole probe at layer 12, fitted on pool maps
and scored on held-out maps:

| | E | E2 | EB2 | EJF |
| --- | --- | --- | --- | --- |
| Hole balanced accuracy, mean of 7 x 7 and 8 x 8 | 0.649 | 0.653 | 0.657 | 0.655 |

- **EJF − EB2:** −0.002, 95% interval −0.010 to 0.006. The margin was 0.05.

**Primary: no reading on perception.** EJF against EB2 on the 400 held-out maps (greedy, exact
McNemar), reported as descriptive because the check failed:

| | Maps | EJF | EB2 | Only EJF | Only EB2 | p |
| --- | --- | --- | --- | --- | --- | --- |
| All held-out | 400 | 76 | 78 | 11 | 13 | 0.84 |

**Secondary: no detectable difference from E2.**

| | Maps | EJF | E2 | Only EJF | Only E2 | p |
| --- | --- | --- | --- | --- | --- | --- |
| All held-out | 400 | 76 | 84 | 17 | 25 | 0.28 |

**The outcome-table row that applies: "Did not take".** Its next step is to check whether the
prediction depended on the latents and whether validation error fell. Unlike EJ2, it did depend on
them; validation error fell a little. See below.

## The term took hold; the latents did not gain target content

**Monitored maps,** with the current model and heads (`training.jepa_error` in summary.json):

| | Start | After phase 1 | After phase 2 |
| --- | --- | --- | --- |
| 32 validation maps: share unexplained | 0.963 | 0.953 | 0.932 |
| 32 validation maps: shuffled gap | 0.084 | 0.110 | 0.146 |
| 32 training maps: share unexplained | 0.590 | 0.705 | 0.631 |

**Training rows** (per-step logs, first and last quarter of the 375 steps):
- the unexplained share fell from 0.91 to 0.71;
- the shuffled gap rose from 0.26 to 0.60.

**Held-out maps,** from the saved latent states (predictions-*.json): E's latents with gate 3's
heads, against EJF's latents with EJF's final heads.

| | E, gate 3's heads | EJF, final heads |
| --- | --- | --- |
| Shuffled gap, 7 x 7 / 8 x 8 | 0.278 / 0.027 | 0.286 / 0.096 |
| Share unexplained, 7 x 7 / 8 x 8 | 0.871 / 0.993 | 0.862 / 0.953 |
| Hole-direction share, 7 x 7 / 8 x 8 | 0.118 / 0.005 | 0.110 / 0.020 |
| Holes from the predictions, 7 x 7 / 8 x 8 | 0.52 / 0.50 | 0.52 / 0.50 |

That table changes the latents and the heads together. Two checks separate them (post hoc, CPU):
- **Gate 3's fixed heads on EJF's latents** (predictions-ejf-gate-heads.json):
  - shuffled gap 0.251 and 0.071;
  - share unexplained 0.919 and 0.994;
  - hole-direction share 0.038 and 0.009.
- **The linear ceiling on each model's saved latents** (linear-ceiling-states.json). It is one ridge
  per size from the four latents to every cell's target, fitted on pool maps and scored on held-out
  maps:

| Held-out, 7 x 7 / 8 x 8 | Share unexplained | Hole-direction share |
| --- | --- | --- |
| E | 0.845 / 0.934 | 0.121 / 0.061 |
| EB2, without the term | 0.824 / 0.917 | 0.161 / 0.094 |
| EJF | 0.832 / 0.945 | 0.123 / 0.031 |

**Reading:**
- **The term took hold.** The predictions came to depend more on the latents, mostly at 8 x 8, and
  error fell on maps never trained on.
- **The latents did not gain target content that a linear readout can see.** EB2's latents, trained
  without the term, predict the targets as well or better at both sizes, and carry more of the hole
  direction.
- **What improved was mainly the co-trained heads.** They may use latent changes that a linear
  readout does not see. That is an interpretation, not a measurement.

## Descriptive

**First mistakes on failed held-out maps:**

| | E | E2 | EB2 | EJF |
| --- | --- | --- | --- | --- |
| Into an adjacent hole, a safe shortest move toward the goal existed | 96 | 79 | 80 | 91 |
| Into an adjacent hole, at a forced point | 130 | 124 | 128 | 124 |
| Safe, but off every shortest route | 91 | 91 | 91 | 92 |
| Stops short, wall bump or overrun | 29 | 22 | 23 | 17 |

The perception count rose, as it did for EJ2 (90).

**Outcomes on the held-out maps:**

| | E | E2 | EB2 | EJF |
| --- | --- | --- | --- | --- |
| Goal by a shortest route | 52 | 82 | 75 | 73 |
| Goal by a longer route | 2 | 2 | 3 | 3 |
| Into a hole | 322 | 295 | 305 | 309 |
| Stops short or passes the goal | 24 | 21 | 17 | 15 |

**Participation ratio** of the latents at layer 12 (7 x 7, 8 x 8): EJF 25.6 and 23.3, against EB2's
22.6 and 21.4, E2's 20.6 and 23.7, and E's 21.4 and 20.7.

**Masked greedy** (only the latents see the image): EJF 70 of 400, against EB2's 73, E2's 66 and
EJ2's 59.

**Harm check on round 1's 240 held-out maps:** EJF 103 against E's 88 (p = 0.006). EB2 100, E2 102.

**Benchmark** (one sampled answer per test map), against LIVR:

| | All 400 | p | Detour (40) | p | No-detour (360) | p |
| --- | --- | --- | --- | --- | --- | --- |
| LIVR | 328 | — | 9 | — | 319 | — |
| EJF | 318 | 0.18 | 16 | 0.039 | 302 | 0.008 |
| EB2 | 325 | 0.73 | 17 | 0.008 | 308 | 0.043 |
| E2 | 322 | 0.48 | 19 | 0.002 | 303 | 0.017 |

This is the same trade as round 2's arms.

## Files

| File | Contents |
| --- | --- |
| comparison.json | `perception_jepa.compare(..., arm="ejf")`: validity, manipulation check, primary and secondary, descriptive readings, and the gate's figures |
| predictions-ejf.json, predictions-e-warm-start.json, predictions-ejf-gate-heads.json | Held-out predictions per size (EJF's heads on EJF's latents; gate 3's heads on E's; gate 3's heads on EJF's): within-position share, holes from the predictions, unexplained share, shuffled gap, hole-direction share |
| linear-ceiling-states.json | The linear ceiling on E's, EB2's and EJF's saved latents (post hoc) |
| summary.json | Set checks, counts, the loss weight's pilot, targets added, training summaries and monitored errors, evaluation counts, time, peak memory |
| manifest.json | Run manifest; the PACE root is written as `$SPARA_REMOTE_ROOT/` |
| greedy.jsonl, masked.jsonl | One row per greedy answer: population, split, row, grid size, route, success |
| phase1/, phase2/ | Each phase's summary and per-step log: answer loss, gradient norms, the JEPA term, its unexplained share, the shuffled gap |
| benchmark-outcomes.jsonl, benchmark-summary.json | Per test map: row, grid size, success and extracted answer; accuracy overall and by size |
