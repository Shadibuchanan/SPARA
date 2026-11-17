# Evidence for the latent JEPA arm, EJ2 (job 7107309)

The arm in the [latent JEPA spec](../../PERCEPTION_JEPA_SPEC.md#the-arm-ej2). EJ2 is round 2's EB2
plus the latent JEPA term in both phases, from E, with the heads, centring means and loss weight of
[gate job 7106850](../jepa-gate-7106850/README.md) (target layer 4, β = 2.47).

- **Compute:** one H200, 1,871 GPU-seconds by Slurm's elapsed time (31 min 11 s; 1,641 s in the job;
  peak 55.5 GiB). The estimate was 30–35 GPU-minutes.
- **Training:** phase 1 took 419 s and phase 2 427 s.
- **The approved allowance:** 3,300 GPU-seconds for the gate and the arm. Used: 3,021 (2 + 1,148 +
  1,871).
- **Source:** the hashes in manifest.json match the local code for every file the run imports. The
  mismatches are files it does not import: the protected PACE files and the package's `SOURCES.txt`.
- **Weights:** EJ2's final weights (15 GB) were deleted from PACE scratch on 2026-11-16, at the
  user's request. The latent states (`states.pt`), the heads and the logs remain on PACE
  (`$SPARA_REMOTE_ROOT/results/jepa-ej2-7107309/`) and in the git-ignored local `results/`.

## Pre-declared readings

The output of `perception_jepa.compare` is in comparison.json.

**Validity: passed.** The gate passed; the set checks passed. EJ2's first optimiser step's answer
loss equals EB2's (0.10009 both).

**Manipulation check: did not take.** Round 2's latent hole probe at layer 12, fitted on pool maps
and scored on held-out maps:

| | E | E2 | EB2 | EJ2 |
| --- | --- | --- | --- | --- |
| Hole balanced accuracy, mean of 7 x 7 and 8 x 8 | 0.649 | 0.653 | 0.657 | 0.664 |

- **EJ2 − EB2:** 0.007, 95% interval −0.0002 to 0.015. The margin was 0.05.

**Primary: no reading on perception.** EJ2 against EB2 on the 400 held-out maps (greedy, exact
McNemar), reported as descriptive because the check failed:

| | Maps | EJ2 | EB2 | Only EJ2 | Only EB2 | p |
| --- | --- | --- | --- | --- | --- | --- |
| All held-out | 400 | 68 | 78 | 15 | 25 | 0.15 |
| 7 x 7 | 198 | 44 | 45 | | | 1.0 |
| 8 x 8 | 202 | 24 | 33 | | | 0.049 |

**Secondary: EJ2 solves fewer than E2.**

| | Maps | EJ2 | E2 | Only EJ2 | Only E2 | p |
| --- | --- | --- | --- | --- | --- | --- |
| All held-out | 400 | 68 | 84 | 18 | 34 | 0.036 |

**The outcome-table row that applies: "Did not take".** Its next step is to check whether the
prediction depended on the latents and whether validation error fell. Both are answered below.

## Why it did not take (post hoc, CPU)

None of this was fixed before the run. The first two rows come from the training logs; the rest come
from [scripts/analyze_jepa_predictions.py](../../../scripts/analyze_jepa_predictions.py), run on the
saved latent states and heads (predictions-*.json).

**The JEPA was learned and generalised: validation error fell sharply.** Share of target variance
unexplained:

| | Start (warm-started heads) | After phase 1 | After phase 2 |
| --- | --- | --- | --- |
| 32 validation maps (never trained on) | 0.649 | 0.220 | 0.106 |
| 32 training maps | 0.527 | 0.149 | 0.071 |

**But the prediction did not depend on the latents.**
- **The shuffled gap:** with another map's latents, the error rose by only 0.016 to 0.029 of target
  variance (logged every step), while the error itself was 0.07 to 0.36.
- **The predictions are position codes.** On the held-out maps, the share of the predictions'
  variance that differs between maps at the same cell is under 0.03%. The rest is explained by
  position alone. A ridge probe on the predictions reads holes at chance (0.50) at both sizes.
- **This held before any language-model update:** E's latents through the warm-started heads give
  the same result. The gate did not test it.

| Predictions, held-out maps | Map-dependent share, 7 x 7 / 8 x 8 | Holes from predictions, 7 x 7 / 8 x 8 |
| --- | --- | --- |
| E's latents, warm-started heads (gate) | 0.03% / 0.04% | 0.50 / 0.50 |
| EJ2's latents, EJ2's heads | 0.016% / 0.006% | 0.50 / 0.50 |

**Reading (interpretation, not a measurement of mechanism):**
- **A shortcut.** The target projector is learned, and the predictor is told each cell's position.
  The projector made the targets mostly a nonlinear function of position, which the predictor could
  hit without the latents.
- **Why centring did not stop it.** Centring removed the per-position mean, and gate 1 checked only
  that position was not *linearly* decodable afterwards.
- **The anti-collapse term did not stop it either.** SIGReg's target statistic rose from about 70 to
  130 over training: the targets moved away from a Gaussian, toward a few position codes.
- **A second cause, found later** ([gate 3 of the retry](../jepa-fixed-gate-7116395/README.md)):
  only 0.003% to 0.04% of E's latents' squared norm differs between maps. The context projector's
  parameter-free layer norm leaves that unchanged, so the predictor saw almost the same input for
  every map. The latents could not have helped, with or without the shortcut.

**What reached the language model.**
- **The JEPA term into the latents was mostly SIGReg.** λ · SIGReg made up 85% of the term's value in
  phase 1's first quarter and 98% in phase 2's last (0.02 × 50 to 130, against an error of 0.04 to
  0.24). The prediction error barely depends on the
  latents, so it gave them little gradient.
- **The latents spread.** Their participation ratio at layer 12 (centred, channel-standardised) is 30.5
  and 25.8 at 7 x 7 and 8 x 8, against 22.6 and 21.4 for EB2, 20.6 and 23.7 for E2, and 21.4 and 20.7
  for E.
- **Plausible but not tested:** spreading the latents without adding map content is consistent with
  the harm against E2.

## Descriptive

**First mistakes on failed held-out maps:**

| | E | E2 | EB2 | EJ2 |
| --- | --- | --- | --- | --- |
| Into an adjacent hole, a safe shortest move toward the goal existed | 96 | 79 | 80 | 90 |
| Into an adjacent hole, at a forced point | 130 | 124 | 128 | 128 |
| Safe, but off every shortest route | 91 | 91 | 91 | 94 |
| Stops short, wall bump or overrun | 29 | 22 | 23 | 20 |

The count the spec watches rose rather than fell.

**Outcomes on the held-out maps:**

| | E | E2 | EB2 | EJ2 |
| --- | --- | --- | --- | --- |
| Goal by a shortest route | 52 | 82 | 75 | 61 |
| Goal by a longer route | 2 | 2 | 3 | 7 |
| Into a hole | 322 | 295 | 305 | 317 |
| Stops short | 22 | 21 | 17 | 15 |

**Masked greedy** (only the latents see the image): EJ2 59 of 400, against 73 for EB2 and 66 for E2.

**Harm check on round 1's 240 held-out maps:** EJ2 100 against E's 88 (p = 0.050). EB2 100, E2 102.

**Benchmark** (one sampled answer per test map), against LIVR:

| | All 400 | p | Detour (40) | p | No-detour (360) | p |
| --- | --- | --- | --- | --- | --- | --- |
| LIVR | 328 | — | 9 | — | 319 | — |
| EJ2 | 323 | 0.53 | 17 | 0.021 | 306 | 0.029 |
| EB2 | 325 | 0.73 | 17 | 0.008 | 308 | 0.043 |
| E2 | 322 | 0.48 | 19 | 0.002 | 303 | 0.017 |

This is the same trade as round 2's arms.

## For a next attempt

These are lessons, not readings:
- **Gate on latent dependence before the arm.** At warm start, require the predictions to vary between
  maps at a cell (the map-dependent share) and to carry holes on held-out maps. This check needs only
  the gate's features and seconds of CPU. It would have stopped this run before the 31-minute arm.
- **Remove the shortcut by construction.** Two options:
  - fixed targets: a frozen projection, such as PCA-whitened centred cell states, with nothing learned
    on the target side;
  - per-position centring in embedding space: subtract a running per-position mean of the projected
    targets, so the position-predictable part is zero by definition.
- **Weigh SIGReg against the prediction term in gradient, not value.** Here it dominated the term.

## Files

| File | Contents |
| --- | --- |
| comparison.json | `perception_jepa.compare`: validity, manipulation check, primary and secondary, descriptive (first mistakes, participation ratios, JEPA errors, harm checks, benchmark, failures), and the gate's figures |
| predictions-ej2.json, predictions-e-warm-start.json | The post-hoc prediction check, per size: map-dependent share and holes from predictions |
| summary.json | Set checks, counts, training summaries and JEPA errors, evaluation counts, time, peak memory |
| manifest.json | Run manifest; the PACE root is written as `$SPARA_REMOTE_ROOT/` |
| greedy.jsonl, masked.jsonl | One row per greedy answer: population, split, row, grid size, route, success |
| phase1/, phase2/ | Each phase's summary and its per-step log: answer loss, gradient norm, the heads' gradient norm, the JEPA term and its parts, the shuffled gap |
| benchmark-outcomes.jsonl, benchmark-summary.json | Per test map: row, grid size, success and extracted answer; accuracy overall and by size |
