# Evidence for the fixed-target retry: features job 7116225 and gate 3 (CPU job 7116395)

The retry is specified in [the latent JEPA spec](../../PERCEPTION_JEPA_SPEC.md#retry-with-fixed-targets-ejf).

## The runs

- **Features, job 7116225:** one H200, 489 GPU-seconds by Slurm's elapsed time (8 min 9 s). The
  features run took 248 s; the rest was setup and loading. Peak memory 15.2 GiB. The estimate was
  about 6 minutes.
  - It saved E's layer-12 latents and layer-4 cells for 2,185 distinct maps:
    - 1,486 training maps, 201 of them from the pool;
    - 500 pool maps;
    - 400 held-out maps;
    - 128 validation maps, within the pool.
  - The file (`features.pt`, 388 MB) is on PACE at `$SPARA_REMOTE_ROOT/results/jepa-features-7116225/`
    and in the git-ignored local `results/`.
- **Gate 3, CPU job 7116395:** one `ice-cpu` node, 1 min 44 s. No GPU.
  - Slurm reports FAILED because the script exits with code 3 when the gate fails.
- **The approved allowance:** 2,400 GPU-seconds. Used: 489.
- **Source:** the hashes in features-manifest.json match the local code for every file the run
  imports. The mismatches are files it does not import: the scaffolding files, the PACE config and
  the package's `SOURCES.txt`.

## Gate 3: failed at 3c

**3a and 3b passed.** The fixed targets at 256 components:

| | 7 x 7 | 8 x 8 |
| --- | --- | --- |
| Holes read from the targets (3a, at least 0.95) | 1.00 | 1.00 |
| Share of target variance the hole label explains (3b, at least 0.10) | 0.21 | 0.23 |
| Share of target variance between positions | 0.008 | 0.008 |

- The 256 components keep 94% of the standardised cells' variance.
- Position explains under 1% of the targets, as designed. EJ2's shortcut is closed.

**3c failed.** The warm-started heads on held-out maps:

| | 7 x 7 | 8 x 8 | Threshold |
| --- | --- | --- | --- |
| Share of target variance unexplained | 1.000 | 1.001 | |
| Shuffled gap | 0.000004 | 0.000004 | at least 0.02 |
| Holes from the predictions | 0.50 | 0.50 | at least 0.55 |

- **The heads learned nothing.** Validation error was 1.003 at the start and 1.001 at the best
  epoch (epoch 2 of 30). Training error stayed at 1.000.
- **The linear ceiling** (one ridge per size from the four latents to all cells' targets) explains
  16% of target variance at 7 x 7 and 7% at 8 x 8. Its holes from the predictions are 0.54 and 0.53.
- **The reading, by the spec's table:** "3c; the linear ceiling fails too: E's latents carry too
  little cell content for the term to grip at the start. Stop; the user decides."

## Why it failed (post hoc, CPU)

None of this was fixed before the run. It comes from
[scripts/diagnose_fixed_gate.py](../../../scripts/diagnose_fixed_gate.py) (diagnosis.json).

**The context projector could not see differences between maps.**
- E's latents carry a few large constant channels, with means up to 285, while the median channel's
  standard deviation across maps is 0.02 to 0.2.
- Only 0.003% to 0.04% of each latent's squared norm differs between held-out maps.
- The context projector begins with a parameter-free layer norm, which leaves that share unchanged
  (0.001% to 0.04%). The predictor therefore received almost the same input for every map.

**This also applies to EJ2.** EJ2's heads used the same context projector on the same latents of
E. Its predictions were position codes. The learned target projector explains how they could still
fit the targets; this explains why the latents could not have helped. That conclusion is an
interpretation, not a separate test.

**With each latent channel standardised first** (by its mean and standard deviation over E's
training maps), gate 3 rerun otherwise unchanged (standardised-gate-summary.json):

| Held-out maps | 7 x 7 | 8 x 8 | Threshold |
| --- | --- | --- | --- |
| Share of target variance unexplained | 0.873 | 0.987 | |
| Shuffled gap | 0.26 | 0.038 | at least 0.02 |
| Holes from the predictions | 0.52 | 0.50 | at least 0.55 |

- The heads now use the latents, with a shuffled gap above the threshold at both sizes. Holes from
  the predictions still fall below 0.55.
- **Why 0.55 was not reachable:** it was set against round 2's latent probe (0.65), which regresses
  holes directly. The predictions here estimate 256-dimensional targets, where holes are about 22% of
  the variance, so even the linear ceiling reads holes at only 0.53 to 0.54.

**In calibrated units.** Per size, the targets' hole direction is the hole-minus-ice mean target on
pool maps. The hole label explains 86% of the targets' variance along it. The share of that
direction's variance on held-out maps that each predictor explains:

| | 7 x 7 | 8 x 8 |
| --- | --- | --- |
| Gate 3's heads (raw latents) | 0.000 | −0.001 |
| Heads on standardised latents | 0.107 | 0.008 |
| Linear ceiling | 0.121 | 0.061 |

- **At 7 x 7,** the standardised heads reach most of what a linear map from E's latents can.
- **At 8 x 8,** they reach little. Only 94 of the 1,486 training maps are 8 x 8 (107 are 7 x 7; the
  rest are 3 x 3 to 6 x 6).
- **E's latents hold little per-cell hole content** to start from, at either size.

## Files

| File | Contents |
| --- | --- |
| gate-summary.json | Gate 3's output: the targets' checks, the warm start's history, 3c, the linear ceiling, the reading |
| features-summary.json | The features job: maps per group, round 2's row counts, time, peak memory |
| features-manifest.json | The features job's run manifest; the PACE root is written as `$SPARA_REMOTE_ROOT/` |
| diagnosis.json | The post hoc diagnosis: the latents' spread before and after standardising, the standardised rerun's 3c, and the hole-direction shares |
| standardised-gate-summary.json | Gate 3's full output on standardised latents (post hoc) |
