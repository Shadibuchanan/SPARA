# Evidence for planning probe 6887458

The probe in [the planning probe spec](../../PLANNING_PROBE_SPEC.md), run on
[LIVR rerun 6253300](../livr-6253300/)'s final Stage 2 checkpoint.

- **Maps:** the 400 test maps, the 200 fresh 5 x 5 and 6 x 6 maps of `fresh/probe`,
  and the 1,000 training maps (with their released answers teacher-forced).
- **Compute:** one H200, 1,831 GPU-seconds (0.51 GPU-hours). That is 70% over the
  0.3 estimate; the headroom optimisation took about 17 seconds per failed map.
- **Decoding:** greedy, at most 128 new tokens.

## Pre-declared results

### Headroom

LIVR fails 73 of the 400 test maps greedily.
- **What was optimised:** the four layer-16 latent states, within the exploration
  probe's basis.
- **Units:** coefficient norm per slot, with r = 4 being a training map's typical
  deviation from its grid size's mean.

| r | Fixed (any success) | Target reproduced | Control reproduced | Target NLL, start → end |
| --- | --- | --- | --- | --- |
| 1 | 2 of 73 | 1 | 0 | 0.473 → 0.467 |
| 2 | 5 of 73 | 4 | 0 | 0.470 → 0.457 |
| 4 | 4 of 73 | 3 | 0 | 0.471 → 0.447 |
| 8 | 7 of 73 | 6 | 0 | 0.468 → 0.420 |

**Headroom (rule): absent.** At r = 8, 9.6% are fixed, below the 20% threshold. Read
this narrowly:
- **The radii were too small, a design error visible before the run.** In the
  exploration probe, random changes of the same size in the same basis changed only
  5.9% of routes (‖a‖ ≈ 8), and 10% at ‖a‖ ≈ 16. So r ≤ 8 sits below the scale at
  which layer-16 changes move routes at all.
- **What the test does answer:** natural-sized changes to the layer-16 latents
  rarely fix a failure.
- **What it leaves open:** whether larger changes would.
- **The optimiser was constrained, not stalled.** Most rows ended on their norm
  bound: 63 to 70 of 73 target rows per radius, and every control row.
- **The landscape inside the ball is nearly flat.** 7 of 73 fixed at r = 8 is about
  what four random draws of that size achieved on fresh failures (6 of 57).

**Specificity (rule): inconclusive (McNemar p = 0.25). It is better called
uninformative.** No control was ever reproduced. But the controls were never
matched in difficulty: median NLL 1.13 against 0.41 for the targets. Every
one-move-wrong version of a correct route is far less likely than the route itself.

### State tracking

The position probe's layer, chosen by held-out training accuracy, is layer 20. The
probe is weak:
- **Held-out training maps:** 43%.
- **Solved test steps:** 39%.
- **Solved fresh steps:** 49%.

| Split | Fatal steps, k ≥ 2 | Decoded correctly | Expected from solved steps | Gap | p | Reading (rule) |
| --- | --- | --- | --- | --- | --- | --- |
| test | 51 | 8 (16%) | 26% | 9.9 points | 0.063 | tracking intact |
| fresh | 42 | 7 (17%) | 40% | 22.9 points | 0.001 | tracking contributes |

- **The test reading sits on the boundary:** a gap of 9.9 points against the
  10-point cutoff, and p = 0.063.
- **Both splits point the same way:** fatal steps decode less often than matched
  solved steps.
- **Fatal first moves are reported separately.** They are 17 on test maps and 8 on
  fresh maps, with the start decoded 71% and 75% of the time.
- **A probe this weak cannot carry a verdict either way.**

**S2.** From a wrongly decoded cell, the fatal move would be safe 84% of the time
(test), against 87% from other cells. This gives no sign that the model "thought it
was somewhere safe". It is uninformative given the probe.

**S3.** The bias check fired, so the spec's comparison of fatal holes with avoided
holes is confounded:
- **What fired:** the probe scores the direction taken higher than other safe
  directions (mean 0.33 against 0.09 on test maps).
- **Fairer comparison:** within the direction taken, fatal holes are read as holes
  41% (test) and 42% (fresh) of the time, against 21% and 29% for safe cells. This
  is a weak signal, reported without a verdict.

## Exploratory, post-hoc checks

None of these was fixed before the run. They come from
[scripts/analyze_planning_probe.py](../../../scripts/analyze_planning_probe.py)
(`exploratory.json`), run on CPU from the run's features.

| Check | Result |
| --- | --- |
| Feature alignment: the move about to be written, decoded from the state that predicts it | 54% at layer 4, rising to 98% at layer 36 |
| Agent's offset from the start (rows, columns), linear, held-out training maps | 100% at step 2 (the previous move alone); 87–89% at step 3 and later, layers 12–24 |
| Offset from the start on the model's own routes, layer 20 | 99% of solved steps (test and fresh); 49 of 51 fatal steps (test), 42 of 42 (fresh) |
| Start cell at the first move (linear) | 45–68% across layers |
| Absolute cell, small MLP, held-out training maps | 33% (layer 16), 55% (layer 20), 53% (layer 28) |
| Position probe at the earlier steps of failed routes | 28% against 25% expected (test); 33% against 36% (fresh) |
| Fatal steps against solved steps next to a hole | 16% against 28% expected (test); 17% against 37% (fresh) |

What these suggest, with caveats:

- **The model does not lose count of its moves.** Its offset from the start is
  linearly present at almost every step, fatal ones included. That offset can be
  computed from the moves already written, so this is bookkeeping on the text, not
  tracking on the map.
- **The start location is the weak link at the writing position.**
  - The absolute cell is the start plus the offset. A linear probe cannot decode
    that sum even when both parts are present.
  - The MLP's 55% is roughly the start accuracy (about 60%) times the offset
    accuracy (about 90%).
  - The start is what is only partly present when moves are written. The battery
    found it almost perfectly encoded in the latents.
- **A probe on one token cannot see what attention fetches when it is needed.**
  "Not resident at the comma" is not the same as "not used".
- **The drop is specific to fatal steps.** Position decodes normally at the earlier
  steps of failed routes and at solved steps next to a hole. It drops only at the
  step before the fatal move. The numbers are small, and the probe is weak.

## Files

| File | Contents |
| --- | --- |
| summary.json | Pre-declared results: layer selection, S1 to S3 and the validity check per split, headroom per radius with the readings, timing and peak GPU memory |
| routes.jsonl | Every map's route: split, row_index, grid size, extracted route, outcome, steps dropped and the decoded completion (test and fresh greedy; training maps' released answers) |
| steps.jsonl | Each test and fresh step: size, k, true and decoded cell, move, fatal, solved, and the four hole-ahead scores (U, D, L, R) |
| headroom.jsonl | Each failed test map: own route, target and control with their unadjusted NLLs, candidate counts, and per radius and kind the matched step, NLL before and after, coefficient norms per slot, greedy route and outcome |
| exploratory.json | Output of the post-hoc script |
| manifest.json | Recipe, package versions, source hashes, data and fresh-set manifests, probe settings and seed |

This copy of `manifest.json` gives the checkpoint and basis paths relative to the
PACE results root, so its hash differs from the PACE original. The other files match
the originals byte for byte.

The hidden-state features (`features.pt`, 190 MB) stay in the PACE run directory,
`results/planning-probe-6887458/probe`. They hold per-move states at nine layers for
the training, test and fresh steps, and no model weights. To recompute the state
probe or the post-hoc checks:

```bash
python -m spara.planning_probe <run directory> data/vsp
python scripts/analyze_planning_probe.py <run directory>
```
