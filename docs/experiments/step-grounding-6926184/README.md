# Evidence for step grounding 6926184

The run specified in [the step-grounding spec](../../STEP_GROUNDING_SPEC.md): LIVR's
Stage 2 retrained with the step-grounding loss from the final Stage 1 weights of
[LIVR rerun 6253300](../livr-6253300/). This condition is called *LIVR + step
grounding* (G).

- **Compute:** one H200, 3,486 GPU-seconds (0.97 GPU-hours), against the 1.1 estimate.
  - The job itself took 3,482 seconds: Stage 2 about 45 minutes (2,522 seconds of
    optimiser steps), the benchmark about 5 minutes, the state probe 7 minutes.
  - A first submission, job 6911236, stopped after 4 seconds on a shell syntax error in
    the job script, before any training.
- **Matched start:** G's answer losses at the first two optimiser steps equal the LIVR
  rerun's to every printed digit (0.08193 and 0.17769). The zero-initialised head
  leaves the first update unchanged, as intended. The logged gradient norms differ
  because G's includes the head's gradient.
- **Labels:** 1,000 training maps, 3,482 decision states, no steps dropped.

## Pre-declared results

### Primary: benchmark success (sampled decoding, 400 test maps)

| | G | LIVR rerun | Only G | Only LIVR | p (McNemar) |
| --- | --- | --- | --- | --- | --- |
| All sizes | 326 | 328 | 14 | 16 | 0.86 |
| 3 x 3 | 92 | 92 | 1 | 1 | 1.0 |
| 4 x 4 | 89 | 88 | 4 | 3 | 1.0 |
| 5 x 5 | 83 | 86 | 3 | 6 | 0.51 |
| 6 x 6 | 62 | 62 | 6 | 6 | 1.0 |

**Reading (rule): no detectable difference.**
- **Same as a rerun:** G and LIVR disagree on 30 maps. Two identical LIVR runs disagree
  on 31.
- **Failure types:** G walks into a hole on 68 maps and ends off the goal on 6. LIVR's
  figures are 69 and 3. The hole count did not fall, as it should have if grounding
  worked as intended.

### Manipulation check: position on maps not trained on

The ridge probe at layer 20, fitted on training steps and read on the model's own
greedy routes:

| | G | LIVR rerun |
| --- | --- | --- |
| Solved test steps | **89%** | 39% |
| Test steps before a fatal move (k ≥ 2) | 76% | 16% |
| Held-out training maps | 83% | 43% |

**Reading (rule): took.** The threshold was LIVR + 10 points, or 49%.

The fresh-map row of the pre-declared readout pools 5 x 5 to 8 x 8 for G: 70% on solved
steps and 48% on fatal ones. LIVR's fresh probe covered 5 x 5 and 6 x 6 only (49% and
17%). The size-matched comparison is under [post-hoc checks](#exploratory-post-hoc-checks).

### Combined reading (rule)

**Took, and no detectable difference.** In the words fixed before the run: position
became decodable but did not change routes. Either the model does not use it, or
tracking position is not the bottleneck.

### Secondary (no verdicts)

**Greedy routes**, against LIVR's from the planning and exploration probes:

| Maps | G | LIVR | Only G | Only LIVR | p | Holes (G / LIVR) | Off the goal (G / LIVR) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Test, 400 | 329 | 327 | 15 | 13 | 0.85 | 67 / 68 | 4 / 5 |
| Fresh, 400 | 203 | 217 | 17 | 31 | 0.059 | 141 / 135 | 56 / 48 |

On fresh maps, by size:
- **5 x 5:** 77 against 77;
- **6 x 6:** 61 against 67;
- **7 x 7:** 39 against 43;
- **8 x 8:** 26 against 30.

The fresh-map difference is not significant, but every size from 6 x 6 up leans
toward LIVR.

**Training fit.** The mean answer loss over the last epoch is 0.00078, against LIVR's
0.013. The spec's check was for "more than double", and this went the other way: G
fits the training answers about 17 times more tightly. Over the last epoch, the
grounding loss is 0.020, and the head's training accuracy is 100% for the cell and
99% for move outcomes.

**State tracking (S1), at the probe's selected layer.** That layer is 24 for G and was
20 for LIVR.

| Split | G: fatal decoded / expected | Gap, p | Reading (rule) | LIVR |
| --- | --- | --- | --- | --- |
| Test | 42 of 54 (78%) / 86% | 8.4 points, 0.057 | tracking intact | 16% / 26%, tracking intact |
| Fresh | 59 of 121 (49%) / 57% | 8.6 points, 0.006 | tracking intact | 17% / 40%, tracking contributes |

G's fresh row includes 7 x 7 and 8 x 8 maps.

**The trained head on G's own routes** (decision states, layer 20):

| | Cell, solved | Cell, fatal k ≥ 2 | Other three outcomes, solved | Other three, fatal k ≥ 2 |
| --- | --- | --- | --- | --- |
| Test | 99% | 94% | 95% | 83% |
| Fresh (5 x 5 to 8 x 8) | 82% | 60% | 92% | 86% |

As the spec states, the head's outcome for the move actually taken is not evidence.
Training routes never enter a hole, and the state already encodes the move.

## Exploratory, post-hoc checks

None of these was fixed before the run. They come from
[scripts/step_grounding_exploratory.py](../../../scripts/step_grounding_exploratory.py)
(`exploratory.json`), run on CPU from the run's features and the trained head.

**Position on fresh maps by size**, at layer 20:

| Fresh maps | Ridge, solved | Ridge, fatal k ≥ 2 | Head, solved | Head, fatal k ≥ 2 |
| --- | --- | --- | --- | --- |
| 5 x 5 and 6 x 6 | 85% (447) | 78% (50) | 98% | 92% |
| 7 x 7 and 8 x 8, row and column in 0–5 | 40% (136) | 32% (60) | 49% | 45% |
| 7 x 7 and 8 x 8, row or column 6–7 | 0% (32) | 0% (11) | 0% | 0% |

LIVR's figures for 5 x 5 and 6 x 6 are 49% (solved) and 17% (fatal).

**Hole ahead in the direction taken.** This is the share of steps where the probe
scores the cell about to be entered above 0.5. It is computed as in the planning probe,
at the probe's selected layer: 24 for G, 20 for LIVR.

| | G: fatal | G: safe | LIVR: fatal | LIVR: safe |
| --- | --- | --- | --- | --- |
| Test | 51% (67) | 8% (859) | 41% (68) | 21% (849) |
| Fresh 5 x 5 and 6 x 6 | 46% (54) | 11% (447) | 42% (50) | 29% (468) |

What these suggest, with caveats:
- **At fatal steps G knows where it is.** On test maps and on fresh 5 x 5 and 6 x 6
  maps, the cell before a fatal move is read correctly 76–78% of the time (ridge) and
  92–94% (head). Yet G walks into holes as often as LIVR. Losing track of position
  does not explain its failures.
- **The state often marks the hole about to be entered.** For about half of the fatal
  moves, a linear probe reads a hole in the direction about to be taken, against
  8–11% false alarms on safe moves. The model writes the move anyway. The layers
  differ between the two runs, and one probe is weak evidence.
- **Grounding did not carry over to larger maps.** Rows and columns 6–7 never appear in
  training labels, so they cannot be read. Even cells inside the trained range decode
  only 40–49% on 7 x 7 and 8 x 8 maps.

## Files

| File | Contents |
| --- | --- |
| comparison.json | Output of [scripts/analyze_step_grounding.py](../../../scripts/analyze_step_grounding.py): primary, secondary and manipulation-check results with the readings |
| exploratory.json | Output of the post-hoc script |
| evaluation-summary.json | Benchmark success overall and per grid size |
| evaluation-outcomes.jsonl | Per test map: row_index, grid size, success and extracted answer (from the run's `responses.jsonl`) |
| steps.jsonl | Stage 2 log per optimiser step: answer loss, gradient norm, learning rate, grounding loss and the head's training accuracies |
| stage2-summary.json | Final training state, with the grounding settings and label counts |
| routes.jsonl | Greedy routes on the 400 test and 400 fresh maps, and the training maps' teacher-forced answers |
| probe-steps.jsonl | Each test and fresh step: size, k, true and decoded cell (selected layer), move, fatal, solved, hole-ahead scores |
| probe-summary.json | State probe: layer selection, S1 to S3 per split, timing, peak GPU memory |
| manifest.json | Stage 2 recipe, grounding settings, package versions, source hashes and checkpoint lineage |

This copy of `manifest.json` gives the Stage 1 checkpoint path relative to the PACE
root, so its hash differs from the original. Apart from that file and the outcomes
extract, the files match the PACE originals byte for byte.

These stay in `results/step-grounding-6926184` on PACE scratch:
- the final weights (`stage2/final`, 15 GB), until the team decides whether to keep
  them;
- the head (`stage2/grounding_head.pt`);
- the hidden-state features (`probe/features.pt`, 216 MB);
- the full benchmark responses.

To recompute:

```bash
python scripts/analyze_step_grounding.py <run directory> --data-root data/vsp
python scripts/step_grounding_exploratory.py <run directory>
```
