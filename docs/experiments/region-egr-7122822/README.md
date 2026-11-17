# Evidence for global + region latents, EGR (job 7122822)

The arm in [the latent JEPA spec](../../PERCEPTION_JEPA_SPEC.md#global--region-latents-egr):
- **What EGR is:** round 2's EB2 from E, with E's four global latents unchanged, plus one block latent
  per 2 x 2 block (4, 9 or 16 by grid size).
- **The mask, in both stages:** each block latent sees only the prompt before the image, its block's
  image tokens and itself.
- **Training:** the answer loss alone. A readout on detached block latents was a monitor only.

## The run

- **Compute:** one H200, 1,781 GPU-seconds by Slurm's elapsed time (29 min 41 s; 1,673 s in the run).
  The allowance was 2,100; the time limit was 34 minutes.
- **Training:** phase 1 took 418 s and phase 2 448 s (EJR: 405 and 432).
- **No benchmark** (pre-declared).
- **Source:** the hashes in manifest.json match the local code for every file the run imports except
  `src/spara/perception_jepa.py`.
  - **That file** differs only by a three-line change to `compare`, made after the run so the
    readings accept an arm without a benchmark. Undoing the change reproduces the run's hash. The arm
    does not call `compare`.
  - **Other mismatches** are files the run does not import.
- **Weights:** EGR's final weights (15 GB) were deleted from PACE scratch on 2026-11-16, at the
  user's request. EGD (job 7124258) started from them, so retraining EGD needs EGR rerun first. The
  latent states, the readout and the logs remain on PACE
  (`$SPARA_REMOTE_ROOT/results/region-egr-7122822/`) and in the git-ignored local `results/`.

**The start block** (summary.json):
- **The identity check passed.** On all 16 checked rows, the final-layer states over E's prefix,
  through the four global latents, equal E's exactly (difference 0.0).
- **Answer loss after adding the block latents:** +0.004 on 3 x 3 and 4 x 4 rows, +0.037 on larger
  rows. EGR's first optimiser step's answer loss was 0.187, against EB2's 0.100.
- **Training recovered:** answer loss 0.140 in the first quarter of steps and 0.077 in the last,
  against EB2's 0.133 and 0.067.
- **Targets added:** 88, for the monitor.

## Pre-declared readings

The output of `region_latents.compare_region(..., arm="egr", offset=4, separation=0.6)` is in
comparison.json.

**Validity: passed.**

**Manipulation check: took.** Both conditions held.
- **The count-matched block probe:** each block read from its own block latent, against EB2's token
  0.

  | | 7 x 7 | 8 x 8 | Mean |
  | --- | --- | --- | --- |
  | EB2, token 0 for every block | 0.699 | 0.679 | 0.689 |
  | EGR, each block's own latent | 0.942 | 0.926 | 0.934 |

  EGR − EB2: +0.246, 95% interval 0.237 to 0.254.
- **Separation:** the mean similarity between block latents' per-cell hole maps is −0.03 (−0.035 at
  7 x 7, −0.028 at 8 x 8), against a threshold of 0.6. EJR's was 0.83 and EB2's 0.85 to 0.86.
  - **Own block against other blocks:** correlation 0.80 against 0.15 at 7 x 7, and 0.73 against 0.11
    at 8 x 8.

**Primary: no detectable difference.** EGR against EB2 on the 400 held-out maps (greedy, exact
McNemar):

| | Maps | EGR | EB2 | Only EGR | Only EB2 | p |
| --- | --- | --- | --- | --- | --- | --- |
| Unmasked | 400 | 75 | 78 | 17 | 20 | 0.74 |

**Secondary: no detectable difference** from E2: 75 against 84 (19 against 28, p = 0.24).

**Discriminating, masked greedy** (the answer sees the image only through the latents): **no
detectable difference.**

| | Maps | EGR | EB2 | Only EGR | Only EB2 | p |
| --- | --- | --- | --- | --- | --- | --- |
| Masked | 400 | 68 | 73 | 12 | 17 | 0.46 |

**The outcome-table row that applies: "took, both flat".**
- **The pre-registered reading:** holding the map block by block in the latents does not change
  decisions, even when the latents are the only path to the image.
- **What that means:** the limit is downstream of the latents, in how the answer uses what they hold.
  The latent-perception line closes.
- **Caveat:** this holds for EB2's recipe, 375 steps from E. A longer bottleneck phase might teach the
  answer to read the block latents. Nothing here shows it would.

## Did the latents know about the holes the model stepped into? (post hoc, CPU)

[scripts/analyze_region_mistakes.py](../../../scripts/analyze_region_mistakes.py), output in
mistakes-vs-latents.json.
- **The test:** take the failed held-out maps whose first mistake steps into a hole, and ask whether
  the count-matched block probe read that cell as a hole.
- **Why the saved states apply to masked greedy:** a block latent sees only the prompt before the
  image, its block's image tokens and itself, and none of those change between stages.

| Hole cells read as holes by the latents | EGR | EB2 |
| --- | --- | --- |
| Every hole on the held-out maps | 89% | 49% |
| First mistakes, masked, a safe toward-goal move existed | 77 of 87 (89%) | 34 of 81 (42%) |
| First mistakes, masked, at a forced point | 110 of 136 (81%) | 55 of 135 (41%) |
| First mistakes, unmasked, a safe toward-goal move existed | 75 of 89 (84%) | 32 of 80 (40%) |
| First mistakes, unmasked, at a forced point | 99 of 117 (85%) | 56 of 128 (44%) |

**Reading:**
- **The latents mostly knew.** In 77 of the 87 masked cases where EGR stepped into a hole beside a
  safe move toward the goal, its latent for that block read the cell as a hole. That is the latents'
  rate on holes in general.
- **Whether the latent knew made no difference** to whether the model stepped in: EGR makes about as
  many of these mistakes as EB2, whose latents know half as often.
- **The answer is not acting on the block latents' hole content.**
- **Caveat:** "read as a hole" is a linear probe. The information is available in the latents; this
  does not show what the answer attends to.

## Descriptive

**Round 2's probes at layer 12** (all fitted jointly; extra latents lower these):

| | E | E2 | EB2 | EGR |
| --- | --- | --- | --- | --- |
| All latents | 0.649 | 0.653 | 0.657 | 0.708 |
| The four global latents alone | | | | 0.658 |
| The block latents alone | | | | 0.705 |

- **The global latents did not change:** 0.658 against EB2's 0.657.

**The monitor readout broke down:**
- **Share of target variance unexplained, 32 validation maps:** 0.56 at the start, 41.8 after phase 1
  and 15.3 after phase 2.
- **Why:** its standardisation was fixed from E's states at the start. The block latents' states
  moved far from those during training, so the readout's inputs grew large.
- **The valid reading is the block probe above,** refitted on the saved states.

**First mistakes on failed held-out maps:**

| | E | E2 | EB2 | EGR |
| --- | --- | --- | --- | --- |
| Into an adjacent hole, a safe shortest move toward the goal existed | 96 | 79 | 80 | 89 |
| Into an adjacent hole, at a forced point | 130 | 124 | 128 | 117 |
| Safe, but off every shortest route | 91 | 91 | 91 | 96 |
| Stops short, wall bump or overrun | 29 | 22 | 23 | 23 |

**Outcomes on the held-out maps:**
- EGR reached the goal by a shortest route on 71 maps (EB2 75) and by a longer one on 4 (EB2 3).
- It fell into a hole on 304 (EB2 305), and stopped short on 21 (EB2 17).

**Participation ratio** of the latents at layer 12 (7 x 7, 8 x 8): EGR 46.7 and 45.9 (20 latents),
against EB2's 22.6 and 21.4.

**Harm check on round 1's 240 held-out maps:** EGR 90 against E's 88 (p = 0.82). EB2 100, E2 102, EJF
103, EJR 95.

## Files

| File | Contents |
| --- | --- |
| comparison.json | `region_latents.compare_region`: validity, the manipulation check with separation, primary, secondary, the masked-greedy reading, descriptive readings |
| mistakes-vs-latents.json | Post hoc: whether the latents read as holes the cells that first mistakes stepped into, by kind and mode, against their recall on all hole cells |
| summary.json | The start check, targets added, the readout's warm start, training and monitored readout errors, evaluation counts, time |
| manifest.json | Run manifest; the PACE root is written as `$SPARA_REMOTE_ROOT/` |
| greedy.jsonl, masked.jsonl | One row per greedy answer, unmasked and masked |
| phase1/, phase2/ | Each phase's summary and per-step log: answer loss, gradient norms, the monitor readout's loss, unexplained share and shuffled gap |
