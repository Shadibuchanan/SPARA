# Text-only RL results

Updated 2026-10-14. One seed each; no test-based selection or tuning.

## Summary

Text-only RL (condition B) scored **326/400 (81.5%)**, against **328/400 (82.0%)**
for the LIVR checkpoint it started from. This is no measurable change: RL and its
starting checkpoint disagree on 20 maps (9 gained, 11 lost), fewer than the 31
maps on which two LIVR baseline runs with identical settings disagree.

| Grid | LIVR rerun (6253300) | Text-only RL (6256423) | LIVR original (6189457) |
| --- | ---: | ---: | ---: |
| 3 x 3 | 92% | 92% | 92% |
| 4 x 4 | 88% | 84% | 91% |
| 5 x 5 | 86% | 86% | 81% |
| 6 x 6 | 62% | 64% | 59% |
| Overall | 328/400 (82.0%) | 326/400 (81.5%) | 323/400 (80.75%) |

## LIVR baseline rerun

Job `6253300` reran the unchanged [recipe](../configs/livr.json) on one H200 under
account `isye`, in 5,652 GPU-seconds. Final checkpoints omitted optimizer state
(`--final-weights-only`) for storage; training is unaffected.

Stage 1 reproduced the original run's 1,250 losses exactly. Stage 2 losses match
for two steps, but its gradient norm differs from the first update (14.0593 vs
14.0622): the backward pass under standard attention is not bitwise deterministic
on this stack, and the difference compounds. The two baselines therefore act as a
rough run-to-run comparison: 310 maps solved by both, 18 only by the rerun, 13 only
by the original, 59 by neither.

## Text-only RL

Job `6256423` ran one epoch (125 updates, 8 maps x 8 rollouts each) with the
[settings](../configs/text_rl.json) and assumptions in the
[specification](TEXT_RL_SPEC.md), followed by reference evaluation, in 3,441
GPU-seconds. Peak GPU memory was 69.8 GiB; updates took about 25 seconds.

The learning signal was weak, as anticipated. Mean training reward stayed near
0.91 throughout, and **only 99 of 1,000 map groups (10%)** had any difference in
reward among their eight rollouts; the rest contributed no policy gradient. KL to
the starting checkpoint stayed between about 0.004 and 0.04. A 10-step pilot
(job `6255753`, 550 GPU-seconds) had shown 20% informative groups in its first
80 maps.

This matches the small effect Scaffolding Minds reports for text-only GRPO
(+0.6 points), but here the dominant limitation is that the training maps are
nearly memorized after LIVR training. Options for a stronger condition B, each a
scope decision: more rollouts per map, higher rollout temperature, more epochs, or
new or larger maps. Any change should apply equally to condition C.

## Learning signal and variance

**Reward signal.** Of 8,000 RL rollouts, 9.9% received a nonzero advantage: 86.3% of
training-map groups had all eight rollouts succeed and 3.8% had all eight fail. All
3x3 groups were uniform; mixed groups rose to 17.2% at 6x6. If the same 91.2% mean
success were spread evenly across maps, about 52% of groups would be mixed; per-map
success is instead bimodal. GRPO signal peaks for maps with success near 0.5
(mixed with 99% probability for groups of 8). Training reward was flat (0.914 vs
0.908 between epoch halves), and 304 of 706 failed rollouts were on 38 maps that
never succeeded, which GRPO cannot learn from.

**Multi-sample evaluation.** Job `6267411` (946 GPU-seconds) drew 16 samples per
test map from both checkpoints with the evaluation decoding settings. This
diagnostic is not the benchmark protocol.

| | LIVR rerun | Text-only RL |
| --- | ---: | ---: |
| Mean accuracy over 16 samples | 81.56% | 81.12% |
| Single-sample accuracy s.d. across draws | 0.66 pts | 0.53 pts |
| Maps never solved / always solved | 60 / 312 | 65 / 317 |
| Maps with success between 0 and 1 | 28 | 18 |

RL minus LIVR, paired by map: -0.44 points (sampling s.e. 0.19; 19 maps better,
20 worse, 361 unchanged). This interval reflects evaluation sampling only, not
training-run variance, which two LIVR runs suggest is larger (their single-sample
scores differ by 1.25 points; two independent samples of one checkpoint would be
expected to disagree on about 10.5 maps, versus 31 observed between runs). RL made
per-map outcomes more deterministic, moving intermediate maps toward always or
never solved, consistent with sharpening an already confident policy.

## Evidence

Per-map samples from the variance check are under `docs/experiments/multisample-6267411/`. Compact evidence is committed under `docs/experiments/livr-6253300/` (both stages'
step losses, per-map outcomes, summary) and `docs/experiments/text-rl-6256423/`
(step log, per-map rollout rewards, per-map outcomes, summary). Full responses,
manifests and checkpoints remain under `~/scratch/SPARA/results/` for
`asingirikonda3` on PACE-ICE.

GPU accounting against the 4-hour ceiling approved on 2026-10-13: 10,591 of 14,400
seconds used, including a 2-second failed submission (job `6253256`, missing
results directory). Recorded in `~/scratch/SPARA/logs/budget.json`.
