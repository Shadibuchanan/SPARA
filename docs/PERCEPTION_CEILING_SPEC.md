# Perception ceiling: round 2 with the map also given as text

Status, 2026-11-15: run and read. O2 (job 7076349) and T2 (job 7077465) both show no
detectable difference from E2. With the map known exactly, the policy fails as often:
**at 7 x 7 and 8 x 8, choosing moves limits, not perception.** See
[Result](#result-o2-job-7076349) and [T2's result](#result-t2-job-7077465). T2 was added
after O2's result, with its readings fixed before it ran.

## Question

How much of round 2's failure on 7 x 7 and 8 x 8 maps is perception?

In [round 2](EXPERT_ITERATION_ROUND2_SPEC.md#result-jobs-70651687065176), walking into a
hole ends 307–310 of 400 attempts for E, E2 and EB2. The layer-12 hole probe reads the
hole that ended an attempt as often as the holes the route passed safely. So the probe
cannot say whether the model fails to see holes or sees them and moves badly. This run
settles that by removing perception as a source of error.

## Design

**One new arm, O2.** It is E2 with one change: every prompt, in training and in
evaluation, also states the map as text after the image.
- **The text:** one line per grid row, top to bottom, with `S` start, `G` goal, `H` hole
  and `F` safe. About 99 tokens on an 8 x 8 map.
- **The same as E2 in everything else:** E's weights (job 7063729), the 3,000 rows from
  search 7065168, the shuffle and the two unmasked phases of 1,496 and 1,504 rows.
- **O2 has perfect perception by construction.** Its success bounds what better
  perception could add to E2.
- **Layouts are used as input only for this diagnostic.** O2 is not a candidate model.
  SPARA's conditions still see the image only.

**The zero-shot read.** In the same job, before training, E answers the 400 new held-out
maps with the text map. E never saw text maps in training, so this reading is descriptive.

**Evaluation:** unmasked greedy decoding with the text map on the four populations of
round 2: the 400 new held-out maps, round 1's 240 held-out maps, and the first 98 new
pool maps. No masked read, latent states or benchmark, because O2 is not deployable.

## Readings (fixed before the run)

**Primary: O2 against E2** on the 400 new held-out maps. Paired by map, exact McNemar,
two-sided. E2's answers come from job 7065175.

| Result | Reading | Next |
| --- | --- | --- |
| O2 solves at least 40 more (10 points), p < 0.05 | Perfect perception helps a lot | Perception is the main limit. Label-free perception work (for example, predicting the image from the latents) can pay off. |
| O2 solves more, by fewer than 40, p < 0.05 | Perfect perception helps a little | Perception limits some failures. Weigh it against policy work by the size of the gain. |
| No detectable difference | Choosing moves limits | Better perception cannot help at these sizes. Spend effort on search and the policy (round 3). |
| O2 solves fewer, p < 0.05 | The text map harms | Check the training loss and the prompt before reading anything else. |

- **Why 40 maps:** E2 gained 16 over E in round 2. A ceiling gain under 40 would be about
  two rounds of expert iteration, which are cheaper than new perception work.
- **Power:** with exact McNemar, a 3:1 split of discordant maps reaches p < 0.05 with
  about 20 of them.

**Descriptive:**
- E's zero-shot success with the text map, against E's success without it.
- Failures per arm (into a hole, stops short, passes the goal) for E, E2, O2 and the
  zero-shot read. If O2 still walks into holes it was told about, those are move-choice
  failures.
- O2 against E2 on round 1's 240 held-out maps.

**Validity:** O2 and E2 train on the same number of rows, and every arm answers the same
maps. The readings are withheld if either fails.

## Budget

One H200 job of about 25 minutes: E2 took 1,718 seconds, including masked reads, states
and the benchmark, which O2 skips, plus about 4 minutes for the zero-shot read. No
weights are saved; scratch has 7.5 GB free.

## Result (O2, job 7076349)

Evidence: [docs/experiments/perception-ceiling-7076349/](experiments/perception-ceiling-7076349/).
1,553 GPU-seconds. Validity passed: O2 and E2 trained on the same 3,000 rows and answered
the same maps.

**Primary: no detectable difference; choosing moves limits.** On the 400 new held-out
maps, O2 solves 77 and E2 72: 20 maps only O2 solves, 15 only E2, p = 0.50. By size: 44
against 43 at 7 x 7 and 33 against 29 at 8 x 8.

**Descriptive.**
- **O2 walks into a hole on 307 of 400 maps,** against 308 for E2, although its prompt
  lists every hole.
- **Zero-shot:** E with the text map solves 45, against 56 without it: 4 maps only with
  the text, 15 only without, p = 0.019. Untrained text hurts.
- **Round 1's 240 held-out maps:** O2 92, E2 98 (p = 0.31).
- **Training loss** is close to E2's: last quarter 0.068 and 0.050 for the two phases,
  against 0.068 and 0.048.

**What else the null could mean.** The image alone was enough for O2 to fit its training
answers, so O2 may have learned to ignore the text. The equal training loss fits that
reading, and so does the null. T2 tests it.

## Follow-up: T2, the text map alone

**Design.** T2 is O2 with the image replaced by a white image of the same size, in
training and in evaluation. The text map is then its only map. Everything else is as O2,
including the zero-shot read of E.

**Reading (fixed before the run): T2 against E2** on the 400 new held-out maps, exact
McNemar.

| Result | Reading | Next |
| --- | --- | --- |
| No detectable difference | The policy plans from the text map as well as from the image | O2's null stands: with the map known exactly, the policy still fails as often. Choosing moves limits; spend effort on search and the policy. |
| T2 solves fewer, p < 0.05 | The policy plans worse from the text map; the ceiling is inconclusive | O2 may have ignored the text. A ceiling needs a form of the map the policy can use. |
| T2 solves more, p < 0.05 | The policy plans better from the text map | The image costs something the text does not. Perception limits, despite O2's null. |

**Descriptive:** T2 against O2, T2's failures by kind, and E's zero-shot success with a
blank image and the text map.

**Budget:** one H200 job of about 26 minutes, as O2.

## Result (T2, job 7077465)

1,484 GPU-seconds. Validity passed: T2 trained on the same 3,000 rows as E2 and answered
the same maps. Evidence is in the same folder as O2's.

**Reading: the policy plans from the text map as well as from the image.** On the 400 new
held-out maps, T2 solves 69 and E2 72: 23 maps only T2 solves, 26 only E2, p = 0.78. By
size: 43 and 43 at 7 x 7, 26 against 29 at 8 x 8.

**T2 learned to read the text map.** Before training, E with a blank image and the text
map solves 3 of 400. After the same 375 steps as E2, T2 solves 69. Its training loss ends
higher than E2's (0.106 against 0.048 in the last quarter of phase 2), and its held-out
success is the same.

**With the map known exactly, the policy still fails.** T2 walks into a hole on 316 of
400 maps, O2 on 307 and E2 on 308. T2 against O2: 69 against 77, p = 0.31.

**Conclusion.** O2's null stands. Removing perception as a source of error, whether
alongside the image (O2) or in place of it (T2), changes success by no detectable amount.
At these sizes the model knows where the holes are and still chooses routes through
them. Better perception of the map cannot raise success here. The gain has to come from
choosing moves: search and the policy, as expert iteration does (E2 against E: 72
against 56).

**Seen maps:** O2 solves 30 of the first 98 new pool maps and T2 18, against E2's 33.

## Limits

- **One training run per arm.**
- **The text is a second input, not a replacement.** O2 can ignore the image. That is
  intended: the question is what the policy does when perception is solved.
- **A null result can also mean the policy cannot use a text grid,** not that it chooses
  well. T2 answers this for the trained policy: from the text alone it matches E2.
- **The policy answers in one pass,** a whole route with no intermediate reasoning. The
  conclusion is about this policy. A policy that reasons step by step might use perfect
  perception differently.
- **FrozenLake at 7 x 7 and 8 x 8 only.** At 3 x 3 to 6 x 6, the latent battery and the
  masked reads already showed the latents carry nearly everything the answers use.
