# Latent battery results

Updated 2026-11-08. One checkpoint, greedy decoding, no training. This is a
measurement pass on LIVR's latent tokens, run before any training on them, to
inform latent RL (condition C) and objectives on the latents themselves.

## Summary

On the original LIVR checkpoint (`6189457`), the route answer is decided by the
four latent tokens, even when the answer can also attend to the image directly.

- **Swapping latents swaps the answer.** Given another map's latents, the model
  answers that map's route in 351 of 400 cases and never gives its own route.
  The swapped answer solves the donor map 320 times, against 328 for the donor's
  own answer. Success on the model's own map falls from 328 to 12.
- **The latents are map-specific.** Replacing them with the training-set mean for
  the grid size gives one or two fixed routes per size (9/400 solved). Dropping
  them from the prompt gives 44/400, although the image remains visible.
- **The latents hold both the map and the route, at different depths.** Linear
  probes trained on training maps and scored on test maps find the goal and start
  cells nearly perfectly in layers 16 to 20, holes only partly, and the route's
  first move and length mainly in late layers. Hole information is the weakest
  and falls with grid size.
- **Failures do not follow the probe's hole errors.** The holes the model walks
  into are read as safe by the hole probe no more often than holes it avoids, so
  this analysis does not tie failures to missing hole information.
- **Image-token features distinguish cells.** A linear probe recovers each safe
  cell's row and column from its image-token state at 96 to 100% accuracy, and no
  two cells in one image are near-identical.

LIVR's reliance on the latents is learned rather than forced: the vanilla SFT
control, trained without latent tokens, scores 320/400
([arm A](experiments/sft-a-6259811/README.md)).

## Setup

Job `6848878` ran `spara battery` ([code](../src/spara/latent_battery.py)) on one
H100 in 2,021 GPU-seconds, after a pilot on two maps per size (job `6848180`,
284 GPU-seconds). The CPU analysis ([script](../scripts/analyze_latent_battery.py))
ran as Slurm job `6850902`. Compact outputs are in
[experiments/latent-battery-6848878](experiments/latent-battery-6848878/README.md).

Each of the 400 test maps was answered under two attention modes and four latent
conditions:

| Mode | Attention |
| --- | --- |
| Unmasked | Stage 2 attention: every token after the image can read it |
| Bottleneck | Stage 1 mask: only the latent tokens read the image |

| Condition | Latent states |
| --- | --- |
| Own | The map's own, recomputed |
| Swap | Another test map's, of the same grid size and a different layout (seeded assignment) |
| Mean | The mean over training maps of that grid size |
| Removed | The prompt is re-encoded without latent tokens |

Swap and mean overwrite the output of every decoder layer at the four latent
positions with the replacement states for that mode. Answers use greedy decoding
with the model's repetition penalty (1.05) and at most 128 new tokens; no answer
was truncated. The reference benchmark evaluation samples instead (temperature
0.7, top-p 0.9), the likely reason greedy own answers score 328/400 here against the
323/400 recorded for this checkpoint. Reference NLL is the mean per-token
negative log-likelihood of the unmasked own answer, teacher-forced under each
condition.

Before the main pass, re-inserting each of the first four maps' own recorded
states reproduced its route in both modes (NLL differences at most 8.2e-4).

## Interventions

Maps solved, by grid size:

| Mode / condition | 3 x 3 | 4 x 4 | 5 x 5 | 6 x 6 | Total | Reference NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unmasked / own | 93 | 93 | 81 | 61 | **328** | 0.010 |
| Unmasked / swap | 5 | 2 | 3 | 2 | 12 | 4.34 |
| Unmasked / mean | 5 | 4 | 0 | 0 | 9 | 3.49 |
| Unmasked / removed | 12 | 15 | 10 | 7 | 44 | 1.40 |
| Bottleneck / own | 92 | 86 | 81 | 59 | **318** | 0.022 |
| Bottleneck / swap | 5 | 2 | 2 | 1 | 10 | 4.32 |
| Bottleneck / mean | 4 | 4 | 1 | 0 | 9 | 3.39 |
| Bottleneck / removed | 10 | 3 | 0 | 1 | 14 | 1.61 |

Every intervention loses 284 to 319 maps against own in its mode and gains none
(exact McNemar p < 1e-85 for each).

**What the swapped answer follows.** In unmasked mode the swapped answer is the
donor's own route exactly in 99, 89, 90 and 73 of 100 cases for sizes 3 to 6,
and the recipient's own route in none. Of the 49 other answers, 25 share their
first two moves with the donor's route and 7 with the recipient's. In bottleneck
mode it is the donor's route in all 400.

**Blind and direct-image baselines.** Under the bottleneck mask, mean and removed
latents leave the answer without image information: it is one fixed route per
grid size (9/400 and 14/400 solved). The unmasked rows measure what the direct
image pathway adds without informative latents: 44/400 with latents removed,
and more departures from the donor's route at larger sizes. The removed prompt
is off-distribution for a model always trained with latents, so it is weaker
evidence than the swap.

## Probes

Ridge probes in the four latents' concatenated states (8,192 features) are fitted
on the training maps of one grid size (100 to 400) and scored on that size's 100
test maps. Strength is chosen by leave-one-out error on the training maps. Layer
L is the output of decoder layer L (one-based; 36 is the last, before the final
norm). Values are per grid size, 3 / 4 / 5 / 6. Holes report balanced accuracy
per cell, averaged; goal and start are cell accuracy; the first move is scored
correct when it begins some shortest route.

**Unmasked latents**

| Target | Chance | Layer 12 | Layer 20 | Layer 36 |
| --- | --- | --- | --- | --- |
| Hole at each cell | .50 | .88 / .87 / .78 / .75 | .89 / .81 / .74 / .74 | .78 / .66 / .60 / .58 |
| Goal cell | .18 / .10 / .09 / .06 | .98 / .99 / .98 / .92 | 1.0 / .99 / .97 / .96 | .97 / .94 / .86 / .76 |
| Start cell | .14 / .11 / .08 / .08 | .96 / .89 / .94 / .68 | .94 / .96 / .98 / .88 | .87 / .75 / .82 / .50 |
| Route length (R²) | 0 | not decodable | .82 / .79 / .49 / .69 | .85 / .81 / .69 / .73 |
| First move | .32 / .36 / .40 / .43 | .70 / .71 / .58 / .48 | .86 / .82 / .84 / .83 | .97 / .96 / .92 / .84 |

Bottleneck latents give nearly the same values. Map content (holes, goal, start)
peaks between layers 8 and 20 and then declines as route content rises. Hole
information is the weakest map content and falls with grid size, as success
does; the next section tests whether failures follow it.
Negative R² in early layers (down to about -3) means route length is not linearly decodable
there with 100 to 400 training maps.

**Last prompt token** (the position that emits the first answer token), layer 36:

| Target | Unmasked | Bottleneck |
| --- | --- | --- |
| Hole at each cell | .68 / .65 / .59 / .56 | .54 / .50 / .50 / .50 |
| Goal cell | .90 / .68 / .48 / .36 | .31 / .43 / .35 / .36 |
| First move | .92 / .92 / .91 / .85 | .88 / .95 / .91 / .84 |
| Route length (R²) | .83 / .76 / .75 / .75 | .87 / .79 / .80 / .74 |

The last token carries the route nearly as well as the latents do, but less of the
map, even when it can read the image. Under the bottleneck, where it reads the
image only through the latents, its hole probe stays near chance (at most .59)
at every probe layer.

## Do failures follow missing hole information?

The predictions and the primary layer (12, the best hole accuracy averaged over
sizes) were fixed before this analysis ran
([script](../scripts/analyze_battery_failures.py), CPU job `6878532`). It uses
the unmasked own answers and the latent hole probe above, and walks each route
with the reference evaluator's rules.

Of the 72 failures, 66 enter a hole (7, 7, 15 and 37 for sizes 3 to 6) and 6 end
off the goal. At layer 12 the probe read 24 of the 66 fatal holes as safe (36%).
That is no more than expected from holes at the same cell in solved maps (20.7
expected, one-sided p = 0.16), or from holes beside the route in solved maps of
the same size, which the model avoided (24.2 expected, p = 0.57). Failed maps
have more misread cells than solved maps (4.5 against 2.3; AUC 0.72), but not
when grid size and hole count are held fixed (stratified permutation, p = 0.45).
At the other eight layers, 2 of 24 one-sided tests fall below 0.05 (layer 4
position-matched, p = 0.019; layer 36 map-level, p = 0.036), about what chance
gives.

The probe's hole errors therefore do not predict which holes the model enters.
The model also avoids many holes the probe misses: at layer 12, 120 of the 400
holes beside solved routes are read as safe. The linear probe understates what
the model uses, so this test cannot tie failures to missing hole information.
Map content (layers 8 to 20) and route content (late layers) both sit at the
latent positions, and the swap shows the answer follows them, so a wrong route
is carried by the latents either way. This test checks only linearly decodable
map content at one layer; it cannot say whether errors start in reading the map
or in planning from it.

## Representation geometry

Within each grid size, the raw latents of different test maps are nearly
parallel (mean cosine 0.98 to 1.00), because a component shared by all maps
dominates them, as is usual for transformer hidden states. After subtracting
the size's mean, maps are on average orthogonal (centred cosine -0.009 to
0.035). The participation ratio of the centred latents is 2.7 to 7.3, but a few
high-variance channels dominate that figure; with each channel z-scored it is
10 to 18, out of a maximum of 99 for 100 maps. So the latents have not
collapsed, since probes decode map content from them, but their variation
across maps lies in comparatively few directions. Values average the four grid
sizes; the two modes agree within 0.5.

## Image-token cells

Each cell's feature is the overlap-weighted mean of the image-token states
covering it (taking the merged tokens as row-major and the map as filling the
image). Across test maps and layers, a safe cell's most similar other cell in
the same image has median cosine 0.93 to 0.97, and at most 0.3% of safe cells
have a neighbour above 0.99. A ridge probe from a safe cell's feature recovers
its row and column (row accuracy x column accuracy) at 1.00 for every size in
layers 4 to 16 and 0.96 to 0.99 at layer 36. Safe cells look alike, so the
probe reads position from the model's positional information or the cell's
surroundings, not from its own appearance. Cell-level prediction targets are
therefore distinguishable; that is a precondition, not evidence of benefit.

## Caveats

- One checkpoint and seed. Condition C must start from the LIVR rerun `6253300`
  used for text-only RL (condition B); the same pattern is expected there but
  not shown.
- The swap overwrites all 36 layers at the latent positions, so it shows that the
  latents determine the answer, not at which layer their content enters.
- Probes are linear and fitted on 100 to 400 maps for thousands of features; they
  are lower bounds on what is decodable. Training maps were seen in LIVR
  training, so their states may differ from test maps' states.
- Seven 3 x 3 test layouts and one 4 x 4 layout also occur in training, and five
  layouts repeat within the test set; 5 x 5 and 6 x 6 have no overlap.
- Greedy answers differ from the sampled reference protocol; success counts here
  are not comparable to reference scores.

## Implications for the next phases

These are hypotheses for planning, not results.

- **Latent RL (condition C).** The answer is determined by the latent states, so
  adjusting them, as Scaffolding RL does, acts on the variable that sets the
  route.
- **Latent objectives.** Hole content is the least linearly decodable part of
  the map in the latents and degrades with size, but the probe's hole errors do
  not predict failures. An auxiliary objective that makes the latents carry the
  map more completely, such as answering further questions about the same map
  or predicting embeddings of the map's content (in the style of JEPA), is
  therefore not yet supported as a fix for failures. Whether failures start
  in the latents' map content or in the planning that follows at the same
  positions is still open.
- **Per-step prediction.** Image-token states identify cells, so per-cell targets
  for an action-conditioned predictor are available from the frozen model.
