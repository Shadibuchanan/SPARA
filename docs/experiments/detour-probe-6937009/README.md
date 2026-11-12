# Evidence for detour reach check 6937009

Step 0 of the [detour-learning spec](../../DETOUR_LEARNING_SPEC.md): how much probability
does LIVR give the correct move where a detour is needed?

- **Checkpoint:** the LIVR rerun's final Stage 2 weights (`baseline-6253300`).
- **Routes:** LIVR's recorded greedy routes on the 400 test maps
  ([planning probe 6887458](../planning-probe-6887458/README.md)) and the 400
  `fresh/probe` maps ([exploration probe 6884608](../explore-probe-6884608/README.md)).
  No generation in this run.
- **Compute:** one H200, 286 GPU-seconds (4 min 46 s of wall time, 65 s of forward
  passes, peak 20.4 GiB). The estimate was about 5 GPU-minutes.
- **Decision points:** 275 on unseen maps (82 test, 193 fresh), plus the 98 training
  detour maps for the validity check. One forward pass each.
- **Source:** the hashes in manifest.json match the local `detour_probe.py`, `cli.py`
  and `planning_probe.py`.

## Reading: withheld by the rule

The second validity check failed, so the spec's rule withholds the reading.

| Validity check | Threshold | Result | |
| --- | --- | --- | --- |
| Teacher-forced argmax (with the repetition penalty) equals the greedy route's next move | ≥ 90% | 275 of 275 (100%) | passed |
| Released move's median probability at the first detour decision, 98 training detour maps | ≥ 0.9 | 0.82 | failed |

The recorded test completions equal the canonical answer tokens of their routes on all
82 test points.

### Why the second check failed (post hoc)

The check assumed Stage 2 had fitted the training detour maps almost exactly. The spec
supported that with an answer loss "near 0.001", which was G's (0.00078). LIVR's
last-epoch answer loss is 0.013. On the 98 training detour maps, at the first point
where the released route must move away from the goal:
- the released move scores at least 0.9 on 40 maps and below 0.5 on 24;
- the argmax is a move into a hole on 20.

The evidence that the measurement itself is sound:
- **Same inputs as training.** All 1,000 released answers are character-identical to
  the canonical answer string the probe teacher-forces, so the token IDs match Stage 2's
  training inputs.
- **Confident moves score high.** At the 33 forced points where the greedy route did
  take an optimal away move, that move has median probability 0.89 and ranks first on
  31.
- **Greedy choices reproduce.** The first check's 100%.

So the failure reflects the checkpoint, not the probe. LIVR did not fully fit the
detour decision on its own training maps. Whether to read the primary measure anyway is for
the team; it is not re-labelled here.

## Primary measure (computed, unread)

52 forced points on unseen maps where the greedy route did not take an optimal move:
48 walk into a hole and 4 onto a safe tile that is not on a shortest route.

| | Points | Within reach (P ≥ 0.01) | Median P(optimal) | Zero under RL sampling | Rank of best optimal move (1/2/3/4) |
| --- | --- | --- | --- | --- | --- |
| All | 52 | 21 (40%) | 0.0076 | 45 (87%) | 0 / 27 / 22 / 3 |
| Test | 21 | 7 (33%) | 0.0076 | 17 (81%) | 0 / 13 / 7 / 1 |
| Fresh | 31 | 14 (45%) | 0.0098 | 28 (90%) | 0 / 14 / 15 / 2 |
| 4 x 4 | 3 | 0 | 0.0018 | 3 | 0 / 1 / 1 / 1 |
| 5 x 5 | 6 | 3 | 0.012 | 5 | 0 / 4 / 2 / 0 |
| 6 x 6 | 23 | 9 | 0.0076 | 20 | 0 / 13 / 9 / 1 |
| 7 x 7 | 8 | 3 | 0.0078 | 7 | 0 / 3 / 4 / 1 |
| 8 x 8 | 12 | 6 | 0.011 | 10 | 0 / 6 / 6 / 0 |

- Under the spec's rule, 40% would read as **partial** (20–50%). This is stated for
  reference only; the rule withholds it.
- The median sits just below the reach line, so the share depends on it. Descriptively:
  65% of points at 0.001, 40% at 0.01, 25% at 0.05, 15% at 0.1.
- The move the greedy route took has median probability 0.96. The model is confident
  in the wrong move.

## Secondary (no verdicts)

- **RL could not sample the detour.** At 45 of 52 primary points, the optimal moves
  get probability exactly zero under B and C's rollout settings: after temperature 0.7,
  top-p 0.9 leaves them outside the nucleus. The 7 exceptions have raw P(optimal) of
  0.24–0.39.
- **Deviation points (147 on unseen maps):** 71% within reach, median P(optimal)
  0.085, 63% zero under RL sampling. The move taken enters a hole at 69%. On detour
  maps (12): median P(optimal) 0.18.
- **Stopped points (43, 41 of them fresh):** the route ends with median probability
  0.986.
- **Repetition penalty:** at 6 deviation points, the raw softmax ranks an optimal move
  first. The checkpoint's repetition penalty (1.05) flips greedy decoding to the move
  taken. At 2 forced points it flips the other way, to the optimal move.
- **No chosen points:** no greedy route on an unseen detour map moved away from the
  goal where it did not have to.

## Exploratory (post hoc)

**With simulator pruning, the detour move is first or second.** At the 52 primary
points, remove the moves into a hole or off the grid. The policy's top remaining move
is then optimal at 38 points and second at the other 14. Policy-guided search that
prunes by the simulator, as step 1 sketches, would reach every one of these first
detour moves within two branches.

This covers only the first detour decision on each route. A detour may need several
moves away from the goal, and the later decisions are not measured here.

## Files

| File | Contents |
| --- | --- |
| summary.json | Validity, primary measure by split and size, secondary measures, run time |
| points.jsonl | One row per decision point: split, row, kind, step, cell, each move's outcome, raw and RL-sampling probabilities, rank, argmax. Training rows add `p_taken` |
| manifest.json | Run manifest: config, package versions, source hashes, data manifest, decoding settings and thresholds. Paths under the PACE results root are written as `$SPARA_REMOTE_ROOT/` |
