# Evidence for JEPA judge run 6951675

Step 2 of the [detour-learning spec](../../DETOUR_LEARNING_SPEC.md), specified in the
[JEPA judge spec](../../JEPA_JUDGE_SPEC.md). Can a transition model learned in LIVR's
frozen latent space stand in for the step simulator in guided search?

- **Checkpoint:** the LIVR rerun's final Stage 2 weights (`baseline-6253300`), frozen.
- **Search setting:** soft veto λ = 10 with re-opening, guard of 50,000 expansions
  ([step 1a″](../../GUIDED_SEARCH_SPEC.md#step-1a-result-cpu-2026-11-11)).
- **Compute:** one H200, 1,715 GPU-seconds by Slurm's elapsed time (28 min 35 s;
  peak 16.2 GiB).
  - **Against the estimates:** the first estimate was 20–25 GPU-minutes, revised to
    30–45 before submission.
  - **In the job:** reading states took 734 s, training 97 s, and search 595 s. Search
    split into 46 s exact, 237 s JEPA and 312 s final verdict only. Loading and checks
    took the rest.
  - **Pass time:** search passes took about 0.08 s each, half of the 0.17 s the revision
    assumed.
- **Source:** the hashes in manifest.json match the local `jepa.py`, `cli.py`,
  `search.py`, `route_search.py`, `configs/jepa_judge.json` and the job script.

## Readings

**Validity: all checks passed.**

| Check | Threshold | Result | |
| --- | --- | --- | --- |
| Training layouts disjoint from the released splits, `fresh/probe` and `fresh/detour-heldout` | none shared | 0 shared with each | passed |
| Batched walk state against a single pass, 20 states | cosine ≥ 0.99 | minimum 0.9997 | passed |
| LIVR's move probabilities at step 0's 52 points | within 0.01 | maximum difference 5.0e-7 | passed |
| Exact arm solves | ≥ 90% of 97 | 97 of 97 | passed |

**Primary: the learned judge hurts search.**
- **Within 64 policy calls:** the JEPA arm solved 29 of 97 maps; final verdict only
  solved 55.
- **Paired by map:** 1 solved by the JEPA arm only and 27 by final verdict only. Exact
  McNemar p = 2.2e-7.

**Secondary: not ready.** 29 of 97, where 88 are needed.

**Where the outcome table points.**
- **Row:** "No difference or hurts": LIVR's frozen states do not carry what a judge
  needs.
- **The oracle:** the outcome read from LIVR's true state *after* the move misses 96.6%
  of the holes taken.
- **Rule:** the information does not appear once the move is written, so by the spec's
  rule the encoder is the gap, not the predictor.
- **Caveat:** the oracle trains only on moves taken, and few of those enter a hole
  (609 of 7,710 in evaluation). Under argmax, a weak signal at that imbalance shows up
  as near-total misses.

## Search on the 97 unseen detour maps

| Arm | Solved within 64 calls | Stops | Policy calls, median (solved maps; all maps, unsolved = 65) | LIVR passes |
| --- | --- | --- | --- | --- |
| Exact simulator | 97 | solved 97 | 5; 5 | 564 |
| JEPA judge | 29 | solved 29, frontier 49, budget 19 | 7; 65 | 2,880 |
| Final verdict only | 55 | solved 55, budget 42 | 15; 47 | 3,851 |

| Grid size | Maps | Exact | JEPA | Final verdict only |
| --- | --- | --- | --- | --- |
| 3 x 3 | 2 | 2 | 1 | 2 |
| 4 x 4 | 10 | 10 | 7 | 9 |
| 5 x 5 | 14 | 14 | 8 | 11 |
| 6 x 6 | 36 | 36 | 9 | 23 |
| 7 x 7 | 18 | 18 | 2 | 7 |
| 8 x 8 | 17 | 17 | 2 | 3 |

- **By population:**
  - test: 40, 14 and 28 of 40;
  - `fresh/probe`: 57, 15 and 27 of 57.
- **Routes:** final verdict only found the exact arm's route on 40 of its 55 maps; the
  JEPA arm on 25 of its 29.
- **Refused answers per map:** JEPA 10.5, final verdict only 9.6.
- **Expansion guard:** no search hit it.

## Judge accuracy

All rates use argmax decisions.

| Moves judged | Judge | Miss rate | False-alarm rate | Arrivals missed / invented | Accuracy |
| --- | --- | --- | --- | --- | --- |
| Evaluation walks, 297 maps (26,560 moves; 5,375 into holes; 335 arrivals) | JEPA | 0.757 | 0.022 | 115 / 43 | 0.824 |
| | Probe | 0.775 | 0.021 | 104 / 56 | 0.822 |
| Moves taken on those walks (7,710; 609 into holes; 313 arrivals) | JEPA | 0.785 | 0.025 | 102 / 16 | 0.900 |
| | Oracle | 0.966 | 0.002 | 80 / 29 | 0.908 |
| Moves the JEPA arm judged in search (2,239 into holes; 8,046 not; 837 arrivals) | JEPA | 0.852 | 0.036 | 232 / 443 | |

**Against step 1a″'s target.** The ε₉₀ for the setting is 0.02 for both error types per
transition, within 64 calls.
- **Misses:** the JEPA's miss rate is about 40 times that.
- **False alarms:** its false-alarm rate is close to it.
- **Arrivals,** which the sweep never modelled: the search missed 232 arrivals and
  invented 443.

**By grid size, JEPA miss rate on the evaluation walks:**
- 3 x 3: 0.41
- 4 x 4: 0.67
- 5 x 5: 0.71
- 6 x 6: 0.80
- 7 x 7: 0.85
- 8 x 8: 0.83

False alarms stay between 0.014 and 0.032. The search maps and `fresh/detour-heldout`
differ by under 0.02 on each rate.

**Imagination.** The move taken j steps ahead was judged two ways by the same head: from
a state imagined by rolling the predictor j steps, and from LIVR's true state there.

| j | Pairs | Miss rate, imagined / true | False-alarm rate, imagined / true | Arrivals missed, imagined / true |
| --- | --- | --- | --- | --- |
| 1 | 6,522 | 0.880 / 0.888 | 0.066 / 0.015 | 189 / 102 |
| 2 | 5,453 | 0.870 / 0.890 | 0.100 / 0.015 | 219 / 102 |
| 3 | 4,465 | 0.860 / 0.919 | 0.124 / 0.015 | 250 / 102 |

**Layer choice and training.**
- **Layer 28 was selected.** Validation outcome loss by layer:
  - 12: 0.462
  - 20: 0.380
  - 28: 0.339
  - 35: 0.348
- **Early stopping:** at layer 28 the JEPA's best epoch was 4 of 10, so it stopped
  early. The probe also stopped after 10 epochs, the oracle after 13.

## Post-hoc observations (not readings)

**Final verdict only is far stronger with LIVR as the policy than with the stand-in
priors.**
- **Count:** it solved 55 of 97 within 64 calls, with a median of 15 calls on solved
  maps. In step 1a′ the stand-ins solved 0, or 9 with moves off the grid ruled out.
- **The spec's assumption failed:** it called the primary a low bar, and it was not.
- **By size:** it falls with size, solving 23 of 36 maps at 6 x 6, 7 of 18 at 7 x 7 and
  3 of 17 at 8 x 8.
- **For step 1b:** search with only the final verdict, with LIVR as the policy, could
  supply step 1b's routes on small and medium maps with no simulator and no judge.

**Missed arrivals can lose a route outright (a likely cause of the JEPA arm's 49
frontier stops).**
- **The mechanism, by construction:** the search walks onto the goal, closes its cell,
  and from then on every move into the goal is a repeat.
- **Re-opening does not help:** it runs only after a refused answer.
- **Not confirmed:** the search logged 232 missed arrivals, but the run did not record
  which stops they caused.

**Predicting the next state did not help.** The JEPA and the probe differ by at most
0.02 on every rate, and rolling the predictor forward adds false alarms and missed
arrivals.

**The deepest layers held the most outcome information,** but no layer separated holes
from safe cells well.

## Post-hoc CPU analysis: thresholds, layers and search (2026-11-11)

Not fixed before the run. Computed from the saved states and weights by
[scripts/analyze_jepa_thresholds.py](../../../scripts/analyze_jepa_thresholds.py), in 44
minutes of CPU and no GPU; the output is thresholds.json.

**Check.** The reloaded judges reproduce the run's argmax rates exactly, on all four
populations below.

**No threshold makes the judge usable.** A move into a hole against any other move, on the
evaluation walks:

| Judge | AUROC | Equal-error rate | Miss rate at false alarms 0.02 / 0.05 / 0.1 / 0.2 |
| --- | --- | --- | --- |
| JEPA | 0.806 | 0.295 | 0.764 / 0.667 / 0.560 / 0.402 |
| Probe | 0.797 | 0.301 | 0.777 / 0.677 / 0.572 / 0.429 |
| JEPA, moves taken | 0.788 | 0.304 | 0.796 / 0.718 / 0.594 / 0.427 |
| Oracle, moves taken | 0.732 | 0.335 | 0.875 / 0.798 / 0.700 / 0.534 |

- **No threshold brings both rates under 0.2,** let alone step 1a″'s 0.02–0.05.
- **Thresholds chosen on the validation maps drift.** One set for 0.05 false alarms
  there gives 0.085 false alarms and 0.58 misses on the evaluation maps.
- **By size, the JEPA's AUROC falls** from 0.93 at 3 x 3 to 0.76 at 8 x 8. It is 0.79 on
  the search maps and 0.81 on `fresh/detour-heldout`.

**The oracle's verdict was not an artefact of argmax.**
- **Threshold-free:** the oracle's AUROC (0.73) is below the JEPA's (0.81). A hole is no
  easier to read once the move is written.
- **No layer does better.** Retrained on CPU at every saved layer, the probe's AUROC stays
  between 0.75 and 0.79 and the oracle's between 0.72 and 0.75:

| Layer | Probe | Oracle |
| --- | --- | --- |
| 12 | 0.751 | 0.748 |
| 20 | 0.786 | 0.733 |
| 28 | 0.794 | 0.732 |
| 35 | 0.767 | 0.724 |

**Arrivals are readable; holes are not.**
- **AUROC:** P(done) separates arrivals almost perfectly, 0.997 for every judge.
- **What a threshold would do:** missing 5% of the 335 arrivals, it would call 1.4% of
  other moves arrivals. Argmax missed 34% of arrivals.
- **So the run's arrival errors came from argmax;** its hole errors did not.

**Search at error rates on the JEPA's curve.**
- **Setup:** step 1a″'s noisy-judge search, with the toward-goal stand-in as the policy,
  errors independent per transition, and arrivals judged correctly. Maps solved of 97,
  mean of 5 seeds:

| Point | Miss rate | False alarms | λ = 10 with re-opening, within 64 / 1,000 calls | λ = 10 alone, within 64 |
| --- | --- | --- | --- | --- |
| Exact judge | 0 | 0 | 97 / 97 | 97 |
| JEPA argmax, evaluation walks | 0.757 | 0.022 | 19.0 / 41.0 | 36.0 |
| JEPA argmax, in search (the run) | 0.852 | 0.036 | 14.6 / 36.0 | 31.8 |
| Curve, false alarms 0.05 | 0.667 | 0.05 | 25.2 / 47.0 | 38.6 |
| Curve, false alarms 0.1 | 0.560 | 0.1 | 30.2 / 49.4 | 40.0 |
| Curve, false alarms 0.2 | 0.402 | 0.2 | 36.4 / 51.2 | 40.6 |
| Curve, false alarms 0.3 | 0.289 | 0.3 | 39.2 / 53.6 | 45.2 |
| Curve, false alarms 0.5 | 0.091 | 0.5 | 46.2 / 56.2 | 46.4 |

- **Best within 64 calls:** about 46 maps, where 87 are needed.
- **Re-opening costs calls within 64.** λ = 10 alone solves more at every noisy point. The
  run used re-opening by the rule registered at 1,000 calls.
- **The level is understated.** At the run's in-search rates the simulation gives 14.6
  maps; the real JEPA arm, guided by LIVR, solved 29. Compare points with each other, not
  with the run's arms.

**Limits.**
- **Walk states only.** The states the search judged were not saved, and its miss rate was
  higher there (0.85 against 0.76).
- **The simulation's errors are independent per transition.** Real errors cluster by cell
  and by map.

**For step 3: G's encoder separates holes better (post hoc, from other runs' evidence).**
- **The measure:** a ridge probe for a hole in the direction about to be taken, on each
  model's own greedy routes. It is computed from probe-steps.jsonl of
  [step grounding 6926184](../step-grounding-6926184/README.md) and steps.jsonl of
  [planning probe 6887458](../planning-probe-6887458/README.md).
- **The models:** G is LIVR's Stage 2 retrained with an outcome loss on the released maps.

| Maps | LIVR, layer 20 (fatal / steps) | G, layer 24 (fatal / steps) |
| --- | --- | --- |
| Test | 0.665 (68 / 1,026) | 0.864 (67 / 1,044) |
| Fresh 5 x 5 and 6 x 6 | 0.596 (50 / 630) | 0.805 (54 / 646) |
| Fresh 7 x 7 and 8 x 8 | — | 0.779 (87 / 669) |

- **Outcome supervision raised the AUROC by about 0.2.** That is still far from the near
  perfect separation that both rates under 0.05 would need.
- **Caveat:** the routes and layers differ between the two models.

## Files

| File | Contents |
| --- | --- |
| summary.json | Readings, validity checks, search per arm (solved, stops, calls, passes), the JEPA's in-search rates, judge accuracy (overall, by population and size), imagination, training histories, data counts, timing |
| summary-before-search.json | The interim summary written before the search phase |
| searches.jsonl | One row per map: each arm's solved, stop, policy and verifier calls, refused answers and route |
| search-exact.json, search-jepa.json, search-verdict_only.json | Each arm as it finished: solved, stops, LIVR passes, seconds; the JEPA arm's in-search rates |
| thresholds.json | Post-hoc CPU analysis: argmax check, AUROC and operating points per judge, population and size; thresholds from the validation maps; probe and oracle per layer; search at points on the JEPA's curve |
| manifest.json | Configuration, packages, source hashes, settings, fresh-set manifests and checks. The PACE root is written as `$SPARA_REMOTE_ROOT/` |

Not copied; on PACE scratch at `$SPARA_REMOTE_ROOT/results/jepa-judge-6951675/`:
- `states.pt` (1.4 GB), kept for CPU analyses;
- `jepa.pt` (37 MB, the JEPA, probe and oracle weights);
- `items-fit.jsonl`, `items-val.jsonl` and `items-eval.jsonl` (19 MB);
- the per-arm `searches-<arm>.jsonl` (the rows of searches.jsonl).
