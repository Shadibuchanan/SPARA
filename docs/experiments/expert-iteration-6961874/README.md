# Evidence for expert iteration run 6961874–6961876

Step 1b of the [detour-learning spec](../../DETOUR_LEARNING_SPEC.md), specified in the
[expert iteration spec](../../EXPERT_ITERATION_SPEC.md). Can LIVR improve its own detour
routes from routes it finds by search, when the search sees only the verifier's accept or
reject?

- **Starting point:** the LIVR rerun's final Stage 2 weights (`baseline-6253300`), for
  the search policy and for both fine-tunes.
- **Jobs:** one H200 each, 2026-11-11. All three completed on the first submission.

| Job | Contents | Wall time | In the job | Peak memory |
| --- | --- | --- | --- | --- |
| 6961874, routes | Checks, the 97 detour maps, the 500 pool maps | 35 min 44 s | Pool search 1,247 s (17,032 reads); detour maps 308 s (3,799 reads) | 17.9 GiB |
| 6961875, E | Fine-tune on found routes, greedy on 338 maps, benchmark | 21 min 44 s | Training 375 steps, 792 s; last-quarter loss 0.042 | 55.1 GiB |
| 6961876, M | Fine-tune on the same maps' labels, greedy on 338 maps | 15 min 55 s | Training 375 steps, 747 s; last-quarter loss 0.040 | 55.1 GiB |

- **GPU-seconds:** 4,403 (73 GPU-minutes) by Slurm's elapsed time, within the estimate
  of 55–75.
  - **The search took longer than planned:** it estimated 8–14 minutes with batching and
    about 30 without, and took 36. Batching failed its check (below), so every read was
    one request per pass, about 0.073 s each.
  - **The detour-learning line so far:** 10,614 GPU-seconds (2 h 57 min). That is the reach
    check (286), detour scaling (4,210), the JEPA judge (1,715) and this run (4,403).
- **Source:** the hashes in each manifest match the local code for every file this run
  uses, with one exception, `src/spara/expert_iteration.py`. Since the run, its
  `compare` renames one output key, so the search job's seconds are no longer
  overwritten. Reverting that one line reproduces the hash the run recorded. Files
  differ that the run does not import: the protected PACE files and the package's
  `SOURCES.txt`.
- **Weights:** E's and M's final weights (15 GB each) are on PACE scratch at
  `$SPARA_REMOTE_ROOT/results/expert-train-*-696187*/stage2/final`. E's are kept for a
  possible second round.

## Validity: all checks passed, after the planned fallback

| Check | Threshold | Result |
| --- | --- | --- |
| Batched policy at the 52 primary points of step 0 | within 0.01 | **Failed:** 0.049 |
| Batched policy: 32 cached answers against single passes | within 0.01 | **Failed:** 0.059 |
| Fallback: one request per pass at the 52 points | within 0.01 | 5.0 × 10⁻⁷ |
| Every accepted route passes the reference evaluator | all | 326 of 326; rechecked on CPU |
| Pool and held-out labels pass the reference evaluator (both fine-tunes) | 100% | 2,200 of 2,200 |
| The held-out set and the pool share no layout with each other, `fresh/probe`, train or test | 0 | 0 in all 7 pairs |

- **The fallback ran as the spec planned.** The job switched to one request per pass,
  passed the 52-point check, and searched again.
- **Rerunning the 97 detour maps reproduced the JEPA run exactly:** all 97 had the same
  outcome and the same route, with 55 solved.
- **Why batching disagreed is not established.** The CPU tests matched batched and
  single passes, but they used a tiny stand-in model with a 2-token image, so the real
  vision encoder and real image sizes were never tested. A difference of 0.05 in a
  probability is large for rounding error alone.

## The search: 326 of 500 pool maps solved

| Map size | Solved |
| --- | --- |
| 3 x 3 | 11 of 11 |
| 4 x 4 | 57 of 70 |
| 5 x 5 | 105 of 161 |
| 6 x 6 | 153 of 258 |

- **Stops:** 326 solved and 174 at the budget.
  - **Policy calls:** median 26, and 13 on solved maps.
  - **Verifier:** 84 maps were solved by the first answer it checked. It refused 4,086
    answers in all.
- **The solved maps have shorter routes.** Their shortest routes average 5.5 moves,
  against 8.2 on the unsolved maps.

**What E was trained on.**

| Kind of found route | Routes |
| --- | --- |
| The same as the label | 212 |
| A different route of the same length | 81 |
| Longer than the label | 33 |

- **How much longer:** 1 move for 12 routes, 2 for 15, 3 for 2, 4 for 3 and 5 for 1.
- **Wasted moves:** 19 of the longer routes move into a wall, which leaves the player in
  place. 3 revisit a cell.
- **Repeats:** each solved map appears 6 or 7 times among the 2,000 extra examples.

## Primary: expert iteration helps

E against L (no fine-tune) on the 240 held-out maps, greedy decoding, exact McNemar.

| | Maps | E | L | Only E | Only L | p |
| --- | --- | --- | --- | --- | --- | --- |
| **All held-out** | 240 | **88** | **42** | 50 | 4 | 3.8 × 10⁻¹¹ |
| Generated (`detour-heldout`) | 200 | 73 | 34 | 42 | 3 | 8.7 × 10⁻¹⁰ |
| Test detour maps | 40 | 15 | 8 | 8 | 1 | 0.039 |
| 3 x 3 | 11 | 8 | 5 | 3 | 0 | 0.25 |
| 4 x 4 | 36 | 21 | 11 | 10 | 0 | 0.0020 |
| 5 x 5 | 73 | 32 | 18 | 17 | 3 | 0.0026 |
| 6 x 6 | 120 | 27 | 8 | 20 | 1 | 2.1 × 10⁻⁵ |

- **The gain holds on the 40 released test detour maps alone,** which the generator did
  not make.
- **E gains at every size from 4 x 4 up.** The largest gain is on 6 x 6, where the search
  solved 59% of the pool maps.

## Secondary: no detectable difference between found routes and labels

E against M on the same 240 maps, with the same test.

| | Maps | E | M | Only E | Only M | p |
| --- | --- | --- | --- | --- | --- | --- |
| **All held-out** | 240 | **88** | **97** | 16 | 25 | 0.21 |
| Generated (`detour-heldout`) | 200 | 73 | 82 | 13 | 22 | 0.18 |
| Test detour maps | 40 | 15 | 15 | 3 | 3 | 1.0 |
| 3 x 3 | 11 | 8 | 9 | 0 | 1 | 1.0 |
| 4 x 4 | 36 | 21 | 23 | 0 | 2 | 0.50 |
| 5 x 5 | 73 | 32 | 36 | 6 | 10 | 0.45 |
| 6 x 6 | 120 | 27 | 29 | 10 | 12 | 0.83 |

- **The point estimate favours the labels by 9 maps,** but the test cannot settle that.
- **This comparison had little power.** 212 of the 326 found routes are the labels, so E
  and M shared about two thirds of their extra examples. Two identical LIVR runs disagree
  on 31 of 400 test maps.
- **The two arms tie on the 40 test detour maps,** 15 each.

**The outcome-table row that applies:** "E helps, with no detectable difference from
M". The spec's next step for it is "a second round from E's weights, or a second
problem."

## Descriptive

**Against V500 (all 500 pool maps, 4 times each, with labels; detour scaling job
6941470).**

| | Held-out | Against V500: only the arm, only V500 | p |
| --- | --- | --- | --- |
| E | 88 | 19, 28 | 0.24 |
| M | 97 | 21, 21 | 1.0 |
| V500 | 97 | — | — |

- **Dropping the 174 maps the search failed cost nothing detectable:** M ties V500.
- **For scale:** V2000, with 2,000 distinct maps and their labels, solved 107.

**Seen maps (the first 98 pool maps).** The search solved 62 of them, so E and M trained
on those 62. Neither trained on the other 36. V500 trained on all 98.

| | Solved: the 62 trained maps | Solved: the 36 other maps |
| --- | --- | --- |
| L | 22 | 0 |
| E | 60 | 0 |
| M | 58 | 2 |
| V500 | 50 | 12 |

The maps the search failed can be learned from labels: V500 solved 12 of the 36.
Expert iteration gets no direct signal from them in this round.

**E did not learn the wasted moves.**
- **Held-out:** 86 of E's 88 solutions are shortest routes, and one moves into a wall. L
  has 40 of 42 shortest.
- **Seen maps:** 58 of E's 60 solutions are shortest routes, and two move into a wall.

**How the arms fail on held-out maps** (failures.json):

| | L | E | M | V500 |
| --- | --- | --- | --- | --- |
| Goal by a shortest route | 40 | 86 | 97 | 95 |
| Goal by a longer route | 2 | 2 | 0 | 2 |
| Into a hole | 192 | 147 | 138 | 139 |
| Stops short of the goal | 5 | 4 | 4 | 3 |
| Passes the goal and moves on | 1 | 1 | 1 | 1 |

Holes still cause most failures. E walks into a hole on 61% of the held-out maps.

**E's benchmark against LIVR** (one sampled answer per test map; M was not
benchmarked):

| | Maps | E | LIVR | Only E | Only LIVR | p |
| --- | --- | --- | --- | --- | --- | --- |
| All | 400 | 320 | 328 | 15 | 23 | 0.26 |
| Detour maps | 40 | 15 | 9 | 6 | 0 | 0.031 |
| No-detour maps | 360 | 305 | 319 | 9 | 23 | 0.020 |

- **The harm the spec expected appeared.** E loses a net 14 no-detour maps. The
  detour-scaling arms lost 9–16 with the same mix.
- **How E's no-detour answers change:** 19 fewer shortest routes, 5 more longer routes,
  10 more into a hole, and 5 more that stop short.
- **Overall,** 320 against 328 is not detectable. Two identical LIVR runs disagree on 31
  of 400 test maps.

## Limits

- **One round, one training run per arm.** McNemar treats the map as the unit, not the
  run.
- **Selection.** E trains only on maps the search solved, which skew small and short.
- **The detour-heavy mix** is the detour scaling check's: 2,000 of 3,000 examples.
- **The pool covers 3 x 3 to 6 x 6.**
- **Greedy decoding** for the held-out readings; the benchmark samples.

## Files

`e/` is E (comparison.json's `search`). `m/` is M (`shortest`). Paths under the PACE
results root are written as `$SPARA_REMOTE_ROOT/`.

| File | Contents |
| --- | --- |
| comparison.json | `expert_iteration.compare` with E, M, L and V500 (the last two from `../detour-scaling-6941468/`). Validity, primary, secondary, descriptive comparisons, in-job seconds (`seconds_in_jobs`) and Slurm GPU-seconds (`slurm_gpu_seconds`) |
| failures.json | The outcome breakdown above, for held-out greedy and the benchmark's 360 no-detour maps |
| routes/summary.json | The search's checks (including the failed batched values), the 97 detour maps and their reproduction of the JEPA run, and the pool search by size, stops, calls, refusals, routes and cost |
| routes/summary-before-pool.json | The same, written before the pool search |
| routes/routes.jsonl | One row per pool map: solved, stop, policy and verifier calls, refused answers, route, answer, label, equal to the label, longer than the label |
| routes/detour-searches.jsonl | One row per detour map from the search with one request per pass |
| routes/manifest.json | Run manifest: config, settings, package versions, source hashes, data and fresh-set manifests |
| `<arm>/summary.json` | Set checks, training summary, greedy counts by population, run time, peak memory |
| `<arm>/greedy.jsonl` | One row per greedy evaluation: population, split, row, grid size, route, success |
| `<arm>/manifest.json` | Run manifest, with the label source, the routes file and the recipe |
| `<arm>/steps.jsonl` | Loss, gradient norm and learning rate at each optimizer step |
| e/benchmark-outcomes.jsonl | Per test map: row, grid size, success and extracted answer (prompts and raw text dropped) |
| e/benchmark-summary.json | Benchmark accuracy, overall and by size |
