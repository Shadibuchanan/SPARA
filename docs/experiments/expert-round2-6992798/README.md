# Evidence for expert iteration round 2, jobs 6992798–6992800

Round 2 of expert iteration, on 7 x 7 and 8 x 8 maps, specified in the
[round 2 spec](../../EXPERT_ITERATION_ROUND2_SPEC.md). Two questions:
- **Primary:** does LIVR's Stage 1 bottleneck in the first training phase (EB2) add to
  expert iteration alone (E2)?
- **Secondary:** does a second round help on larger maps (E2 against E)?

- **Starting point:** E, round 1's fine-tune
  (`$SPARA_REMOTE_ROOT/results/expert-train-search-6961875/stage2/final`). It is the search
  policy and the start of both arms. L is the LIVR rerun (`baseline-6253300`).
- **Jobs:** one H200 each, 2026-11-12. All three completed on the first submission. The
  search waited about 2 h 50 min in the queue; queue time is not GPU time.

| Job | Contents | Wall time | In the job | Peak memory |
| --- | --- | --- | --- | --- |
| 6992798, search | E's evaluation, the checks, the frontier (searched twice), the new pool, L's greedy | 1 h 22 min 34 s | Frontier with one request per pass 729 s (9,741 reads); pool 2,102 s (23,170 reads); L's greedy 197 s | 20.1 GiB |
| 6992799, E2 | Two unmasked phases (187 + 188 steps), evaluation, benchmark | 30 min 7 s | Phase 1 405 s; phase 2 414 s | 55.3 GiB |
| 6992800, EB2 | The same, with the bottleneck mask in phase 1 | 26 min 2 s | Phase 1 387 s; phase 2 408 s | 55.3 GiB |

- **GPU-seconds:** 8,323 (139 GPU-minutes) by Slurm's elapsed time.
  - **Against the estimates:** about 90–95 with TF32 reads, and about 125 if the job fell
    back to round 1's reads. It fell back. The search took 83 minutes against about 70.
  - **Where the extra went:** about 15 minutes on TF32 reads that were then discarded.
    - **The design:** the code acts on the read check only after searching the 174
      frontier maps, so the job searched them twice. The check failed before that
      search.
    - **No faster anyway:** the TF32 pass took about 15 minutes, checks included,
      against 12 minutes for the one-at-a-time pass. This is from file times.
  - **The detour-learning line so far:** 18,937 GPU-seconds (5 h 16 min). Round 2's CPU
    precondition jobs, 6989902 and 6989964, used no GPU.
- **Source:** the hashes in each manifest match the local code for every file the runs
  use. The only differences are files the runs do not import: the protected PACE files
  and the package's `SOURCES.txt`.
- **Weights:** E2's and EB2's final weights (15 GB each) are on PACE scratch at
  `$SPARA_REMOTE_ROOT/results/round2-e2-6992799/phase2/final` and
  `$SPARA_REMOTE_ROOT/results/round2-eb2-6992800/phase2/final`. Scratch is at 266 of
  300 GB.
- **Latent states** (`states.pt`, about 44 MB per run) are not in this folder. They are on
  PACE scratch beside each run, and in the git-ignored local `results/`.

## Validity: all checks passed, after the planned fallback

| Check | Threshold | Result |
| --- | --- | --- |
| TF32 batched reads against single reads, 52 pool maps at step 0 | within 0.01 | **Failed:** 0.0107 |
| TF32 batched reads, 32 cached reads after the frontier | within 0.01 | 0.00054 |
| Fallback: one request per pass under bfloat16, the 52 maps | within 0.01 | 0.0 |
| Loading: E's greedy outcomes on round 1's 240 held-out maps match round 1 | at least 228 | 240 of 240 |
| Every accepted route passes the reference evaluator | all | 248 of 248 (201 new, 47 frontier); rechecked on CPU |
| Labels of the two new sets pass the reference evaluator | 100% | 900 of 900 |
| The new sets share no layout with each other, the other fresh sets, train or test | 0 | 0 in all 13 pairs |

- **The fallback ran as the spec planned:** one request per pass, the 52-map check
  passed, and the frontier was searched again.
- **TF32 missed by 0.0007** at step 0. The CPU precondition had measured bfloat16 at up
  to 0.030.

## The search

**The frontier:** E's search solved 47 of the 174 maps L's search failed in round 1.

| Map size | Solved |
| --- | --- |
| 4 x 4 | 5 of 13 |
| 5 x 5 | 17 of 56 |
| 6 x 6 | 25 of 105 |

**The new pool:** 201 of 500 maps solved.

| Map size | Solved |
| --- | --- |
| 7 x 7 | 107 of 231 |
| 8 x 8 | 94 of 269 |

- **Stops:** 201 solved and 299 at the budget.
  - **Policy calls:** median 65, and 15 on solved maps.
  - **Verifier:** 66 maps were solved by the first answer it checked. It refused 5,382
    answers in all.
- **The solved maps have shorter routes.** Their shortest routes average 6.4 moves,
  against 9.9 on the unsolved maps.
- **The found routes:** 108 are the label, 76 are another route of the same length, and
  17 are longer.
  - **How much longer:** 1 move for 7 routes, 2 for 8, 4 for 1 and 5 for 1.
  - **Wall bumps:** 9 routes move into a wall, which leaves the player in place.
- **The variety floor** of 100 solved new maps held.

**What E2 and EB2 trained on.** 3,000 rows, the same for both arms:
- the 1,000 released maps;
- 1,000 rows from the 201 solved new maps;
- 1,000 rows from 373 solved smaller maps: round 1's 326 and the 47 frontier maps.

## Manipulation check: the bottleneck did not take

The probe reads hole or not at each cell from the four latents at layer 12. It is fitted
on each size's pool maps and scored on that size's held-out maps (balanced accuracy).

| | E | E2 | EB2 |
| --- | --- | --- | --- |
| Hole probe, layer 12 | 0.649 | 0.653 | 0.657 |
| 7 x 7 | 0.657 | 0.669 | 0.665 |
| 8 x 8 | 0.641 | 0.638 | 0.649 |

- **EB2 − E2:** 0.004, 95% interval −0.004 to 0.011 (paired bootstrap, 2,000 resamples).
  The margin was 0.05.
- **Fine-tuning barely moved the probe:** E, E2 and EB2 are within 0.01 of each other.

## Primary: no reading on perception

EB2 against E2 on the 400 new held-out maps, greedy decoding, exact McNemar. The check
failed, so this is descriptive.

| | Maps | EB2 | E2 | Only EB2 | Only E2 | p |
| --- | --- | --- | --- | --- | --- | --- |
| **All held-out** | 400 | **78** | **84** | 20 | 26 | 0.46 |
| 7 x 7 | 198 | 45 | 51 | 11 | 17 | 0.34 |
| 8 x 8 | 202 | 33 | 33 | 9 | 9 | 1.0 |

## Secondary: round 2 helps on larger maps

E2 against E on the same 400 maps, with the same test.

| | Maps | E2 | E | Only E2 | Only E | p |
| --- | --- | --- | --- | --- | --- | --- |
| **All held-out** | 400 | **84** | **54** | 39 | 9 | 1.5 × 10⁻⁵ |
| 7 x 7 | 198 | 51 | 35 | 22 | 6 | 0.0037 |
| 8 x 8 | 202 | 33 | 19 | 17 | 3 | 0.0026 |

- **E already generalised from round 1,** which trained only on 3 x 3 to 6 x 6: 54
  against L's 17 on these maps (40 only E, 3 only L, p = 3.0 × 10⁻⁹).

**The outcome-table rows that apply:**
- **"Did not take."** The mask did not change what the latents hold at these sizes. The
  spec's next step is a label-free alternative, predicting the image from the latents,
  which is out of its scope.
- **The secondary reading:** round 2 helps on larger maps.

## Descriptive

**Masked greedy** (prompt and answer cannot attend to the image; only the latents can).

| | Unmasked | Masked | Gap |
| --- | --- | --- | --- |
| E | 54 | 48 | 6 |
| E2 | 84 | 66 | 18 |
| EB2 | 78 | 73 | 5 |

- **E2's gap is not small,** so the spec's "little to force in" reading does not apply.
  E2's unmasked fine-tune widened the gap from E's 6 to 18. E2 solves about 18 maps only
  when the answer can attend to the image directly.
- **EB2 kept the gap at 5, yet the hole probe did not move.** EB2 practised answering
  under the mask, so the smaller gap need not mean its latents hold more of the map, and
  the probe says they do not.
- **Masked, EB2 against E2:** 73 against 66 (26 only EB2, 19 only E2, p = 0.37).

**The probe at other layers, and goal and start cells.**

| | E | E2 | EB2 |
| --- | --- | --- | --- |
| Hole probe, layer 20 | 0.627 | 0.632 | 0.635 |
| Hole probe, layer 28 | 0.533 | 0.536 | 0.537 |
| Goal cell, layer 12 (7 x 7, 8 x 8) | 0.78, 0.68 | 0.81, 0.70 | 0.79, 0.69 |
| Start cell, layer 12 (7 x 7, 8 x 8) | 0.77, 0.62 | 0.79, 0.56 | 0.75, 0.59 |

**Harm checks on round 1's 240 held-out maps (3 x 3 to 6 x 6).** Neither arm lost ground
on smaller maps; both gained.

| Arm | The arm solves | E solves | Only the arm | Only E | p |
| --- | --- | --- | --- | --- | --- |
| E2 | 102 | 88 | 21 | 7 | 0.013 |
| EB2 | 100 | 88 | 19 | 7 | 0.029 |

**Benchmark** (one sampled answer per test map).

| | All 400 | p | Detour maps (40) | p | No-detour maps (360) | p |
| --- | --- | --- | --- | --- | --- | --- |
| LIVR | 328 | — | 9 | — | 319 | — |
| E (round 1) | 320 | — | 15 | — | 305 | — |
| E2 | 322 | 0.48 against LIVR, 0.88 against E | 19 | 0.0020 against LIVR | 303 | 0.017 against LIVR |
| EB2 | 325 | 0.73 against LIVR, 0.50 against E | 17 | 0.0078 against LIVR | 308 | 0.043 against LIVR |

- **The same trade as round 1:** gains on detour maps, losses on no-detour maps, and no
  detectable change overall.
- **Against E,** neither arm differs detectably on any of the three.

**How the arms fail on the 400 new held-out maps** (comparison.json, `failures`):

| | L | E | E2 | EB2 |
| --- | --- | --- | --- | --- |
| Goal by a shortest route | 16 | 52 | 82 | 75 |
| Goal by a longer route | 1 | 2 | 2 | 3 |
| Into a hole | 367 | 322 | 295 | 305 |
| Stops short of the goal | 16 | 22 | 21 | 17 |
| Passes the goal and moves on | 0 | 2 | 0 | 0 |

Holes still cause most failures: E2 walks into a hole on 74% of these maps.

**Seen maps (the first 98 pool maps).** The search solved 45 of them, so both arms
trained on those 45 and not on the other 53.

| | Solved: the 45 trained maps | Solved: the 53 other maps |
| --- | --- | --- |
| E | 14 | 0 |
| E2 | 34 | 0 |
| EB2 | 35 | 0 |

As in round 1, the arms solve none of the maps the search failed. Expert iteration gets
no signal from them.

**Training.** Answer loss per phase.

| | Phase 1 first step | Phase 1 last quarter | Phase 2 first step | Phase 2 last quarter |
| --- | --- | --- | --- | --- |
| E2 | 0.082 | 0.061 | 0.107 | 0.060 |
| EB2 (phase 1 masked) | 0.100 | 0.071 | 0.079 | 0.056 |

The masked phase started higher, as the spec expected, and stayed higher. EB2 ended
phase 2 lower than E2.

## Limits

- **One training run per arm.** McNemar treats the map as the unit, not the run. Two
  identical LIVR runs disagree on 31 of 400 test maps.
- **The probe is weak at every layer:** 0.65 at best, where 0.5 is chance. A bottleneck
  effect smaller than its noise would not show.
- **187 masked steps.** E's history was LIVR's Stage 1 (masked), LIVR's Stage 2
  (unmasked), then round 1 (unmasked). Round 2 added 187 masked steps after those two
  unmasked phases. Whether a longer masked phase would move the probe is not tested.
- **Selection:** E2 and EB2 train only on maps the search solved, which have shorter
  routes.
- **Greedy decoding** for the held-out readings; the benchmark samples.

## Files

`search/` is the search job, `e2/` and `eb2/` the arms. Paths under the PACE results
root are written as `$SPARA_REMOTE_ROOT/`.

| File | Contents |
| --- | --- |
| comparison.json | `expert_round2.compare` with E, E2, EB2 and L. Validity, manipulation, primary, secondary, descriptive, in-job seconds (`seconds_in_jobs`) and Slurm GPU-seconds (`slurm_gpu_seconds`) |
| search/summary.json | The checks (including the failed TF32 values), E's evaluation, the frontier and pool search by size, stops, calls, refusals, routes and cost |
| search/summary-before-pool.json | The same, written before the pool search |
| search/routes.jsonl | One row per new pool map: solved, stop, policy and verifier calls, refused answers, route, answer, label, equal to the label, longer than the label |
| search/frontier.jsonl | The same for the 174 frontier maps, from the search with one request per pass |
| search/greedy.jsonl, search/masked.jsonl | E's greedy answers, unmasked on 738 maps and masked on the 400 new held-out maps |
| search/livr-greedy.jsonl, search/livr-summary.json | L's greedy answers on the 400 new held-out maps |
| `<run>/manifest.json` | Run manifest: config, settings, package versions, source hashes, data and fresh-set manifests; for the arms, the training counts |
| `<arm>/summary.json` | Set checks, both phases' training summaries, greedy counts, run time, peak memory |
| `<arm>/greedy.jsonl`, `<arm>/masked.jsonl` | One row per greedy answer: population, split, row, grid size, route, success |
| `<arm>/phase1/`, `<arm>/phase2/` | Each phase's summary and its loss, gradient norm and learning rate at each optimizer step |
| `<arm>/benchmark-outcomes.jsonl` | Per test map: row, grid size, success and extracted answer (prompts and raw text dropped) |
| `<arm>/benchmark-summary.json` | Benchmark accuracy, overall and by size |
