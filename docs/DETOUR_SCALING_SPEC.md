# Detour scaling check: does layout variety help detour learning generalize?

Status, 2026-11-10: run (jobs 6941468–6941471). All validity checks passed. Primary:
**variety helps**. Secondary: **detour fine-tuning helps**. No arm improves the overall
benchmark, because each loses no-detour maps. See [Result](#result-jobs-69414686941471).
The reading rules were fixed before the run, and the user approved running it.

Scaling at 7 x 7 and 8 x 8, 2026-11-19: n8000 (8,000 more maps) solves 162 of the 400
held-out maps against EB3's 83 (p < 0.001), so more distinct maps help
([result](#result-n8000-job-7167924)). p8000 (the same maps, with the plan written out
before the route) solves 395 of the 400 against n8000's 162 (p < 0.001)
([result](#result-p8000-job-7167925)), and 396 of 400 fresh maps that no experiment had
read ([confirmation](#result-confirmation-job-7170755)). n24000 (24,000 more maps, route
alone) solves 273 ([result](#result-n24000-job-7168959)). The plan's lines are targets
computed from the map layout, beyond the route answer, so p8000 is reported as a separate
reference and p24000 was cancelled ([scope](#scope-after-p8000)). h12000 (the route alone
on 12,000 maps with deep detours, from n24000) solves 275, no detectable difference from
n24000: it gains 36 of the 112 maps that need two or more moves away and loses 34 of the
288 that need one ([result](#result-h12000-job-7172913)). e8000 (n8000's maps with routes
the model's own search finds, no labels) solves 141 against EB3's 83 (p < 0.001) and
n8000's 162 (p = 0.03) ([result](#result-e8000-job-7207178)). Round 2 of it, e2, solves
160 against e8000's 141 (p = 0.008), level with n8000 ([result](#result-e2-job-7213832)).
Round 3, e3, is specified; not yet read. See
[Scaling at 7 x 7 and 8 x 8](#scaling-at-7-x-7-and-8-x-8-d1).

## Question

LIVR often gets the first forced detour right on its training maps (the top move is
correct on 78 of 98) but rarely on unseen maps (33 of 85 such points). It solves 17–22%
of unseen detour maps. One explanation:
- 902 of the 1,000 training maps reward heading for the goal;
- the 98 detour maps are memorised as exceptions rather than learned as a rule.

If that is right, more **distinct** detour layouts should help where repeated exposure
to the same few does not.

**Why the answer gates the plan.**
- On FrozenLake, guided search mostly finds shortest routes (90 of 97 with the
  toward-goal stand-in). So expert iteration there is close to supervised training on
  more maps with near-oracle labels.
- The JEPA loop also passes its gains to the policy through supervised training.
- If more varied detour data does not generalize, neither will reach unseen maps on
  FrozenLake ([detour-learning spec](DETOUR_LEARNING_SPEC.md),
  [guided-search spec](GUIDED_SEARCH_SPEC.md)).

## Maps

Two new sets, drawn by `spara.fresh_maps` in detour-only mode:
- **Size:** 3 x 3 to 6 x 6, drawn on each attempt with the released training split's
  weights (10, 20, 30 and 40%).
- **Which layouts are kept:** solvable detour maps only. This yields the released
  detour maps' mix of sizes.
- **Labels:** the shortest route that comes first in U, L, R, D order. Every released
  answer is a shortest route, and this rule reproduces 976 of the 1,000 released
  training answers.
- **The sets:**
  - `fresh/detour-heldout`: 200 maps, seed 201, generated first;
  - `fresh/detour-pool`: 2,000 maps, seed 202.
- **Disjoint by construction:** each set excludes the released layouts and every
  earlier fresh set (`probe`, then `detour-heldout`).

## Arms

All start from the LIVR rerun's final Stage 2 weights (`baseline-6253300`).

| Arm | Training | Extra detour examples | Distinct extra layouts | Repeats per layout |
| --- | --- | --- | --- | --- |
| L | none | — | — | — |
| V98 | one epoch of fine-tuning | 2,000 | 98 | about 20 |
| V500 | one epoch of fine-tuning | 2,000 | 500 | 4 |
| V2000 | one epoch of fine-tuning | 2,000 | 2,000 | 1 |

- **Fine-tuning data:** the released 1,000 training maps plus 2,000 examples from the
  first *n* maps of `detour-pool`, so the sets are nested.
- **Recipe:** 3,000 examples and 375 optimizer steps. Otherwise LIVR's Stage 2 recipe:
  learning rate 1e-5, cosine schedule with 5% warmup, accumulation 8, seed 42.
- **Only variety differs:** every V arm sees the same number of detour examples, in the
  same mix with released maps.
- **Shared maps:** the first 98 pool maps are in every V arm's data.

## Evaluation

**Greedy decoding** (deterministic; the decoding used in earlier fresh-map comparisons),
on three populations:
- **held-out:** the 200 `detour-heldout` maps and the 40 test detour maps (240);
- **seen:** the first 98 `detour-pool` maps;
- for L, the same three.

**Reference benchmark** for the V arms: the reported protocol, with one sampled answer
per test map. L's comes from the recorded LIVR evaluation, so it is not rerun.

## Readings (fixed before the run)

**Primary: V2000 against V98 on the 240 held-out maps.** Paired by map, exact McNemar,
two-sided.
- **variety helps:** V2000 solves more, p < 0.05;
- **variety hurts:** V2000 solves fewer, p < 0.05;
- **no detectable effect of variety:** otherwise.

**Secondary: V98 against L on the same 240 maps**, with the same test. The readings are
*detour fine-tuning helps*, *harms* or *no detectable difference*. This comparison
cannot separate 98 new layouts from detour-heavy training as such.

**Descriptive (no verdicts):**
- V500, as a trend;
- each comparison on the generated and test populations separately, and by size;
- each arm's success on the seen maps. The V2000 arm saw them once; the V98 arm saw
  them about 20 times;
- the reference benchmark: overall, on the 40 detour maps, and on the 360 no-detour
  maps, against LIVR (a harm check);
- the last epoch's answer loss.

**Validity (the readings are withheld if any check fails):**
1. **Labels:** every pool and held-out label passes the reference evaluator (100%).
2. **Disjoint:** the held-out set and the pool share no layout with each other, with
   `fresh/probe`, or with the released splits (100%). The released splits share 7
   layouts between themselves. None is a test detour map, so the 40 test detour maps are
   unseen in training.
3. **Reproduction:** L's greedy routes on the 40 test detour maps match the routes
   recorded by [planning probe 6887458](experiments/planning-probe-6887458/README.md)
   on at least 38.
4. **Training took:** V98 solves at least 90% of the 98 seen maps. Otherwise the
   fine-tune did not change the model enough for a null result to mean anything.

**How the outcomes would steer the plan:**

| Outcome | Interpretation | Next |
| --- | --- | --- |
| Variety helps | Memorised exceptions, cured by variety | Generated maps are the lever on FrozenLake. Expert iteration and JEPA matter where no oracle labels exist. |
| No effect of variety; V98 beats L | Detour exposure helps, variety does not | The gain is a shift toward detouring. Check the no-detour harm before relying on it. |
| No effect of variety; V98 does not beat L | The limit is the model or its training, not the data | Supervised routes will not reach unseen detours on FrozenLake. Rethink before building expert iteration or JEPA. |

## Result (jobs 6941468–6941471)

Evidence: [docs/experiments/detour-scaling-6941468/](experiments/detour-scaling-6941468/README.md).
4,210 GPU-seconds (about 70 GPU-minutes) in four jobs, against an estimate of about 80.

**Validity: all four checks passed.**
- labels 2,200 of 2,200;
- no shared layouts;
- L reproduced 40 of 40 recorded test routes;
- V98 solved 98 of 98 seen maps.

**Primary: variety helps.** On the 240 held-out maps:
- V2000 solves 107 and V98 solves 64 (51 maps only V2000, 8 only V98, p = 9 × 10⁻⁹);
- the gain holds on the released test detour maps alone (20 against 11, p = 0.035), so
  it is not only a match to the generator.

**Secondary: detour fine-tuning helps** (V98 64, L 42, p = 0.0016). The significance
comes from the generated maps; on the 40 test detour maps alone it is 11 against 8,
p = 0.45.

**Descriptive.**

| | L | V98 | V500 | V2000 |
| --- | --- | --- | --- | --- |
| Held-out greedy, 240 | 42 | 64 | 97 | 107 |
| Seen maps, 98 | 22 | 98 | 62 | 50 |
| Benchmark, all 400 | 328 | 316 | 328 | 319 |
| Benchmark, 40 detour maps | 9 | 10 | 18 | 16 |
| Benchmark, 360 no-detour maps | 319 | 306 | 310 | 303 |

- **Saturation.** Most of the gain comes by 500 layouts. V2000 against V500 is not
  distinguishable (p = 0.11).
- **Memorisation on the seen maps.**
  - V98 solves all 98, after about 20 passes each, but only 27% of held-out maps.
  - V2000 saw them once and solves 51%, near its 45% on held-out maps.
- **Holes still dominate.** Into a hole is the most common failure in every arm: 130
  of 240 held-out maps for V2000.
- **No-detour harm.** Every V arm loses no-detour benchmark maps (9–16).
  - **Clear for V98 and V2000, weaker for V500.** Computed after the run, one sampled
    pass of the LIVR rerun is expected to solve 318 ± 2.2 of them, and the two LIVR runs
    scored 316 and 319.
    - V98 (306) and V2000 (303) are more than 5 standard deviations below that.
    - V500 (310) is 3.6 below, and its paired test is not significant (p = 0.12).
  - **For V2000**, the losses are 7 more falls into a hole, 9 more stops short of the
    goal and 5 more longer routes. The last two are unneeded detours.
  - **The detour gains are cancelled.** On the benchmark, V500 and V2000 roughly double
    LIVR on detour maps (18 and 16 against 9), but no arm improves the overall score.

**Where this lands in the outcome table.** It is the first row: the detour
exceptions were memorised, and variety cures part of that. Generated detour maps are
the lever on FrozenLake.
- **The second row's caution applies too.** The harm check fails. A fine-tune with
  70% detour examples, starting from a converged model, shifts the policy toward
  detouring on maps that do not need it.
- **Detour success is still low.** V2000 still fails 55% of held-out detour maps,
  mostly by walking into a hole.

**Open, not run (needs separate approval):**
- whether a mix that keeps no-detour maps at their released weight keeps the detour
  gain without the harm;
- whether training Stage 2 from Stage 1 with generated maps does better than
  fine-tuning a converged model;
- training-run variance. Each arm is one run.

## Limits

- **One training run per arm.** McNemar treats the map as the unit, not the run, so
  run-to-run variation is not measured. The two LIVR runs differed by 2 of 40 test
  detour maps.
- **The fine-tunes start from a converged model.** A null result does not rule out
  training Stage 2 from Stage 1 with many generated maps.
- **The fine-tuning mix is 70% detour examples.** All V arms share this shift; L does
  not.
- **The primary uses greedy decoding;** the benchmark samples.

## Scope

This diagnostic trains on generated maps, with the user's approval for this run. The
question in issue #8, and any ADR, stays open for reported conditions. The benchmark
and its decoding are unchanged.

## Budget

| Item | Estimate | Basis |
| --- | --- | --- |
| Training, per V arm | about 12.5 minutes | 3,000 examples at about 0.25 s each (the step-grounding run) |
| Loading and saving, per V arm | about 3 minutes | |
| Greedy evaluation, per arm | about 4 minutes | 338 maps |
| Benchmark, per V arm | about 5 minutes | |

- **Total:** about 25 GPU-minutes per V arm and 6 for L, about 80 GPU-minutes (1.35
  GPU-hours) in all.
- **Jobs:** four, one H200 each. Wall limits are 45 minutes for V arms and 20 for L.
  GPU-seconds are counted across all jobs and any retries.
- **Storage:** final weights only, estimated at about 7.5 GB per V arm. The actual
  size was 15 GB, because the weights are saved in fp32. About 91 GB of the scratch
  quota was free. The weights were deleted after the run, at the user's request.

## Running

```bash
python -m spara.fresh_maps --data-root data/vsp --name detour-heldout --detour-count 200 \
  --sizes 3,4,5,6 --weights 1,2,3,4 --seed 201 --exclude probe
python -m spara.fresh_maps --data-root data/vsp --name detour-pool --detour-count 2000 \
  --sizes 3,4,5,6 --weights 1,2,3,4 --seed 202 --exclude probe,detour-heldout
python -m spara.cli detour-scaling --arm n98 --recipe configs/detour_scaling.json \
  --config configs/livr.json --data-root data/vsp --checkpoint /path/to/stage2/final --output /path/to/run
python scripts/analyze_detour_scaling.py --runs livr=RUN_L n98=RUN_98 n500=RUN_500 n2000=RUN_2000
```

On PACE, [scripts/pace/detour_scaling.sbatch](../scripts/pace/detour_scaling.sbatch) with
`SPARA_LIVR_CHECKPOINT` and `SPARA_ARM`.

## Scaling at 7 x 7 and 8 x 8: D1

EB3, the best model on the 400 maps of `detour-heldout-78` (83 solved), has trained on 472
distinct 7 x 7 and 8 x 8 detour maps in all, and solves 34 of the 98 it has seen. The
oracle check found that with every hole reading right the policy still fails 234 of the
400 maps ([result](HOLE_CHECK_SPEC.md#result-jobs-7154267-and-7155886)). The check above
found that distinct layouts are what generalize at 3 x 3 to 6 x 6.

**Question: do many more distinct maps at these sizes raise success on unseen ones?**

### Design

- **Maps:** `fresh/detour-pool-78c`, 8,000 labelled detour maps drawn as the held-out set
  was (7 x 7 and 8 x 8 with equal weight, seed 207), excluding every earlier set.
- **Arm n8000:** one epoch on the 1,000 released training maps plus the 8,000 new maps,
  each once: 9,000 examples, 1,125 optimizer steps, this spec's recipe otherwise.
- **Start:** EB3's final weights (job 7088136).
- **Labels, not found routes.** Expert iteration found no detectable difference between
  the two, twice ([expert iteration](EXPERT_ITERATION_SPEC.md)), and search costs GPU time
  that labels do not.
- **Evaluation:** greedy on the 400 held-out maps, the 40 test detour maps and the first
  98 pool maps; then the reference benchmark.
- **The weights are saved,** so a larger run can continue from them.

### Readings (fixed before the run)

**Primary: n8000 against EB3 on the 400 held-out maps,** paired by map, exact McNemar,
two-sided, at 0.05. EB3's side is its recorded greedy pass.

| Outcome | Reading | Next |
| --- | --- | --- |
| n8000 solves more, p < 0.05 | More distinct maps help at these sizes | Continue from n8000's weights on the 24,000 maps of `detour-pool-78d`. |
| Otherwise | Variety at this recipe is not the limit | Do not scale the data. Change the form of the answer so the plan is written out. |

**Validity (the reading is withheld if any check fails):**
1. **The sets:** every label passes the reference evaluator, and the pool shares no
   layout with the held-out set, the other fresh sets or the released splits.
2. **The run is whole:** 1,125 optimizer steps, and an answer for each of the 538 maps.
3. **The cap:** at most 2% of held-out answers reach 128 new tokens.

**Descriptive (no verdicts):** by size; the 98 seen maps (EB3 34); the 40 test detour
maps (EB3 19); the reference benchmark against EB3's 330 of 400, as a harm check; the
loss.

**Budget:** one job on one H200, about 55 minutes: 9,000 examples at EB3's 0.27 seconds
each, and about 13 minutes of loading and evaluation. The limit is 80 minutes.

**Limits:**
- **One run, one seed.**
- **EB3 is not rerun.** Its side of the comparison is the greedy pass its own job
  recorded, by the same evaluation code.
- **Many experiments have read this held-out set.** A final model needs a fresh one.

**Implementation:** the recipe `configs/detour_scaling_78.json`. Its `base_steps` lets
the scaling check start from a later fine-tune's weights. On PACE,
`scripts/pace/detour_scaling.sbatch` with `SPARA_RECIPE`, `SPARA_ARM=n8000` and
`SPARA_LIVR_CHECKPOINT` set to EB3's weights.

### Result: n8000 (job 7167924)

**Validity: all three checks pass.** The set checks pass, the run took 1,125 optimizer
steps and answered all 538 maps, and none of the 400 held-out answers reached the cap.

**Primary: n8000 solves 162 of the 400 held-out maps, EB3 83.** 91 maps only n8000
solves, 12 only EB3, p < 0.001 (3.5e-16). **More distinct maps help at these sizes.**
Per the table, the next run continues from n8000's weights on `detour-pool-78d`.

**Descriptive:**

| | n8000 | EB3 |
| --- | --- | --- |
| Held-out 7 x 7 (198) | 91 | 51 |
| Held-out 8 x 8 (202) | 71 | 32 |
| Seen maps (98; each arm's own) | 36 | 34 |
| Test detour maps (40) | 15 | 19 |
| Reference benchmark (400, sampled) | 345 | 330 |

- **Benchmark:** 31 maps only n8000 solves, 16 only EB3 (p = 0.040, not a fixed reading).
- **Loss:** 0.178 over the run, 0.128 over the last quarter.
- **Time:** 2,650 seconds of training (0.29 seconds an example); the job took 55 minutes.

**What limits the number.** n8000 solves 36 of the 98 pool maps it trained on once (37%)
and 40% of maps it never saw. It is not memorising maps; one pass a map leaves it
equally short on both. More maps are more of the same useful signal.

**Post hoc, from the recorded routes** (categories from the analysis of EB3 above):

| Moves away from the goal the shortest route needs | Maps | n8000 | EB3 |
| --- | --- | --- | --- |
| One | 288 | 154 | 81 |
| Two | 64 | 8 | 2 |
| Three or more | 48 | 0 | 0 |

- **By shortest route length:** 4 to 7 moves 121 of 173, 8 to 11 moves 41 of 175, 12 or
  more 0 of 52.
- **Failures:** all 238 step into a hole; 7 on the first move (EB3: 50). 267 of the 400
  routes have the shortest route's length (EB3: 146).

**Limits:** one run, one seed. The test detour maps fell from 19 to 15 of 40; that is
not tested. The gain is on maps with one detour; maps needing three or more moves away
stay at none.

### More maps: n24000

**Question: does the gain continue with three times as many new maps?**

- **Maps:** `fresh/detour-pool-78d`, 24,000 labelled detour maps drawn as the held-out set
  was (seed 208), excluding every earlier set.
- **Arm n24000:** one epoch on the 1,000 released training maps plus the 24,000 new maps,
  each once: 25,000 examples, 3,125 optimizer steps, n8000's recipe otherwise.
- **Start:** n8000's final weights (job 7167924).
- **Evaluation:** as n8000, with the first 98 maps of this pool as the seen maps. The
  weights are saved.

**Readings (fixed before the run).**

**Primary: n24000 against n8000 on the 400 held-out maps,** paired by map, exact McNemar,
two-sided, at 0.05. n8000's side is its recorded greedy pass.

| Outcome | Reading | Next |
| --- | --- | --- |
| n24000 solves more, p < 0.05 | The route alone still gains from more maps | Scale the form that leads once p8000 is read: the plan arm if p8000 beat n8000, else a larger pool from n24000's weights. |
| Otherwise | The route alone has stopped gaining from maps at this recipe | Do not scale it further. The plan arm's table decides the next run. |

**Second reading: the 48 held-out maps that need three or more moves away from the
goal.** EB3 and n8000 solve none. If n24000 also solves none, more maps of the same mix
do not reach deep detours, and the next change has to target them whatever the primary
says.

**Validity (the reading is withheld if any check fails):** n8000's three checks, with
3,125 optimizer steps.

**Descriptive (no verdicts):** by size; by moves away needed; the seen maps; the test
detour maps (n8000 15, EB3 19); the reference benchmark against n8000's 345; the loss.

**Budget:** one job on one H200, about 135 minutes: 25,000 examples at n8000's 0.29
seconds, and about 12 minutes of loading and evaluation. The limit is 180 minutes.

**Limits:** one run, one seed. The learning rate schedule restarts, so n24000 is a second
fine-tune and not one longer run. The held-out set has now been read by many experiments;
`fresh/detour-heldout-78b` (400 maps, seed 209) is kept unread for the final model.

**Implementation:** the recipe `configs/detour_scaling_78d.json`, run as n8000 with
`SPARA_ARM=n24000` and `SPARA_LIVR_CHECKPOINT` set to n8000's weights.

### Result: n24000 (job 7168959)

**Validity: all three checks pass.** The set checks pass, the run took 3,125 optimizer
steps and answered all 538 maps, and none of the 400 held-out answers reached the cap.

**Primary: n24000 solves 273 of the 400 held-out maps, n8000 162.** 116 maps only n24000
solves, 5 only n8000, p < 0.001. **The route alone still gains from more maps.** p8000
beat n8000, so per the table the plan arm is the one scaled next (p24000).

**Second reading: n24000 solves 7 of the 48 maps that need three or more moves away from
the goal.** More maps of the same mix do reach some deep detours: 7 of the 24 that need
three moves away, none of the 24 that need four or more.

**Descriptive:**

| | n24000 | n8000 | EB3 | p8000 |
| --- | --- | --- | --- | --- |
| Held-out 7 x 7 (198) | 144 | 91 | 51 | 197 |
| Held-out 8 x 8 (202) | 129 | 71 | 32 | 198 |
| Moves away needed: one (288) | 233 | 154 | 81 | 286 |
| Moves away needed: two (64) | 33 | 8 | 2 | 61 |
| Moves away needed: three or more (48) | 7 | 0 | 0 | 48 |
| Seen maps (98; each arm's own) | 68 | 36 | 34 | 97 |
| Test detour maps (40) | 22 | 15 | 19 | 40 |
| Reference benchmark (400, sampled) | 369 | 345 | 330 | 399 |

- **Against p8000:** 3 maps only n24000 solves, 125 only p8000 (p < 0.001). With four
  times the new maps, the route alone is still well short of the plan.
- **By shortest route length:** 4 to 7 moves 158 of 173, 8 to 11 moves 111 of 175, 12 or
  more 4 of 52.
- **Failures:** all 127 step into a hole, none on the first move.
- **Loss:** 0.121 over the run, 0.068 over the last quarter.
- **Time:** 7,932 seconds of training (0.32 seconds an example); the job took 143 minutes.

**Limits:** one run, one seed. n24000 is a second fine-tune after n8000, so its 273 is the
score at 32,000 new maps in two passes of the schedule, not at 24,000.

### The plan arm: p8000

EB3 solves 81 of the 288 held-out maps that need one move away from the goal, 2 of the 64
that need two, and none of the 48 that need more; 292 of its 317 failures step into a
hole, 50 of them on the first move (post hoc, from its recorded routes). A route written
in one pass has to hold the whole map and the whole search in the model's states.

**Question: at the same data, does writing the plan out before the route solve more
maps than writing the route alone?**

**The plan** is a fixed function of the map layout, written before the boxed route:

```
size 7
holes: 2,1 2,6 5,0 5,4 6,5
start 6,4 goal 0,6
1: 0,5 1,6
2: 0,4 1,5
...
10: 4,0 5,1 6,2 6,4
path: 6,4 L 6,3 U 5,3 ... R 0,6
\boxed{LEFT, UP, UP, UP, UP, UP, UP, RIGHT, RIGHT, RIGHT}
```

- **Lines `1:`, `2:`, ...** list the safe cells at each distance from the goal, until the
  line that holds the start. Each cell follows from the line before and the hole list.
- **`path:`** walks from the start, each move the first in U, L, R, D order that lowers
  the distance. It is the label's route.
- **No helper image and no new input.** The prompt and the image are unchanged; only the
  training answers differ.

**Arm p8000** is n8000 with every training answer written as the plan: the same 9,000
examples in the same order, from EB3, on the same schedule. Plans run to a median of 288
tokens and at most 535 (measured on 900 maps with the model's tokenizer), so greedy
answers are capped at 640.

**Readings (fixed before the run).**

**Primary: p8000 against n8000 on the 400 held-out maps,** paired by map, exact McNemar,
two-sided, at 0.05. If n8000 has no valid run, p8000 is read against EB3.

| Outcome | Reading | Next |
| --- | --- | --- |
| p8000 solves more, p < 0.05 | Writing the plan out helps | Scale the plan arm: continue from p8000's weights on `detour-pool-78d`. |
| n8000 solves more, p < 0.05 | The plan harms at this data | Scale the direct arm if it beat EB3. Report where plans first go wrong. |
| Otherwise | No detectable difference | Decide from the sections: fix the first section that goes wrong most. |

**Validity (the reading is withheld if any check fails):** n8000's checks 1 and 2, and at
most 2% of held-out answers reach 640 new tokens.

**Descriptive (no verdicts):** the first section of each held-out plan that is not the
map's (size, holes, ends, levels, path, route, or none); the share of holes listed and
the number invented; success among plans that are right throughout; p8000 against EB3;
by size; the seen maps; the test detour maps; the reference benchmark, which samples at
temperature 0.7 and so tests long plans harder than greedy does.

**Budget:** one job on one H200, about 110 minutes: 9,000 examples at an estimated 0.45
seconds, about 30 minutes of greedy answers and the benchmark, and loading. The limit is
150 minutes.

**Limits:** one run, one seed. The plan's length grows with the map, so it does not carry
to much larger maps as written. The plan is text the model writes, so it gives no reading
on LIVR's latents.

**Implementation:** `spara/plan_trace.py` (`with_plan` writes the answer, `read_plan`
scores one); the recipe `configs/detour_scaling_78_plan.json` (`"answer": "plan"`), run as
n8000 with `SPARA_ARM=p8000`.

### Result: p8000 (job 7167925)

**Validity: the checks pass.** The set checks pass, the run took 1,125 optimizer steps and
answered all 538 maps, and none of the 400 held-out answers reached the cap (the longest
is 536 tokens).

**Primary: p8000 solves 395 of the 400 held-out maps, n8000 162.** 233 maps only p8000
solves, none only n8000, p < 0.001. **Writing the plan out helps.** Per the table, the
next run continues from p8000's weights on `detour-pool-78d`.

**Descriptive:**

| | p8000 | n8000 | EB3 |
| --- | --- | --- | --- |
| Held-out 7 x 7 (198) | 197 | 91 | 51 |
| Held-out 8 x 8 (202) | 198 | 71 | 32 |
| Moves away needed: one (288) | 286 | 154 | 81 |
| Moves away needed: two (64) | 61 | 8 | 2 |
| Moves away needed: three or more (48) | 48 | 0 | 0 |
| Seen maps (98; each arm's own) | 97 | 36 | 34 |
| Test detour maps (40) | 40 | 15 | 19 |
| Reference benchmark (400, sampled) | 399 | 345 | 330 |

- **The plans.** 372 of the 400 held-out plans are the map's own throughout, and all 372
  are solved. Of the 28 with a wrong section, 23 are still solved. The first wrong section
  is the distance lines in 20, the path in 5, the holes in 2 and the route in 1.
- **The holes.** The plans list 4,768 of the maps' 4,770 holes and invent 2.
- **The five failures** all list every hole. The first wrong section is the path in two,
  the distance lines in two and the route in one. Four step into a hole; one stops short.
- **Loss:** 0.080 over the run, 0.0009 over the last quarter.
- **Time:** 2,636 seconds of training, as n8000. The job took 126 minutes; 78 of them are
  evaluation, since a plan is about 283 tokens against about 25 for a route alone.

**What the number measures** (checked after the run, before trusting it):
- **The prompt holds no layout.** A held-out prompt is the benchmark's fixed text and the
  image. The hole list for an unseen map can only come from the image.
- **The score is the route's.** Code that shares nothing with the evaluator walks each
  boxed route on the layout and agrees on all 538 answers.
- **The maps are unseen.** The held-out set shares no layout with any training set (the
  set check), and the 400 answers are 400 different texts.

**Limits:**
- **One run, one seed.**
- **The answer is about eleven times longer,** and its length grows with the map.
- **The supervision is richer.** The plan's lines come from breadth-first search on the
  layout, the same source as the labels. No helper image is used.
- **The plan is written text.** No latent token appears in any answer, so this gives no
  reading on LIVR's latents.
- **Many experiments have read this held-out set,** though no choice in this arm was made
  on it. The confirmation below reads a fresh one.

### The plan with more maps: p24000

**Question: do more maps remove the remaining plan errors?**

- **Arm p24000:** n24000's data and schedule (25,000 examples, 3,125 optimizer steps) with
  every answer written as the plan, from p8000's weights (job 7167925). Evaluated as
  p8000. The weights are saved.

**Readings (fixed before the run).** p8000 leaves five held-out maps unsolved, so the
solved count cannot show a gain at 0.05 (five maps gained and none lost gives p = 0.06).
The primary is the count with more room.

**Primary: plans that are the map's own throughout, p24000 against p8000 (372 of 400),**
paired by map, exact McNemar, two-sided, at 0.05.

| Outcome | Reading | Next |
| --- | --- | --- |
| p24000 has more, p < 0.05 | More maps make the plans more exact | p24000 is the final model if it also solves at least 395. |
| p8000 has more, p < 0.05 | More training harms the plans | p8000 stays the final model. |
| Otherwise | No detectable difference | The final model is the one that solves more of the 400; p8000 on a tie. |

**The final model** is then read once on `fresh/detour-heldout-78b`.

**Validity (the reading is withheld if any check fails):** p8000's checks, with 3,125
optimizer steps.

**Descriptive (no verdicts):** solved of 400 and the paired table against p8000; the
first wrong section; the holes; by size; the seen maps; the test detour maps; the
reference benchmark.

**Budget:** one job on one H200, about 205 minutes: 25,000 examples at 0.29 seconds,
about 80 minutes of evaluation, and loading. The limit is 260 minutes.

**Limits:** one run, one seed. The schedule restarts, as for n24000.

**Implementation:** the recipe `configs/detour_scaling_78d_plan.json`, with
`SPARA_ARM=p24000` and `SPARA_LIVR_CHECKPOINT` set to p8000's weights.

**Cancelled.** Job 7170756 was stopped after 44 minutes, at about step 800 of 3,125, when
the scope below was set. It has no reading.

### Confirmation on a fresh held-out set

`fresh/detour-heldout-78b` is 400 detour maps drawn as `detour-heldout-78` was (seed
209), excluding every other set. No experiment has read it.

**Question: does p8000's score hold on maps no experiment has read?**

- **Run:** p8000's saved weights, no training, greedy answers on the 400 fresh maps. The
  job also repeats the 40 test detour maps and p8000's 98 seen maps, where the same
  weights should give the same answers (40 and 97).

**Reading (fixed before the run):** solved of 400 against 395 of 400 on
`detour-heldout-78`, Fisher's exact test, two-sided, at 0.05.

| Outcome | Reading |
| --- | --- |
| No detectable difference | The score holds on fresh maps. |
| Fewer on the fresh set, p < 0.05 | The first held-out set flattered the model; the fresh count is the one to report. |

**Validity:** the set checks pass for the fresh set, and all 538 maps are answered with at
most 2% of the fresh answers at the cap.

**Budget:** one job on one H200, about 50 minutes; the limit is 80.

**Implementation:** the recipe `configs/detour_scaling_78b_plan.json`, whose arm `confirm`
trains on no maps, with `SPARA_LIVR_CHECKPOINT` set to p8000's weights.

### Result: confirmation (job 7170755)

**Validity: the checks pass.** The fresh set's labels are all accepted and it shares no
layout with any other set; all 538 maps are answered; none of the 400 fresh answers
reached the cap.

**p8000 solves 396 of the 400 fresh maps,** against 395 of 400 on `detour-heldout-78`
(Fisher's exact p = 1.00). **The score holds on fresh maps.**

**Descriptive:**
- **By size:** 190 of 191 at 7 x 7, 206 of 209 at 8 x 8.
- **The repeats:** all 138 repeated maps get the same answer as in job 7167925, token for
  token (40 of 40 test detour maps, 97 of 98 seen maps).
- **The plans:** 376 of 400 are the map's own throughout, and all 376 are solved. The
  first wrong section is the distance lines in 19, the path in 4 and the size in 1. The
  plans list 4,715 of the maps' 4,719 holes and invent 1.
- **The four failures:** three list every hole and go wrong in the distance lines or the
  path. One reads an 8 x 8 map as 7 x 7 and lists 10 of its 14 holes.
- **The score is the route's:** the independent walk agrees with the evaluator on all 538.
- **Time:** 59 minutes, all of it evaluation.

**Limits:** one model from one run. The fresh set is drawn the same way as the first, so
this says nothing about other sizes or hole densities.

### Scope after p8000

The project trains on route answers and simulator rewards and excludes helper images
([project plan](SPARA_PROJECT_PLAN.md)); LIVR learns from final-answer supervision.
- **n8000 and n24000 are in scope:** their only target is the route.
- **p8000 is a separate reference.** It uses no helper image, but its hole list, distance
  lines and path are targets computed from the map layout, as step grounding's labels
  are. It is reported apart from the route-only arms, and it is not scaled further.
- **Deriving the plan from search in the step simulator would not change this.** An
  exhaustive search in the exact simulator recovers the layout, so the text and the
  supervision would be the same.

The best arm in scope is n24000, with 273 of the 400.

### Deep detours: h12000

n24000 solves 233 of the 288 held-out maps that need one move away from the goal and 40
of the 112 that need two or more. Maps that need two or more are 28% of the pools it
trained on.

**Question: does training the route-only arm on maps with deep detours raise its
score?**

- **Maps:** `fresh/detour-pool-78e`, 12,000 labelled detour maps that each need at least
  two moves away from the goal (6,790 need two, 2,878 three, 2,332 four or more; 5,359 at
  7 x 7 and 6,641 at 8 x 8; seed 210), excluding every earlier set.
- **Arm h12000:** one epoch on the 1,000 released training maps plus the 12,000 new maps,
  each once: 13,000 examples, 1,625 optimizer steps, n8000's recipe otherwise. The only
  target is the route.
- **Start:** n24000's final weights (job 7168959).
- **Evaluation:** as n8000, with the first 98 maps of this pool as the seen maps. The
  weights are saved.

**Readings (fixed before the run).**

**Primary: h12000 against n24000 on the 400 held-out maps,** paired by map, exact McNemar,
two-sided, at 0.05. n24000's side is its recorded greedy pass.

| Outcome | Reading | Next |
| --- | --- | --- |
| h12000 solves more, p < 0.05 | Deep-detour maps raise the route-only score | Continue from h12000 on more maps of this kind. |
| n24000 solves more, p < 0.05 | Training on deep detours alone harms | Return to the natural mix from n24000's weights. |
| Otherwise | No detectable difference | Decide from the two halves below. |

**The two halves** (same test, each at 0.05): the 112 held-out maps that need two or more
moves away (n24000: 40), where the gain should be; and the 288 that need one (n24000:
233), where training on none of that kind could cost. A gain on the first with a loss on
the second means the next pool should mix both kinds.

**Validity (the reading is withheld if any check fails):** n8000's three checks, with
1,625 optimizer steps.

**Descriptive (no verdicts):** by size; by moves away needed; the seen maps; the test
detour maps (n24000: 22); the reference benchmark against n24000's 369; the loss.

**Budget:** one job on one H200, about 80 minutes: 13,000 examples at n24000's 0.32
seconds, and about 10 minutes of loading and evaluation. The limit is 110 minutes.

**Limits:** one run, one seed. There is no arm with 12,000 maps of the natural mix, so a
gain cannot be split between more maps and harder maps. The held-out set has been read by
every arm here; the fresh set `detour-heldout-78b` has been read once, by p8000.

**Implementation:** `spara.fresh_maps --min-away 2` draws the pool; the recipe
`configs/detour_scaling_78e.json`, with `SPARA_ARM=h12000` and `SPARA_LIVR_CHECKPOINT` set
to n24000's weights.

### Result: h12000 (job 7172913)

**Validity: all three checks pass.** The set checks pass, the run took 1,625 optimizer
steps and answered all 538 maps, and none of the 400 held-out answers reached the cap.

**Primary: h12000 solves 275 of the 400 held-out maps, n24000 273.** 58 maps only h12000
solves, 56 only n24000, p = 0.93. No detectable difference, so the two halves decide.

**The two halves: a gain on deep detours and a loss on the rest.**

| Held-out maps | h12000 | n24000 | Only h12000 | Only n24000 | p |
| --- | --- | --- | --- | --- | --- |
| Need two or more moves away (112) | 76 | 40 | 37 | 1 | < 0.001 |
| Need one move away (288) | 199 | 233 | 21 | 55 | < 0.001 |

Per the table, **the next pool should mix both kinds.** Training on deep detours alone
moves the model toward them and away from the common case; 114 of the 400 maps change
sides.

**Descriptive:**

| | h12000 | n24000 | n8000 | EB3 |
| --- | --- | --- | --- | --- |
| Held-out 7 x 7 (198) | 150 | 144 | 91 | 51 |
| Held-out 8 x 8 (202) | 125 | 129 | 71 | 32 |
| Moves away needed: one (288) | 199 | 233 | 154 | 81 |
| Moves away needed: two (64) | 52 | 33 | 8 | 2 |
| Moves away needed: three or more (48) | 24 | 7 | 0 | 0 |
| Seen maps (98; each arm's own) | 64 | 68 | 36 | 34 |
| Test detour maps (40) | 31 | 22 | 15 | 19 |
| Reference benchmark (400, sampled) | 375 | 369 | 345 | 330 |

- **Deep detours:** 17 of the 24 maps that need three moves away and 7 of the 24 that
  need four or more (n24000: 7 and 0).
- **Seen maps:** h12000's seen maps all need two or more moves away. It solves 64 of the
  98, against 76 of the 112 held-out maps of that kind, so there is no gap between seen
  and unseen.
- **By shortest route length:** 4 to 7 moves 137 of 173, 8 to 11 moves 120 of 175, 12 or
  more 18 of 52.
- **Failures:** 117 step into a hole, 7 pass the goal, 1 stops short.
- **Loss:** 0.104 over the run, 0.058 over the last quarter.
- **Time:** 3,809 seconds of training (0.29 seconds an example); the job took 75 minutes
  on an H200.

**Limits:** one run, one seed. There is no arm with 12,000 more maps of the natural mix,
so the gain on deep detours cannot be split between more maps and harder maps. The loss
on one-away maps could come from the mix or from a third pass of the schedule.

**Not run next.** A labelled arm on a pool that mixes both kinds is what the table names.
It is held, because the goal since 2026-11-19 is no labels on the new maps (e8000 below);
the finding carries over as a rule for the label-free rounds, whose pools should hold
both kinds.

### No labels: e8000

n8000 and n24000 train on a label for every new map: the shortest route, computed from the
layout. Expert iteration needs none. The policy proposes whole routes, most likely first;
the verifier says only whether each reaches the goal; the policy then trains on the routes
that passed. EB3 was built this way on 472 maps at these sizes
([round 3](EXPERT_ITERATION_ROUND3_SPEC.md)), and labels and found routes gave no
detectable difference in rounds 1 and 2.

**Question: with no labels on the new maps, does a round of expert iteration on n8000's
8,000 maps raise the score?**

- **Search:** final-verdict-only search with EB3 as the policy over the 8,000 maps of
  `detour-pool-78c`: at most 64 policy calls a map, routes of at most 64 moves, one
  sequence per pass, as in rounds 2 and 3. No step simulator, no judge and no label enters
  the search. The pool is searched in eight parts of 1,000 maps, one job each.
- **Arm e8000:** one epoch from EB3 on the 1,000 released training maps plus each map the
  search solved, once, with the found route as the answer. n8000's recipe otherwise.
- **Evaluation:** as n8000. The weights are saved.

**Readings (fixed before the run).**

**Primary: e8000 against EB3 on the 400 held-out maps,** paired by map, exact McNemar,
two-sided, at 0.05.

| Outcome | Reading | Next |
| --- | --- | --- |
| e8000 solves more, p < 0.05 | Expert iteration at this scale raises the score without labels | Round 2: e8000 searches the maps it left unsolved and new maps, then trains on what it finds. |
| Otherwise | No detectable gain | Read what the search found (how many, how deep) before another round. |

**Second: e8000 against n8000,** the same test. n8000 has a label for all 8,000 maps and
e8000 a route for the ones its search solved, so this is the cost of having no labels at
one round.

**Validity (the reading is withheld if any check fails):**
1. **The search:** the eight parts cover each of the 8,000 maps once, and every route kept
   passes the reference evaluator (the search stops if one does not).
2. **The sets, the whole run and the cap,** as n8000, with one optimizer step for every
   eight training examples.

**Descriptive (no verdicts):** maps the search solved, by size and by moves away from the
goal needed; found routes equal to the label, another shortest route, or longer; policy
calls; by size; the seen maps; the test detour maps; the reference benchmark; the loss.

**Budget:** eight search jobs of about 70 minutes on an H200 (4 seconds a map in round 3),
longer on a smaller GPU; the limit is 200 minutes each. The search peaks near 18 GiB, so
any GPU of 40 GB or more is requested. Then one training job of about 40 minutes on an
80 GB GPU; the limit is 90 minutes.

**Limits:** one round, one seed. The search solves the easier maps first, so e8000 trains
on fewer and easier maps than n8000. The search jobs may land on different GPU types, whose
arithmetic can differ in the last digits and so reorder near-ties; every kept route is
verified either way. The verifier is exact, as in every round before.

**Implementation:** `spara.expert_iteration.search_pool` (the `--search START:STOP` stage of
`detour-scaling`) and `--routes` to train on what it found; on PACE, `SPARA_SEARCH` and
`SPARA_ROUTES` with `SPARA_ARM=e8000` and the recipe `configs/detour_scaling_78.json`.

### Result: e8000 (job 7207178)

**Validity: all checks pass.**
1. **The search** (jobs 7172965 to 7172972): the eight parts cover maps 0 to 7,999 once
   each. All 4,039 kept routes pass the reference evaluator, in the jobs and in a local
   re-check.
2. **The sets, the whole run and the cap:** the set checks pass, the run took 630
   optimizer steps (5,039 examples) and answered all 538 maps, and none of the 400
   held-out answers reached the cap.

**Primary: e8000 solves 141 of the 400 held-out maps, EB3 83.** 61 maps only e8000
solves, 3 only EB3, p < 0.001. **Expert iteration at this scale raises the score without
labels.** Per the table, round 2 is next.

**Second: n8000 solves 162, e8000 141.** 51 maps only n8000 solves, 30 only e8000,
p = 0.03. A label for every one of the 8,000 maps is worth 21 held-out maps more than one
round of search on them.

**What the search found.** 4,039 of the 8,000 maps, in 329,860 policy calls (median 13
when solved) and 10.6 GPU-hours.

| Moves away needed | Maps | Solved by the search |
| --- | --- | --- |
| One | 5,566 | 3,629 |
| Two | 1,379 | 353 |
| Three | 588 | 54 |
| Four or more | 467 | 3 |

- **By size:** 2,061 of 3,679 at 7 x 7 and 1,978 of 4,321 at 8 x 8.
- **Routes:** 2,182 equal the label, 1,584 are another shortest route, 273 are longer.

**Descriptive:**

| | e8000 | EB3 | n8000 |
| --- | --- | --- | --- |
| Held-out 7 x 7 (198) | 78 | 51 | 91 |
| Held-out 8 x 8 (202) | 63 | 32 | 71 |
| Moves away needed: one (288) | 135 | 81 | 154 |
| Moves away needed: two (64) | 6 | 2 | 8 |
| Moves away needed: three or more (48) | 0 | 0 | 0 |
| Seen maps (98; each arm's own) | 29 | 34 | 36 |
| Test detour maps (40) | 20 | 19 | 15 |
| Reference benchmark (400, sampled) | 334 | 330 | 345 |

- **Seen maps:** the first 98 maps of the pool. The search solved 44 of them, so e8000
  trained on those 44; it solves 28 of the 44 and 1 of the other 54.
- **By shortest route length:** 4 to 7 moves 115 of 173, 8 to 11 moves 26 of 175, 12 or
  more 0 of 52.
- **Failures:** 251 step into a hole, 8 stop short.
- **Loss:** 0.109 over the run, 0.077 over the last quarter.
- **Time:** 1,489 seconds of training (0.30 seconds an example); the job took 35 minutes.

**Limits:** one round, one seed. e8000 trained on half as many new maps as n8000, and 90%
of them need one move away, so the second reading mixes the cost of no labels with the
cost of fewer and easier maps.

### No labels, round 2: e2

e8000's search solved 26% of the maps that need two moves away and 9% of those that need
three, so e8000 saw few deep detours and solves 6 of the 112 held-out ones. h12000 showed
that the route-only model learns deep detours when it trains on them. Round 2 gives the
better policy more of those maps to search.

**Question: does a second round, with e8000 as the policy, raise the score over e8000?**

- **Pool:** `fresh/detour-pool-78f`, 11,961 maps taken from existing sets; no map is new.
  - **The frontier:** the 3,961 maps of `detour-pool-78c` that round 1's search left
    unsolved (1,937 need one move away, 1,026 two, 998 three or more).
  - **The natural mix:** the first 4,000 maps of `detour-pool-78d` (2,852 need one move
    away, 1,148 two or more).
  - **Deep detours:** the first 4,000 maps of `detour-pool-78e` (all need two or more).
  The e-chain has not trained on any of these maps. Their labels are in the files and do
  not enter the search or the training.
- **Search:** as round 1, with e8000 as the policy: at most 64 policy calls a map, routes
  of at most 64 moves, one sequence per pass, in twelve parts.
- **Arm e2:** one epoch from e8000 (job 7207178) on the 1,000 released training maps plus
  each map round 2's search solves, once, with the found route. Round 1's solved maps are
  not repeated, as n24000 did not repeat n8000's maps.
- **Evaluation:** as n8000, with the first 98 maps of this pool as the seen maps. The
  weights are saved.

**Readings (fixed before the run).**

**Primary: e2 against e8000 on the 400 held-out maps,** paired by map, exact McNemar,
two-sided, at 0.05.

| Outcome | Reading | Next |
| --- | --- | --- |
| e2 solves more, p < 0.05 | Round 2 pays | Round 3 the same way: e2 searches its frontier and maps not yet searched. |
| No detectable difference | Returns have slowed | Read the search by depth. The next lever is the search (a larger budget on the frontier) before more rounds. |
| e8000 solves more, p < 0.05 | Round 2 harms | Check the loss and the mix of found routes before anything else. |

**Second: the 112 held-out maps that need two or more moves away,** e2 against e8000 (6),
the same test. This is where round 2's pool aims.

**Third: e2 against n8000 (162),** the same test: two rounds without labels against one
pass with 8,000 labels.

**Validity (the reading is withheld if any check fails):**
1. **The search:** the twelve parts cover each of the 11,961 maps once, and every kept
   route passes the reference evaluator.
2. **The sets, the whole run and the cap,** as n8000, with one optimizer step for every
   eight training examples. The set check compares the pool with the held-out sets, the
   released splits and the earlier sets, and not with the three sets it is taken from.

**Descriptive (no verdicts):** maps the search solved, by source and by moves away
needed; found routes equal to the label, another shortest route, or longer; policy calls;
e2 by size and by moves away; the seen maps; the test detour maps; the benchmark; the
loss.

**Budget:** twelve search jobs of 1,000 maps (the last 961). Round 1 took 59 to 148
minutes a part; this pool has more maps the search will not solve, each of which costs
all 64 calls, so the limit is 300 minutes. Any GPU of 40 GB or more. Then one training job
on an 80 GB GPU, with a limit of 120 minutes.

**Limits:** one seed. The pool is 60% deep detours by choice, not the natural mix, so a
gain cannot be split between a second round and a harder pool. The frontier maps were
chosen by round 1's outcome. The held-out set has been read by every arm here.

**Implementation:** the recipe `configs/detour_scaling_78f.json` with `SPARA_ARM=e2`,
`SPARA_SEARCH` and `SPARA_ROUTES` as in round 1, and `SPARA_LIVR_CHECKPOINT` set to
e8000's weights.

### Result: e2 (job 7213832)

**Validity: all checks pass.**
1. **The search** (jobs 7208595 to 7208606): the twelve parts cover maps 0 to 11,960 once
   each. All 4,268 kept routes pass the reference evaluator, in the jobs and in a local
   re-check.
2. **The sets, the whole run and the cap:** the set checks pass, the run took 659
   optimizer steps (5,268 examples) and answered all 538 maps, and none of the 400
   held-out answers reached the cap.

**Primary: e2 solves 160 of the 400 held-out maps, e8000 141.** 33 maps only e2 solves,
14 only e8000, p = 0.008. **Round 2 pays.** Per the table, round 3 runs the same way.

**Second: on the 112 held-out maps that need two or more moves away, e2 solves 9 and
e8000 6.** 3 maps only e2 solves, none only e8000, p = 0.25. No detectable gain on deep
detours. The gain is on the 288 maps that need one move away: 151 against 135.

**Third: e2 solves 160, n8000 162.** 35 maps only e2 solves, 37 only n8000, p = 0.91. Two
rounds without labels are level with one pass with 8,000 labels.

**What the search found.** 4,268 of the 11,961 maps, in 587,623 policy calls (median 16
when solved) and 21.1 GPU-hours.

| Moves away needed | Frontier | Natural mix | Deep detours | All |
| --- | --- | --- | --- | --- |
| One | 672 of 1,937 | 2,214 of 2,852 | | 2,886 of 4,789 |
| Two | 148 of 1,026 | 214 of 630 | 817 of 2,278 | 1,179 of 3,934 |
| Three | 17 of 534 | 37 of 306 | 135 of 939 | 189 of 1,779 |
| Four or more | 1 of 464 | 3 of 212 | 10 of 783 | 14 of 1,459 |
| All | 838 of 3,961 | 2,468 of 4,000 | 962 of 4,000 | 4,268 of 11,961 |

- **Against round 1's search:** on the natural mix e8000's search solves 78% of the maps
  that need one move away, 34% of those that need two and 12% of those that need three.
  EB3's search in round 1 solved 65%, 26% and 9%. The maps differ and are drawn the same
  way.
- **Routes:** 2,231 equal the label, 1,790 are another shortest route, 247 are longer.
- **Deep routes:** 1,382 of the 4,268 found routes need two or more moves away, against
  410 of round 1's 4,039.

**Descriptive:**

| | e2 | e8000 | EB3 | n8000 |
| --- | --- | --- | --- | --- |
| Held-out 7 x 7 (198) | 92 | 78 | 51 | 91 |
| Held-out 8 x 8 (202) | 68 | 63 | 32 | 71 |
| Moves away needed: one (288) | 151 | 135 | 81 | 154 |
| Moves away needed: two (64) | 9 | 6 | 2 | 8 |
| Moves away needed: three or more (48) | 0 | 0 | 0 | 0 |
| Seen maps (98; each arm's own) | 12 | 29 | 34 | 36 |
| Test detour maps (40) | 19 | 20 | 19 | 15 |
| Reference benchmark (400, sampled) | 343 | 334 | 330 | 345 |

- **Seen maps:** the first 98 maps of the pool, all from the frontier. The search solved
  21 of them; e2 solves 10 of those 21 and 2 of the other 77.
- **By shortest route length:** 4 to 7 moves 127 of 173, 8 to 11 moves 33 of 175, 12 or
  more 0 of 52.
- **Failures:** 236 of the 240 step into a hole.
- **Loss:** 0.116 over the run, 0.086 over the last quarter.
- **Time:** 2,170 seconds of training; the job took 52 minutes.

**Post hoc (not a fixed reading).**
- **Found routes are worth about as much as labels, map for map.** The chain has trained
  on 8,307 found routes over two rounds and solves 160; n8000 trained on 8,000 labels and
  solves 162. Of the found routes 1,792 need two or more moves away; of n8000's labels
  about 2,400 do. What the chain lacks against n24000 (273, on 32,000 labels) is the
  number of solved maps, and above all of deep ones.
- **The search budget is not exhausted.** 1,111 of round 2's 4,268 solves needed more
  than 32 of the 64 policy calls, and 84% of the calls went to maps left unsolved.

**Limits:** one seed. Two things changed from round 1 (a second round, and a pool with
more deep detours), so the gain cannot be split between them.

### No labels, round 3: e3

**Question: does a third round, with e2 as the policy and a pool about three times
larger, raise the score over e2?**

- **Pool:** `fresh/detour-pool-78g`, 31,693 maps taken from existing sets; no map is new.
  - **The frontier:** the 7,693 maps of round 2's pool that its search left unsolved
    (1,903 need one move away, 2,755 two, 3,035 three or more).
  - **The natural mix:** maps 4,000 to 23,999 of `detour-pool-78d`, 20,000 maps (14,004
    need one move away, 5,996 two or more).
  - **Deep detours:** maps 4,000 to 7,999 of `detour-pool-78e`.
  The e-chain has not trained on any of these maps. Their labels are in the files and do
  not enter the search or the training.
- **Search:** as rounds 1 and 2, with e2 as the policy: at most 64 policy calls a map,
  routes of at most 64 moves, one sequence per pass, in 32 parts.
- **Arm e3:** one epoch from e2 (job 7213832) on the 1,000 released training maps plus
  each map round 3's search solves, once, with the found route.
- **Evaluation:** as n8000, with the first 98 maps of this pool as the seen maps. The
  weights are saved.

**Readings (fixed before the run).**

**Primary: e3 against e2 on the 400 held-out maps,** paired by map, exact McNemar,
two-sided, at 0.05.

| Outcome | Reading | Next |
| --- | --- | --- |
| e3 solves more, p < 0.05 | Round 3 pays | Round 4 the same way, on newly drawn maps and the frontier. |
| No detectable difference | Returns have slowed | The next lever is the search: a larger budget on the frontier. |
| e2 solves more, p < 0.05 | Round 3 harms | Check the loss and the mix of found routes before anything else. |

**Second: the 112 held-out maps that need two or more moves away,** e3 against e2 (9),
the same test.

**Third: e3 against n24000 (273),** the same test. n24000 trained on 32,000 labels over
two passes; this asks how far the chain is from it once it has found a similar number of
routes. It is read with the count of routes the chain has trained on.

**Validity (the reading is withheld if any check fails):** as round 2, with 32 parts
covering each of the 31,693 maps once.

**Descriptive (no verdicts):** as round 2.

**Budget:** 32 search jobs of 1,000 maps (the last 693), about 47 GPU-hours at round 2's
0.13 seconds a policy call; the limit is 300 minutes each, on any GPU of 40 GB or more.
Then one training job on an 80 GB GPU, with a limit of 200 minutes.

**Limits:** one seed. The pool is larger and of a different mix than round 2's, so a gain
is the gain of this round as run, not of a round alone. The held-out set has been read by
every arm here.

**Implementation:** the recipe `configs/detour_scaling_78g.json` with `SPARA_ARM=e3`, and
`SPARA_LIVR_CHECKPOINT` set to e2's weights.

### Does e2 use its latents?

The four latent tokens sit in the prompt and read the image in one pass; nothing feeds
them back. If e2's planning does not run through them, the gains of the e-chain say
nothing about latent reasoning, and the project's question needs latents that compute in
steps. The [latent battery](LATENT_BATTERY_RESULTS.md) answers this for LIVR on the
benchmark: swapping in another map's latents made 351 of 400 answers follow the donor.

**Question: on the 400 held-out detour maps, does e2's answer depend on its latents?**

- **Run:** the latent battery on e2's weights (job 7213832) over `fresh/detour-heldout-78`:
  greedy answers with its own latents, with another map's latents of the same size
  (swap), with the mean latent of the 400 maps of that size (the training maps have no
  7 x 7 or 8 x 8 grid), and with the latent tokens removed; each with and without the
  Stage 1 mask. Evaluation only.

**Readings (fixed before the run), unmasked mode:**

| Outcome | Reading | Next |
| --- | --- | --- |
| Removed solves fewer than own, p < 0.05 (exact McNemar) | e2's answers use the latents | Read the swap row: whether the latents carry the map or the plan. |
| Otherwise | No detectable use: e2 plans without its latents | The e-chain's gains are text-side; latent reasoning needs a different build (latents that compute in steps). |

**Second:** of the swapped answers, how many give the donor map's own route. Many means the
route is set by what the latents carry; few means the answer reads the image directly.

**Validity:** the battery's identity check passes (own states patched back give the same
answers), and all 400 maps are answered in every condition.

**Descriptive:** the mean condition; the masked mode; each condition by moves away needed.

**Limits:** one checkpoint. A latent the answer ignores at test time may still have shaped
training. The mean is taken over the held-out maps themselves (each map's own share is
about 1 in 200).

### Written reasoning without labels: the probe

The deep-research pass (2026-11-23) found no verified method in which a small
vision-language model plans in latents without helper images or step labels. Every working
latent method trains on reasoning text that a script or a stronger model wrote, and
pretrained models fine-tuned that way mostly ignore their latents. The best-supported route
for Qwen2.5-VL-3B on FrozenLake is written reasoning under outcome reward (VAGEN: 0.09
without reasoning, 0.68 with it). The label-free form is STaR: the model writes its own
reasoning and route, the verifier keeps the answers that pass, and the model trains on
them. It can start only if the model's own reasoning sometimes ends in a passing route.

**Question: does the released Qwen2.5-VL-3B-Instruct, asked to reason first, reach
passing routes often enough to start label-free training on its own reasoning?**

- **Maps:** `fresh/detour-probe-78`, 100 new detour maps (43 at 7 x 7, 57 at 8 x 8; 74
  need one move away, 16 two, 10 three or more; seed 211), sharing no layout with any
  earlier set. For the curriculum reading, the first 25 maps of each size (3 to 6) of
  the released training split.
- **Probe:** 8 sampled answers a map (temperature 0.7, top-p 0.9, at most 1,024 new
  tokens), each scored by the reference evaluator. The prompt is the released one plus
  one sentence asking the model to find the start, the goal and the holes and trace a
  safe path before the boxed answer. No label, layout fact or helper image enters it.
- **Runs:** R1, the released model asked to reason, on the 100 new maps; R0, the released
  model with the released prompt alone, on the same maps; RS, the released model asked to
  reason, on the 100 small training maps; RE, e2 asked to reason, on the 100 new maps.

**Readings (fixed before the run).**

**Primary: maps R1 solves with at least one of its 8 answers.**

| Outcome | Reading | Next |
| --- | --- | --- |
| 25 or more of 100 | Label-free reasoning can start at 7 x 7 and 8 x 8 | STaR round 1: 8 reasoned answers on each map of a pool, the released model trains on the passing ones. |
| Fewer, and RS 25 or more of 100 | It can start only on smaller grids | STaR from the smallest grids up, a size at a time. |
| Fewer on both | The released model's reasoning gives too little signal | Report; label-free written reasoning needs another start. |

25 is the floor because STaR's first round on a pool of 8,000 maps would then keep about
2,000 maps or more, half of e8000's 4,039.

**Second: R1 against R0,** maps solved by any answer, paired by map, exact McNemar at
0.05. Does asking for reasoning help before any training?

**Descriptive (no verdicts):** each run by size and by moves away needed; mean answer
length and answers that reached the cap; RE's answers (whether e2 writes reasoning when
asked); 20 maps' answers kept in full for reading.

**Validity:** every map is answered 8 times in each run, and R1 and R0 read the same maps.

**Limits:** one seed, 100 maps a run. pass@8 under sampling is a ceiling on what one
STaR round keeps, not a score. The released prompt asks for the fewest moves; any passing
route counts.

**Implementation:** `spara.reasoning_probe` (`reasoning-probe` in the CLI);
`scripts/pace/reasoning_probe.sbatch`.

### Result: the probe (jobs 7272027 to 7272030)

**Validity: passes.** Every map was answered 8 times in each run, and R1 and R0 read the
same 100 maps. The zeros are not a parsing failure: of R1's kept answers, 72 walk into a
hole and 88 end off the goal, and answers in the same `\boxed{R,D}` form pass on the small
maps.

**Primary: R1 solves none of the 100 new maps with any of its 8 answers, and RS solves 21
of the 100 small maps.** Both are under 25, so per the table **the released model's own
reasoning gives too little signal** for label-free training on its reasoning.

| Run | Maps solved by any of 8 answers | Answers that pass | Mean answer length (tokens) |
| --- | --- | --- | --- |
| R1, released model asked to reason, new 7 x 7 and 8 x 8 maps | 0 of 100 | 0% | 554 |
| R0, released model, released prompt, same maps | 0 of 100 | 0% | 275 |
| RS, released model asked to reason, small training maps | 21 of 100 | 3.6% | 413 |
| RE, e2 asked to reason, same new maps | 57 of 100 | 33% | 18 |

**Second: R1 against R0,** no detectable difference (0 against 0).

**Descriptive:**
- **The released model cannot read these maps.** Its reasoning names cells off the grid,
  such as (8, 11) on an 8 x 8 map, and places holes where there are none.
- **RS by size:** 15 of 25 at 3 x 3, 4 at 4 x 4, 2 at 5 x 5, none at 6 x 6. All 21 are
  maps that need no detour.
- **RE:** e2 ignores the instruction and writes the route alone (18 tokens). 51 of the
  74 maps that need one move away, 5 of 16 that need two, 1 of 10 that need three or more.
- **Answers at the cap:** R1 225 of 800, R0 207, RS 111, RE none.

**Reading.** The released model's weak point here is reading the map, not reasoning about
it. LIVR's training on the released maps is what taught the e-chain to read them, and the
e-chain no longer writes reasoning. Label-free written reasoning would need a start that
both reads the map and still reasons; neither model is one.

### Is the search budget the limit? b256

Round 3's search, with e2 as the policy and at most 64 policy calls a map, left 15,514 of
31,693 maps unsolved, and solved only 60 of the 3,389 maps that need four or more moves
away. A label-free round can only train on what its search finds.

**Question: of the maps e2's search left unsolved at 64 calls, how many does it solve with
up to 256?**

- **Maps:** `fresh/detour-frontier-78h`, 500 maps drawn at random (seed 212) from the
  15,514 left unsolved (214 at 7 x 7, 286 at 8 x 8; 131 need one move away, 175 two, 100
  three, 94 four or more).
- **Search:** round 3's final-verdict-only search with e2 as the policy, at most 256 policy
  calls a map, routes of at most 64 moves, in four parts of 125 maps. The search is best
  first and deterministic, so its first 64 calls repeat round 3's.
- No training; evaluation only.

**Readings (fixed before the run).**

| Outcome | Reading | Next |
| --- | --- | --- |
| 50 or more of 500 solved | The budget limits what a round finds | The next round searches its frontier with up to 256 calls. |
| Fewer | The budget is not the limit | Deep detours need a better policy, not a longer search. |

**Descriptive (no verdicts):** maps solved by moves away needed and by size; policy calls
the solved maps needed (65 to 128, 129 to 192, 193 to 256); routes equal to the label,
another shortest, or longer.

**Validity:** the four parts cover each of the 500 maps once, every kept route passes the
reference evaluator, and no map is solved within 64 calls (a few may be, where the GPU type
reorders near-ties; their count is reported).

**Budget:** four jobs of about an hour on any GPU of 40 GB or more; the limit is 180
minutes. The recipe is `configs/detour_scaling_78h.json`.

### Result: does e2 use its latents? (job 7274384)

A first run (job 7271385) stopped before any reading: the feature file's name took the
set's path separator. The file name was fixed and the run repeated.

**Validity: passes.** The identity check passes (each map's own latent states patched back
give the same answers), and all 400 maps are answered in every condition. e2's own greedy
answers here solve 159; its recorded pass solved 160, and the two agree on 395 of 400 maps
(a different GPU type).

**Primary: removing the latent tokens leaves e2 with 0 of the 400, against 159 with its
own** (159 maps lost, none gained, p < 0.001). **e2's answers use the latents.**

**Second: given another map's latents, e2 writes that map's own route exactly in 240 of
400 answers** (unmasked), and in all 400 under the Stage 1 mask. It solves its own map in
4. **The route is set by what the latents carry.**

| Unmasked | Solved (of 400) | Routes changed | Answers equal to the donor's route |
| --- | --- | --- | --- |
| Own latents | 159 | | |
| Another map's latents | 4 | 397 | 240 |
| Mean latent of the 400 maps | 0 | 400 | |
| Latent tokens removed | 0 | 400 | |

**Descriptive:**
- **Under the Stage 1 mask:** own 149, swap 3 (donor's route in 400), mean 1, removed 0.
- **By moves away needed, own latents:** as e2's recorded pass (151, 9 and 0 of 288, 64
  and 48).
- **Against the LIVR battery:** LIVR on the benchmark's 3 x 3 to 6 x 6 maps gave the
  donor's route in 351 of 400; e2 on 7 x 7 and 8 x 8 detour maps gives it in 240.

**Reading.** As in the LIVR battery, the four latents carry what decides the route, and
the answer follows them even when it can read the image. The LIVR battery's probes found
map content in the latents' middle layers and the route's first move in their late layers
(0.84 to 0.97 accuracy at layer 36). The latents compute in one pass, 36 layers deep, at
four positions; nothing feeds them back.

**Limits:** removing the tokens gives a prompt e2 never saw in training, so the swap is the
stronger evidence. The swap cannot tell a map the answer plans from apart from a route the
latents already hold; the LIVR probes suggest both are there, at different depths.

### Result: e3 (job 7271268)

**Validity: all checks pass.**
1. **The search** (jobs 7215275 to 7215305 and 7215307): the 32 parts cover maps 0 to
   31,692 once each. All 16,179 kept routes pass the reference evaluator, in the jobs and
   in a local re-check.
2. **The sets, the whole run and the cap:** the set checks pass, the run took 2,148
   optimizer steps (17,179 examples) and answered all 538 maps, and none of the 400
   held-out answers reached the cap.

**Primary: e3 solves 197 of the 400 held-out maps, e2 160.** 45 maps only e3 solves, 8
only e2, p < 0.001. **Round 3 pays.** Per the table, round 4 runs the same way, on newly
drawn maps and the frontier.

**Second: on the 112 held-out maps that need two or more moves away, e3 solves 15 and e2
9.** 8 maps only e3 solves, 2 only e2, p = 0.11. No detectable gain on deep detours; none
of the 48 that need three or more is solved. The gain is on the 288 that need one move
away: 182 against 151.

**Third: e3 solves 197, n24000 273** (15 only e3, 91 only n24000, p < 0.001). The chain has
trained on 24,486 found routes over three rounds; n24000 on 32,000 labels over two passes.

**What the search found.** 16,179 of the 31,693 maps, in 1,298,747 policy calls (median 12
when solved) and 48.1 GPU-hours.

| Moves away needed | Frontier | Natural mix | Deep detours | All |
| --- | --- | --- | --- | --- |
| One | 569 of 1,903 | 11,535 of 14,004 | | 12,104 of 15,907 |
| Two | 597 of 2,755 | 1,668 of 3,441 | 1,087 of 2,248 | 3,352 of 8,444 |
| Three | 135 of 1,590 | 293 of 1,383 | 235 of 980 | 663 of 3,953 |
| Four or more | 21 of 1,445 | 21 of 1,172 | 18 of 772 | 60 of 3,389 |
| All | 1,322 of 7,693 | 13,517 of 20,000 | 1,340 of 4,000 | 16,179 of 31,693 |

- **Against round 2's search:** on the natural mix e2's search solves 82% of the maps that
  need one move away, 48% of those that need two and 21% of those that need three;
  e8000's solved 78%, 34% and 12%.
- **Routes:** 7,838 equal the label, 7,343 are another shortest route, 998 are longer.

**Descriptive:**

| | e3 | e2 | e8000 | EB3 | n8000 | n24000 |
| --- | --- | --- | --- | --- | --- | --- |
| Held-out 7 x 7 (198) | 109 | 92 | 78 | 51 | 91 | 144 |
| Held-out 8 x 8 (202) | 88 | 68 | 63 | 32 | 71 | 129 |
| Moves away needed: one (288) | 182 | 151 | 135 | 81 | 154 | 233 |
| Moves away needed: two (64) | 15 | 9 | 6 | 2 | 8 | 33 |
| Moves away needed: three or more (48) | 0 | 0 | 0 | 0 | 0 | 7 |
| Test detour maps (40) | 21 | 19 | 20 | 19 | 15 | 22 |
| Reference benchmark (400, sampled) | 354 | 343 | 334 | 330 | 345 | 369 |

- **Benchmark (post hoc):** 354 against LIVR's 328 (p < 0.001); 328 of the 360 maps that
  need no detour, against LIVR's 319.
- **Seen maps:** the first 98 of the pool, all from the frontier. The search solved 20; e3
  solves 10 of those and 1 of the other 78.
- **By shortest route length:** 4 to 7 moves 140 of 173, 8 to 11 moves 57 of 175, 12 or
  more 0 of 52.
- **Failures:** 202 step into a hole, 1 stops short.
- **Loss:** 0.100 over the run, 0.065 over the last quarter.
- **Time:** 4,933 seconds of training; the job took 91 minutes.

**Limits:** one seed. The pool was larger and of a different mix than round 2's, so the
gain is this round's as run, not a round's alone.

### No labels, round 4: e4, and fed-back latents: e4l

**Round 4 (e4).** e3's table says round 4 runs the same way, on newly drawn maps and the
frontier.

- **Pool:** `fresh/detour-pool-78j`, 49,514 maps; no label enters the search or training.
  - **New maps:** the 30,000 of `fresh/detour-pool-78i` (seed 213; 20,892 need one move
    away, 5,196 two, 2,224 three, 1,688 four or more), maps 0 to 29,999.
  - **Deep detours:** maps 8,000 to 11,999 of `detour-pool-78e`, not searched before
    (2,264 need two moves away, 959 three, 777 four or more), maps 30,000 to 33,999.
  - **The frontier:** the 15,514 maps round 3's search left unsolved, maps 34,000 to
    49,513.
- **Search:** with e3 as the policy, as round 3: maps 0 to 33,999 at most 64 policy calls a
  map. The frontier's budget follows b256's table: up to 256 calls if b256 solves 50 or
  more of its 500, otherwise 64. Routes of at most 64 moves, parts of 1,000 maps.
- **Arm e4:** one epoch from e3 (job 7271268) on the 1,000 released training maps plus each
  map the search solves, once, with the found route.

**Fed-back latents (e4l).** e2 decides its route in its four latents, which get one pass
of 36 layers; maps that need several moves away may need more serial steps. e4l is e4
with one extra pass over the prompt: the latents' last-layer states, rescaled to their
embeddings' size and times a learned gate, are added to the latents' inputs in a second
pass, and the second pass answers. The gate starts at 0, so e4l starts as e3 exactly, and
it has its own learning rate (1e-3) so it can move within one epoch.

- **Arm e4l:** e4's rows, order and recipe, with `latent_loops` 1
  (`configs/detour_scaling_78j_loops.json`). e4l is trained on routes e3's search found,
  not its own.

**Readings (fixed before the run).**

**Primary for round 4: e4 against e3 (197) on the 400 held-out maps,** paired by map, exact
McNemar, two-sided, at 0.05.

| Outcome | Reading | Next |
| --- | --- | --- |
| e4 solves more, p < 0.05 | Round 4 pays | Round 5 the same way. |
| No detectable difference | Returns have slowed | Read e4l: the lever is the policy's form. |
| e3 solves more, p < 0.05 | Round 4 harms | Check the loss and the mix of found routes. |

**Primary for the fed-back latents: e4l against e4 on the 400 held-out maps,** the same
test.

| Outcome | Reading | Next |
| --- | --- | --- |
| e4l solves more, p < 0.05 | A second latent pass raises the score | Round 5 searches and trains with fed-back latents; then try two extra passes. |
| No detectable difference | No detectable gain from a second pass | Read the gate and the deep half before more passes. |
| e4 solves more, p < 0.05 | The second pass harms | Check the gate and the loss. |

**Second (both): the 112 held-out maps that need two or more moves away** (e3: 15), e4
against e3 and e4l against e4, the same test.

**Validity (a reading is withheld if a check fails):**
1. **The search:** the parts cover each of the 49,514 maps once, and every kept route passes
   the reference evaluator.
2. **The sets, the whole run and the cap,** as n8000, with one optimizer step for every
   eight training examples. The set check leaves out the sets the pool is taken from.
3. **For e4l:** the same rows and steps as e4, and the gate ends at least 0.05 from 0 (else
   the extra pass was not taken up and the reading is "not used").

**Descriptive (no verdicts):** the search by source and by moves away needed; e4 and e4l by
size and by moves away; the gate's value; the benchmark; the loss; training time.

**Limits:** one seed. e4l trains on routes found by a policy without the extra pass; its own
search would differ. The frontier's budget is chosen by b256, a test on 500 of its maps.

### Result: b256 (jobs 7281410 to 7281413)

**Validity: passes.** The four parts cover each of the 500 maps once, all 148 kept routes
pass the reference evaluator, and no map is solved within 64 calls: the longer search
repeats round 3's first 64 calls.

**Primary: with up to 256 policy calls, e2's search solves 148 of the 500 maps it had left
unsolved at 64.** 50 or more, so per the table **the budget limits what a round finds**,
and the next round searches its frontier with up to 256 calls.

**Descriptive:**

| Moves away needed | Solved at 256 (of maps unsolved at 64) |
| --- | --- |
| One | 55 of 131 |
| Two | 72 of 175 |
| Three | 14 of 100 |
| Four or more | 7 of 94 |

- **Calls the solved maps needed:** 75 took 65 to 128, 44 took 129 to 192, 29 took 193 to
  256.
- **By size:** 62 of 214 at 7 x 7 and 86 of 286 at 8 x 8.
- **Routes:** 60 equal the label, 77 are another shortest route, 11 are longer.
- **Cost:** 4.4 GPU-hours for the 500 maps.

**Limits:** 500 maps, one policy (e2). Round 4's frontier is searched by e3.
