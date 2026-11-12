# Detour scaling check: does layout variety help detour learning generalize?

Status, 2026-11-10: run (jobs 6941468–6941471). All validity checks passed. Primary:
**variety helps**. Secondary: **detour fine-tuning helps**. No arm improves the overall
benchmark, because each loses no-detour maps. See [Result](#result-jobs-69414686941471).
The reading rules were fixed before the run, and the user approved running it.

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
