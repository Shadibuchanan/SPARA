# Expert iteration with final-verdict-only search (step 1b, round 1)

Status, 2026-11-11: run as jobs 6961874 (search), 6961875 (E) and 6961876 (M). All
validity checks passed, after the planned fallback from batched reads. Primary: **expert
iteration helps** (88 against 42 of 240 held-out maps). Secondary: **no detectable
difference between found routes and labels** (88 against 97, p = 0.21). See
[Result](#result-jobs-69618746961876). The readings were fixed before the run, and the
user approved step 1b with this route source.

This is step 1b of the [guided-search spec](GUIDED_SEARCH_SPEC.md), part of the
[detour-learning spec](DETOUR_LEARNING_SPEC.md).

## Question

Can LIVR improve its own detour routes from routes it finds by search, when the search
sees only the verifier's accept or reject? That search uses no step simulator, no learned
judge and no labels. It needs what most RL problems offer: candidate actions, a policy
and a final check.

**Why this route source.**
- **It was stronger than expected.** In the [JEPA judge run](JEPA_JUDGE_SPEC.md#result-job-6951675),
  final-verdict-only search with LIVR as the policy solved 55 of the 97 unseen detour maps
  within 64 policy calls. On 3 x 3 to 6 x 6 maps it solved 45 of 62.
- **The learned judge is not usable.** The JEPA judge solved 29 of 97 and misses most
  holes, so it is not used here.
- **It avoids a scope question.** Issue #8 asks whether per-step simulator feedback is
  within "simulator rewards". This search uses only the final verdict, so only the
  question of generated maps remains. The user approved generated maps for this line of
  work.

## Design

**Maps.** The first 500 maps of `fresh/detour-pool`: 11 of 3 x 3, 70 of 4 x 4, 161 of
5 x 5 and 258 of 6 x 6. These are V500's maps in the
[detour scaling check](DETOUR_SCALING_SPEC.md), so its arms are pairable references.

**Search.** Step 1a′'s final-verdict-only search, exactly as the JEPA run's third arm ran
it.
- **Policy:** the LIVR rerun's final Stage 2 weights (`baseline-6253300`), with its raw
  move probabilities and, for ending, its probability of closing the answer.
- **Budget and cap:** 64 policy calls per map, and a depth cap of 64.
- **Verifier:** the reference evaluator.
- **What a solved map gives:** its first accepted answer, the most probable answer under
  LIVR that the search verified within the budget. It need not be a shortest route.

**Batching.** The policy reads up to 16 requests per forward pass, rows from different
maps right-padded together. The JEPA run read one request per pass. Results are cached
per (map, prefix), as before. This is the guided-search spec's deferred optimisation,
built here to save GPU time.

**Arms.** Each fine-tunes the LIVR rerun's final Stage 2 weights with the detour-scaling
recipe: the released 1,000 training maps plus 2,000 examples, one epoch, learning rate
1e-5, cosine schedule with 5% warmup, accumulation 8, seed 42.

| Arm | The 2,000 extra examples | Run |
| --- | --- | --- |
| E, expert iteration | The found routes, drawn in turn from the solved maps in pool order | New |
| M, matched control | The same maps in the same order, with their shortest-route labels | New |
| L, no fine-tune | — | Detour scaling, job 6941468 |
| V500 | All 500 maps, 4 times each, with their labels | Detour scaling, job 6941470 |

- **E and M differ only in the answers.** They have the same maps, repeats and order. M
  is FrozenLake-only: few problems have a shortest-route oracle.
- **M against V500** shows what dropping the maps the search failed costs.

**Evaluation,** as in the detour scaling check:
- **Greedy decoding on 338 maps:** the 240 held-out maps (the 200 of `fresh/detour-heldout`
  and the 40 test detour maps) and the first 98 pool maps (seen).
- **The reference benchmark,** one sampled answer per test map, for E only. It is the
  harm check: every detour-scaling arm lost 9–16 no-detour maps.

## Readings (fixed before the run)

**Primary: E against L** on the 240 held-out maps. Paired by map, exact McNemar,
two-sided.
- **Expert iteration helps:** E solves more, p < 0.05.
- **Expert iteration harms:** E solves fewer, p < 0.05.
- **No detectable difference:** otherwise.

**Secondary: E against M** on the same 240 maps, with the same test.
- **Found routes teach better than labels:** E solves more, p < 0.05.
- **Found routes teach worse than labels:** E solves fewer, p < 0.05.
- **No detectable difference between found routes and labels:** otherwise. Where the
  found routes equal the labels, E and M train on the same answers. Then they can differ
  only through training noise; two identical LIVR runs disagree on 31 of 400 test maps.

**Descriptive (no verdicts):**
- **The search:** maps solved by size, policy calls, refused answers, and how many
  found routes equal the label or are longer.
- **The comparisons by population and size;** E and M against V500.
- **Seen maps,** split into those the search solved (trained on) and the rest.
- **E's benchmark** against LIVR: overall, detour and no-detour maps.
- **The 97 detour maps searched again** with the batched policy, against the JEPA run's
  55 (same maps and routes).
- **The last quarter's training loss,** and GPU-seconds per job.

**Validity (the readings are withheld if any check fails):**
1. **The batched policy reads what single passes read.**
   - At step 0's 52 primary points, its move probabilities are within 0.01 of the reach
     check's. All 52 are requested together, so rows of different maps share passes.
   - 32 cached answers from the search are each within 0.01 of a single pass.
   - **Fallback:** if either fails, the job switches to one request per pass. It then
     searches again, and must pass the 52-point check at that setting.
2. **Every accepted route** passes the reference evaluator. The job stops otherwise.
3. **Sets:** every pool and held-out label passes the reference evaluator, and the sets
   share no layout. This is the detour scaling check's first two checks, rerun in each
   fine-tune job.

**How the outcomes would steer the plan:**

| Outcome | Next |
| --- | --- |
| E helps, with no detectable difference from M | Final-verdict-only search is a usable route source on FrozenLake. Next: a second round from E's weights, or a second problem. |
| E helps, worse than M | The found routes teach, but less than labels. Look at the maps where they differ. |
| No detectable difference from L | One round of self-found routes on 500 maps shows no gain. More maps or rounds before a second problem. |
| E harms | Check E's routes and loss before anything else. |

## Result (jobs 6961874–6961876)

Evidence: [docs/experiments/expert-iteration-6961874/](experiments/expert-iteration-6961874/README.md).
4,403 GPU-seconds (73 GPU-minutes) by Slurm's elapsed time, within the estimate of
55–75.

**Validity: all checks passed.**
- **Batched reads failed their check,** off by up to 0.049 at the 52 points and 0.059 on
  cached answers. The job fell back to one request per pass, as planned. There it matched
  the 52 points to 5 × 10⁻⁷, and the 97 detour maps reproduced the JEPA run exactly: the
  same outcome and route on all 97.
  - **Cost:** the search took 36 minutes instead of 8–14.
  - **Cause:** not established. The CPU tests did not use the real vision encoder or
    real image sizes.
- **Routes:** all 326 accepted routes pass the reference evaluator.
- **Sets:** labels 2,200 of 2,200, and no shared layouts.

**The search solved 326 of the 500 pool maps.**
- **By size:** 11 of 11 at 3 x 3, 57 of 70 at 4 x 4, 105 of 161 at 5 x 5 and 153 of 258
  at 6 x 6.
- **Routes:**
  - 212 are the label, 81 are another route of the same length, and 33 are longer.
  - 19 of the longer routes move into a wall, which leaves the player in place.

**Primary: expert iteration helps.** On the 240 held-out maps:
- E solves 88 and L 42: 50 maps only E solves, 4 only L, p = 3.8 × 10⁻¹¹.
- **On the 40 released test detour maps alone:** 15 against 8, p = 0.039.

**Secondary: no detectable difference between found routes and labels.**
- **Held out:** E solves 88 and M 97: 16 maps only E solves, 25 only M, p = 0.21.
- **On the 40 test detour maps:** they tie, 15 and 15.
- **The point estimate favours the labels.** The test had little power: 212 of the 326
  found routes are the labels.

**The outcome-table row that applies:** "E helps, with no detectable difference from M".
The next step it gives is a second round from E's weights, or a second problem.

**Descriptive.**
- **M ties V500,** 97 and 97. Dropping the 174 maps the search failed cost nothing
  detectable.
- **E did not learn the wasted moves.** 86 of its 88 held-out solutions are shortest
  routes, against 40 of 42 for L.
- **Seen maps:**
  - The search solved 62 of the first 98 pool maps, and E solves 60 of those 62.
  - Of the 36 it did not solve, E solves 0 and V500, trained on them with labels, 12.
- **Holes still cause most failures:** E walks into a hole on 147 of the 240 held-out maps.
- **Benchmark against LIVR:**
  - overall 320 against 328 (p = 0.26);
  - detour maps 15 against 9 (p = 0.031);
  - no-detour maps 305 against 319 (p = 0.020).
  - The no-detour loss, a net 14 maps, is the harm the spec expected. The detour-scaling
    arms lost 9–16.

## Limits

- **One round, one training run per arm.** McNemar treats the map as the unit, not the
  run.
- **Selection.** Routes exist only for maps the search solves, which skew small. M and
  V500 show what that costs.
- **The detour-heavy mix** (2,000 of 3,000 examples) is the detour scaling check's. The
  no-detour harm it showed is expected again. A mix at the released detour share is a
  separate question.
- **The pool covers 3 x 3 to 6 x 6,** LIVR's training sizes.
- **Greedy decoding** for the held-out reading; the benchmark samples.

## Scope

- **Training-time search only.** The benchmark and its decoding are unchanged.
- **Generated maps** are approved by the user for this line of work. For reported
  conditions, the question stays open in issue #8.
- **No step simulator, no judge and no labels** in E. M uses labels as a control only.

## Budget

| Job | Contents | Estimate |
| --- | --- | --- |
| Search | Loading; the checks; the 97 detour maps (about 3,900 reads); the 500 pool maps (about 15,400 reads) | 8–14 minutes batched; about 30 if it falls back |
| E | Fine-tune (375 steps; V500 took 906 s), greedy on 338 maps, benchmark, loading and saving | about 25 minutes |
| M | The same without the benchmark | about 20 minutes |

- **Total:** about 55–75 GPU-minutes in three one-H200 jobs. E and M start when the
  search job completes, and can run at the same time.
  - **Wall limits:** 45 minutes for the search and E, 35 for M.
  - **Accounting:** GPU-seconds are reported from Slurm, across any retries.
- **The pool's reads** assume about 31 per map: the mean policy calls of the JEPA run's
  final-verdict-only arm by size, weighted by the pool's mix. One read took about 0.08 s
  unbatched.
- **Storage:** E's and M's final weights, about 15 GB each. E's are kept for a possible
  second round.

## Running

```bash
python -m spara.cli expert-iteration --stage search --settings configs/expert_iteration.json \
  --config configs/livr.json --data-root data/vsp --checkpoint /path/to/stage2/final \
  --probe-points /path/to/detour-probe/points.jsonl \
  --reference /path/to/jepa-judge/searches-verdict_only.jsonl --output /path/to/search
python -m spara.cli expert-iteration --stage train --labels search --routes /path/to/search/routes.jsonl \
  --settings configs/expert_iteration.json --config configs/livr.json --data-root data/vsp \
  --checkpoint /path/to/stage2/final --output /path/to/expert-search      # and --labels shortest
```

On PACE, [scripts/pace/expert_iteration.sbatch](../scripts/pace/expert_iteration.sbatch)
with `SPARA_LIVR_CHECKPOINT` and `SPARA_STAGE` (`search` or `train`); for `train`, also
`SPARA_LABELS` and `SPARA_ROUTES`. `expert_iteration.compare` applies the readings.

## Not in scope

- A second round, more maps, or other route sources (the exact simulator or the JEPA
  judge).
- A mix at the released detour share.
- Any change to the benchmark or its decoding.
