# Expert iteration round 3 on 7 x 7 and 8 x 8 maps

Status, 2026-11-15: run and read: no detectable difference. See [Result](#result-jobs-70881347088136).
The readings below were fixed before any round 3 result.

This follows [round 2](EXPERT_ITERATION_ROUND2_SPEC.md#result-jobs-70651687065176) and the
[perception ceiling](PERCEPTION_CEILING_SPEC.md). The ceiling found that at 7 x 7 and
8 x 8, choosing moves limits success, not perception. The lever that moved success was
expert iteration: E2 solved 72 of 400 new held-out maps against E's 56 (p = 0.033).

## Question

Does a third round of expert iteration still add on 7 x 7 and 8 x 8 maps?

A second question rides along, descriptive only: does a better policy find more routes?
Round 3's search uses round 2's model as the policy, on maps drawn the same way as round
2's pool.

## Design

**The base, B: EB2's recipe, rerun with its weights saved.** Round 2 did not save its
arms' weights (scratch space), so round 3 starts by training EB2 again.
- **The same recipe and rows:** from E (job 7063729), on round 2's 3,000 rows, with LIVR's
  Stage 1 mask in phase 1. The refactored row builder reproduces EB2's rows exactly
  (SHA-256 of the 3,000 rows, checked against job 7065176's inputs).
- **Why EB2 and not E2:** EB2 is at or above E2 on every reading, though none is
  significant alone. On held-out maps it scores 76 against 72, masked 73 against 69, and
  103 against 98 on round 1's maps. On the benchmark it ties E at 325, against E2's 318.
  E2's benchmark loss was on no-detour maps (305 against LIVR's 316, p = 0.071), and
  later rounds inherit it.
- **The same training run is not the same weights.** GPU training is not bitwise
  repeatable, so B is compared with round 2's EB2 as a description, not a check.

**New pool:** `fresh/detour-pool-78b`, 500 detour maps, 7 x 7 and 8 x 8 with equal
weight, seed 206. It excludes every layout in train, test and the other fresh sets,
`detour-pool-78` and `detour-heldout-78` included.

**Search.** One job, with B as the policy. Final-verdict-only search as in rounds 1 and 2:
budget 64 policy calls, depth cap 64, the reference evaluator. It searches, in order:
1. **The frontier:** the 301 maps of `detour-pool-78` that E's round 2 search failed.
2. **The 500 maps of `detour-pool-78b`.**

The job first evaluates B as round 2's search job evaluated E.

**EB3: round 3's arm.** It starts from B and uses EB2's recipe unchanged: 3,000 rows,
shuffled with seed 42, phase 1 of 1,496 rows under the Stage 1 mask, then phase 2 of
1,504 rows unmasked. Each phase uses a fresh optimizer, learning rate 1e-5, cosine
schedule with 5% warmup, accumulation 8. The rows are:
- the released 1,000 maps;
- **1,000** drawn in turn from every solved 7 x 7 and 8 x 8 map: round 2's 199 on
  `detour-pool-78`, round 3's frontier solves on the same pool, then round 3's solves on
  `detour-pool-78b`, each pool in order;
- **1,000** drawn in turn from the 387 solved maps of round 1's pool, as in round 2.

**Variety floor:** if B's search solves fewer than 100 of the 500 new maps, EB3 is not
trained. That is held for a decision.

**Evaluation,** as round 2's arms: unmasked greedy on the 400 maps of
`detour-heldout-78`, round 1's 240 held-out maps and the first 98 maps of the new pool;
masked greedy on the 400; latent states; the reference benchmark.

## Readings (fixed before the run)

**Primary: EB3 against B** on the 400 maps of `detour-heldout-78`. Paired by map, exact
McNemar, two-sided. B's answers come from the search job.

| Result | Reading | Next |
| --- | --- | --- |
| EB3 solves more, p < 0.05 | Round 3 helps | Expert iteration still pays at these sizes. Run round 4 the same way. |
| No detectable difference | Round 3 adds nothing detectable | Returns have slowed below what 400 maps can detect. The next lever is the search itself (a larger budget, or search guided by the policy's move probabilities) or the policy's form (step-by-step answers). |
| EB3 solves fewer, p < 0.05 | Round 3 harms | Check the training loss and the row mix before anything else. |

**Power.** Round 2 gained 16 maps (72 against 56) from 50 discordant maps, 33 against 17,
p = 0.033. A gain of half that size would not be detected.

**Validity (the readings are withheld if any check fails):**
1. **B trained on EB2's rows:** the same counts (199 new maps, 387 old maps, 3,000 rows).
2. **The search,** as in round 2:
   - batched policy reads agree with single passes within 0.01, or the search falls back
     to one request per pass;
   - every accepted route passes the reference evaluator;
   - every label passes the reference evaluator, and no two sets share a layout;
   - B's greedy answers in the search job match the base job's on at least 228 of round
     1's 240 held-out maps (the loading check).
3. **EB3 and B answer the same maps.**

**Descriptive (no verdicts):**
- **B against round 2's EB2:** success on the 400 held-out maps and maps with the same
  outcome. This measures how far a rerun of the same recipe moves.
- **EB3 against E:** two rounds on top of round 1.
- **The search:**
  - the frontier: of the 301 maps E's search failed, how many B solves, by size;
  - the new pool, by size, against E's 199 of 500 on `detour-pool-78`. The maps differ
    but are drawn the same way.
  - policy calls, and found routes against the label.
- **Harm checks:**
  - EB3 against B on round 1's 240 held-out maps;
  - the benchmark: EB3 against B and against LIVR, and B against round 2's EB2; overall,
    detour and no-detour.
- **Failures:** into a hole, stops short, passes the goal.
- **Seen maps** (the first 98 of the new pool), split into trained and untrained.
- **Training loss** per phase, and GPU-seconds per job.

## Budget

| Job | Contents | Estimate |
| --- | --- | --- |
| B | EB2's recipe from E; evaluation and benchmark; save the weights | about 30 minutes |
| Search | B's evaluation; the frontier (301 maps) and the new pool (500) | about 70 minutes |
| EB3 | Two-phase fine-tune from B; evaluation and benchmark; save the weights | about 30 minutes |

- **Total:** about 2.2 GPU-hours.
- **The search estimate:** most frontier maps use all 65 calls (about 19,500 reads). If B
  solves 45% of the new pool at a median of 16 calls, the pool takes about 21,500. At
  round 2's rate (about 0.09 s per read), that is about 62 minutes, plus about 10 for
  the evaluation.
- **Storage:** B and EB3 each take about 15 GB. Scratch had 66 GB free after the user
  cleared `hf_cache`.

## Implementation

Round 3 reuses round 2's stages in `spara/expert_round2.py`, with `configs/expert_round3.json`:
- **The search** takes its frontier from `frontier_split` and `--frontier-routes`, and the
  loading check's answers from `--reference-greedy`.
- **The training rows** come from round 1's routes and from each later search's frontier
  and pool, in order (`--search-run` takes several runs). Each run's manifest names its
  splits. A map solved twice is an error.
- **`--save-weights`** saves an arm's final weights; `base_steps` names the starting
  checkpoint's step count (375 for E, 188 for B).
- **`round3`** applies the readings above.

## Running

On PACE, [scripts/pace/expert_round2.sbatch](../scripts/pace/expert_round2.sbatch), with
`R=$SPARA_REMOTE_ROOT/results`:

```bash
# B: EB2's recipe from E, weights saved (round 2's settings)
SPARA_STAGE=train SPARA_ARM=eb2 SPARA_SAVE_WEIGHTS=1 SPARA_ROUND=3 \
  SPARA_E_CHECKPOINT=$R/expert-train-search-7063729/stage2/final \
  SPARA_ROUND1_ROUTES=$R/expert-routes-7063156/routes.jsonl SPARA_SEARCH_RUN=$R/round2-search-7065168
# The search, with B as the policy
SPARA_STAGE=search SPARA_ROUND=3 SPARA_SETTINGS=configs/expert_round3.json \
  SPARA_E_CHECKPOINT=$R/round3-eb2-<B>/phase2/final SPARA_ROUND1_ROUTES=$R/expert-routes-7063156/routes.jsonl \
  SPARA_FRONTIER_ROUTES=$R/round2-search-7065168/routes.jsonl SPARA_ROUND1_GREEDY=$R/round3-eb2-<B>/greedy.jsonl \
  SPARA_LIVR_GREEDY=$R/round2-search-7065168
# EB3
SPARA_STAGE=train SPARA_ARM=eb3 SPARA_SAVE_WEIGHTS=1 SPARA_ROUND=3 SPARA_SETTINGS=configs/expert_round3.json \
  SPARA_E_CHECKPOINT=$R/round3-eb2-<B>/phase2/final SPARA_ROUND1_ROUTES=$R/expert-routes-7063156/routes.jsonl \
  SPARA_SEARCH_RUN="$R/round2-search-7065168 $R/round3-search-<search>"
```

The new pool was generated with the renderer pinned in `requirements/fresh-maps.txt`:

```bash
python -m spara.fresh_maps --name detour-pool-78b --sizes 7,8 --detour-count 500 --weights 1,1 \
  --seed 206 --exclude probe,detour-pool,detour-heldout,jepa-train,detour-heldout-78,detour-pool-78
```

## Limits

- **One training run per arm,** and B itself is a second draw of EB2's recipe.
- **The held-out set has now been read by many arms** (E, E2, EB2, O2, T2). No arm trains
  on it, and the reading here is fixed in advance.
- **Selection:** training maps are those the search solves, so they skew easy.
- **Sizes stop at 8 x 8.**

## Result (jobs 7088134–7088136)

Evidence: [docs/experiments/expert-iteration-round3-7088134/](experiments/expert-iteration-round3-7088134/),
with `round3.json` holding every reading below. B's and EB3's weights are on PACE.

**What changed from the plan.**
- **B's first submission (7086894) failed in its first second** on a shell quoting error in
  the job script, before any work. The script was fixed and B resubmitted as 7088134.
- **The search ran one request per pass.** Batched reads of B differed from single passes
  by 0.060, above the 0.01 limit, so the search fell back as specified. It took 6,202
  seconds against the 70-minute estimate.

**Jobs.** 9,454 GPU-seconds (2 h 38 min) by the jobs' own clocks: B 1,635, the search
6,202 and EB3 1,617.

**Validity: all checks passed.**
- **B trained on EB2's rows:** 199 new maps, 387 old maps, 3,000 rows.
- **The search:** 900 of 900 labels accepted, no shared layouts, and B's greedy outcomes
  on round 1's 240 held-out maps match the base job on 240 of 240.
- **EB3 and B answer the same maps.**

**Search: 228 of the 500 new pool maps,** above the variety floor of 100.
- **By size:** 110 of 206 at 7 x 7 (53%) and 118 of 294 at 8 x 8 (40%). E solved 199 of
  500 on round 2's pool: 47% and 34%. The maps differ but are drawn the same way.
- **The frontier:** 45 of the 301 maps E's search failed, 23 of 123 at 7 x 7 and 22 of
  178 at 8 x 8.
- **Policy calls:** median 13 on solved pool maps, against E's 16. 272 maps used the full
  budget.
- **EB3's rows** draw from 472 solved 7 x 7 and 8 x 8 maps, against 199 for B, so each
  map appears about twice in the 1,000 new rows rather than five times.

**Primary: no detectable difference.** On the 400 new held-out maps, EB3 solves 83 and B
76: 26 maps only EB3 solves, 19 only B, p = 0.37. By size: 51 against 44 at 7 x 7 (14
against 7, p = 0.19) and 32 against 32 at 8 x 8.

**The noise floor.** B reruns EB2's recipe on the same rows. B and EB2 both solve 76, but
they disagree on 20 maps, 10 each way. A gain of 7 sits close to what a rerun moves.

**Descriptive.**
- **Two rounds on top of round 1:** EB3 83 against E's 56, 39 maps only EB3 solves and 12
  only E, p = 0.0002. At 7 x 7, 51 against 35; at 8 x 8, 32 against 21.
- **Masked greedy (not a fixed reading):** EB3 93 against B 78, 28 against 13, p = 0.028.
  This comparison was not named in advance and is one of many read here, so it is a
  lead, not a finding.
- **Round 1's 240 held-out maps:** EB3 111, B 105 (p = 0.38). No harm.
- **Benchmark:** EB3 330, B 328, LIVR 323, round 2's EB2 325.
  - **Detour maps:** EB3 19 of 40, B 16, LIVR 7 (EB3 against LIVR: 12 against 0).
  - **No-detour maps:** EB3 311, B 312, LIVR 316 (EB3 against LIVR p = 0.47).
- **Failures on the 400:** EB3 walks into a hole on 292, B on 308. EB3 stops short on 24,
  B on 16.
- **Seen maps** (the first 98 of the new pool): of the 52 the search solved, EB3 solves 34
  and B 25. Of the 46 it failed, both solve none.
- **Training loss, last quarter:** EB3 0.076 and 0.065 for the two phases, B 0.072 and
  0.056.

**What this means.** Following the fixed table, returns have slowed below what 400 maps can
detect. Round 2 gained 16 maps on E and round 3 gains 7 on B, with no harm anywhere. The
search improved with the policy, so the loop still runs. But each round now adds less to
the greedy policy than one training run's spread, and holes remain the failure on three
maps in four. The next lever is the search or the policy's form, not a fourth round of
the same recipe. The masked gain is worth a fixed test before anything is built on it.
