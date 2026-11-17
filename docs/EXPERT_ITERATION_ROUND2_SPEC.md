# Expert iteration round 2 on larger maps, with LIVR's bottleneck

Status, 2026-11-12: run as jobs 6992798 (search), 6992799 (E2) and 6992800 (EB2), after the
CPU precondition (job 6989964). All validity checks passed, after the planned fallback
from TF32 batched reads. Manipulation check: **the bottleneck did not take**, so the
primary has no reading on perception (EB2 78 against E2 84 of 400, p = 0.46, descriptive).
Secondary: **round 2 helps on larger maps** (E2 84 against E 54, p = 1.5 × 10⁻⁵). See
[Result](#result-jobs-69927986992800).

Rerun, 2026-11-14, from the round 1 rerun's E (jobs 7065168–7065176): run and read. Round 2
helps again (E2 72 against E's 56 of 400, p = 0.033), and EB2 shows no detectable difference
from E2 (76 against 72). See [Rerun](#result-jobs-70651687065176).

- **Sizes:** 7 x 7 and 8 x 8 only. The user approved them and the GPU budget on
  2026-11-12.
- **Superseded:** the 2026-11-11 version of this spec trained the latents with
  layout-derived map labels ("map grounding"). The user chose LIVR's bottleneck instead,
  which uses no labels: SPARA's goal is perception without helper images or extra targets.

The readings below are fixed before any run.

This follows round 1 in the [expert iteration spec](EXPERT_ITERATION_SPEC.md#result-jobs-69618746961876).
It draws on the [latent battery](LATENT_BATTERY_RESULTS.md#probes), the
[direct fine-tuning control](SFT_CONTROL_RESULTS.md) and the
[JEPA judge](JEPA_JUDGE_SPEC.md#result-job-6951675).

## Question

This run asks two questions:
1. **Does a second round of expert iteration, from E's weights, carry to larger maps?**
   The new maps are 7 x 7 and 8 x 8, beyond LIVR's 3 x 3 to 6 x 6 training range.
2. **Does training under LIVR's bottleneck add to expert iteration?** Under LIVR's Stage 1
   mask, the prompt and the answer cannot attend to the image; only the four latent
   tokens can. Training on found routes under the mask makes the latents carry what
   those routes need. The only supervision is still the verifier's accept or reject.

**Why expert iteration.** In round 1, E solved 88 of 240 held-out detour maps against
42 for LIVR. The outcome table's next step was a second round from E's weights.

**Why the bottleneck.**
- **It is LIVR's own way of improving perception.** In LIVR's ablation (Table 5), adding
  latent tokens without the mask matched plain fine-tuning (79.5, 61.2, 58.2 against
  79.5, 61.2, 58.9). With the mask, LIVR scored 83.6, 64.8 and 67.8.
- **It needs no labels or helper images,** and it applies to any task with an image and
  an answer.
- **Combined with expert iteration,** the search supplies the targets and the mask decides
  where learning from them goes.

**The evidence against it, at 3 x 3 to 6 x 6:**
- **On the released maps it added nothing measurable over plain fine-tuning:** 323 of
  400 for LIVR, against 320 and 323 for the two direct fine-tuning arms.
- **LIVR's latents already carry nearly everything there.** With the mask on at test
  time, LIVR solves 318 of 400, against 328 unmasked.

**Why larger maps might differ.**
- **Hole information fades with size.** The latent battery's hole probe falls from 0.88
  at 3 x 3 to 0.75 at 6 x 6 (balanced accuracy, layer 12). The JEPA judge's hole AUROC
  falls from 0.93 at 3 x 3 to 0.76 at 8 x 8.
- **Search becomes rare.** With LIVR as the policy, final-verdict-only search solved 7 of
  18 detour maps at 7 x 7 and 3 of 17 at 8 x 8 ([JEPA judge run](experiments/jepa-judge-6951675/README.md)).

## Design

**New sets** (generated on CPU with `spara.fresh_maps`):

| Set | Contents | Seed | Purpose |
| --- | --- | --- | --- |
| `fresh/detour-heldout-78` | 400 detour maps, 7 x 7 and 8 x 8 drawn with equal weight | 205 | The new held-out set |
| `fresh/detour-pool-78` | 500 detour maps, drawn the same way | 204 | Round 2's new search pool |

- **Exclusions:** both sets exclude every layout in train, test, `probe`, `detour-pool`,
  `detour-heldout`, `jepa-train` and each other. The held-out set is generated first;
  the pool also excludes it.
- **8 x 8 is the generator's limit.** The renderer keeps the released 64-pixel cells
  only up to 512 pixels.

**Search.** One job, with E's final weights as the policy. Final-verdict-only search as
in round 1: budget 64 policy calls, depth cap 64, the reference evaluator. It searches,
in this order:
1. **The frontier:** the 174 maps of round 1's pool that LIVR's search failed.
2. **The 500 maps of `detour-pool-78`.**

**Training data, both arms.** The arms start from E's final weights. They train on the
released 1,000 maps plus 2,000 extra examples:
- **1,000 extra examples** are drawn in turn from the solved 7 x 7 and 8 x 8 pool maps,
  in pool order.
- **The other 1,000** are drawn in turn from every solved 3 x 3 to 6 x 6 pool map: round
  1's 326, plus any frontier maps E's search solves.
- **Expected yield:** LIVR's search solved 29% of 7 x 7 and 8 x 8 detour maps. If E's
  solves 30–50% of the new pool, that gives 150–250 maps, each repeated 4–7 times.
- **Variety floor:** if E's search solves fewer than 100 of the 500 new maps, the
  fine-tunes are not run. They are held for a decision. Round 1's detour scaling check
  showed that few distinct maps teach less.

**The schedule, both arms.**
- **Shuffle, then split.** The 3,000 rows are shuffled with seed 42 and split into two
  phases: the first 1,496 rows (187 optimizer steps) and the other 1,504 (188 steps).
  That is 375 steps in all, as in round 1.
  - Shuffling first gives both phases released maps and found routes.
  - Each released map is seen in one phase only. Each solved pool map is repeated 3–7
    times, so most appear in both.
- **Each phase** starts a fresh optimizer with round 1's recipe: learning rate 1e-5,
  cosine schedule with 5% warmup, accumulation 8.
- **50/50,** because LIVR's FrozenLake recipe gives its two stages 10 epochs each.

**Arms.**

| Arm | Phase 1 (1,496 rows) | Phase 2 (1,504 rows) | Run |
| --- | --- | --- | --- |
| E2, round 2 | Unmasked | Unmasked | New |
| EB2, round 2 under the bottleneck | LIVR's Stage 1 mask | Unmasked | New |
| E, round 1 | — | — | Job 6961875; evaluated on the new sets in the search job |
| L, LIVR | — | — | Evaluated on the new held-out set in the search job |

- **E2 and EB2 differ only in the mask during phase 1.** Both restart the optimizer
  between phases, so the restart is not a difference between them. (The direct
  fine-tuning control showed that a restart alone changes the loss curve.)
- **E2 differs from round 1's recipe** by that restart. E2 against E therefore measures
  round 2 with this schedule.

**Evaluation.** For each arm, and for E in the search job:
- **Unmasked greedy decoding** (as deployed) on:
  - the 400 new held-out maps;
  - the 240 held-out maps of round 1 (3 x 3 to 6 x 6);
  - the first 98 maps of `detour-pool-78`, as seen maps, split into trained and untrained.
- **Masked greedy decoding** on the 400 new held-out maps: the Stage 1 mask at test time,
  so the answer sees the image only through the latents. This is the latent battery's
  bottleneck mode.
- **Latent states:** the four latent tokens' hidden states at decoder layers 12, 20 and 28
  (one-based, as in the latent battery). They are read in an unmasked prompt pass on the
  500 pool maps and the 400 new held-out maps.
- **The reference benchmark** for E2 and EB2: one sampled answer per test map, as the
  harm check.
- **L** gets unmasked greedy on the 400 new held-out maps only.

## Readings (fixed before the run)

**Manipulation check: did the bottleneck put more of the map into the latents?**
- **The probe:** ridge regression from the four latents' layer-12 states (concatenated) to
  hole or not at each cell, as in the latent battery. One probe per grid size and arm,
  fitted on that size's pool maps and scored on that size's held-out maps. Strength is
  chosen by leave-one-out error.
- **The score:** balanced accuracy per cell, averaged over cells, then over the two
  sizes. The map layouts are used here only to score the probe, never in training.
- **Took:** EB2's score is at least 0.05 above E2's, and the 95% interval of the
  difference lies above zero. The interval comes from a paired bootstrap: 2,000
  resamples of held-out maps within each size, with the probes' predictions fixed.
- **Why 0.05:** the mask adds what the routes need, not every hole, so this margin is
  half the earlier map-grounding spec's 0.10. It is a judgment, fixed before the run.
- **Why not masked success as the gate:** EB2 practises answering under the mask and E2
  does not. EB2 beating E2 there could mean it learned that condition, not that its
  latents hold more. Masked success is reported, not used as the gate.

**Primary: EB2 against E2** on the 400 new held-out maps. Paired by map, exact McNemar,
two-sided.
- **The bottleneck adds to expert iteration:** the check took, and EB2 solves more,
  p < 0.05.
- **The bottleneck harms expert iteration:** the check took, and EB2 solves fewer,
  p < 0.05.
- **No detectable difference:** the check took, and neither of the above holds.
- **The bottleneck did not take; no reading on perception:** the check failed. The
  McNemar result is still reported, as descriptive.

**Secondary: E2 against E** on the same 400 maps, with the same test.
- **Round 2 helps on larger maps:** E2 solves more, p < 0.05.
- **Round 2 harms on larger maps:** E2 solves fewer, p < 0.05.
- **No detectable difference:** otherwise.

**Power.** With exact McNemar, a 2:1 split of discordant maps reaches p < 0.05 only with
about 42 of them. A 3:1 split needs about 20. That is why the new held-out set has 400
maps.

**Descriptive (no verdicts):**
- **Masked greedy:** E, E2 and EB2, and each one's gap between unmasked and masked
  success. If E2's gap is already small, the mask had little to force in.
- **The probe** at layers 20 and 28, for E, E2 and EB2. Goal and start cell accuracy
  at layer 12.
- **The frontier:** of the 174 maps LIVR's search failed, how many E's search solves, by
  size.
- **The search at 7 x 7 and 8 x 8:**
  - maps solved by size, and policy calls;
  - the found routes: equal to the label, another shortest route, or longer, and wall
    bumps.
- **E against L** on the new held-out maps.
- **Harm checks:**
  - each arm against E on round 1's 240 held-out maps;
  - E2's and EB2's benchmark against LIVR and against E: overall, detour and no-detour.
- **Failures:** into a hole, stops short, or passes the goal.
- **Seen maps,** split into trained and untrained.
- **Training:** each phase's last-quarter answer loss. The masked phase may start high,
  because E was trained unmasked.
- **GPU-seconds per job.**

**Validity (the readings are withheld if any check fails):**
1. **Policy reads.** If reads are batched, they must agree with single passes at the same
   precision, computed in the same job, within 0.01.
   - **When:** at step 0 of 52 pool maps before searching, and on 32 cached reads after
     the frontier search.
   - **Single passes are computed in the job.** Round 1's reference reads were LIVR's, so
     they don't apply to E.
   - **Fallback:** round 1's reads, one request per pass under bfloat16 autocast, with the
     frontier searched again.
2. **Every accepted route** passes the reference evaluator.
3. **Sets:** every label passes the reference evaluator, and no set pair above shares a
   layout.
4. **Loading:** E's greedy outcomes on round 1's 240 held-out maps match round 1 on at
   least 228 of 240 (95%).

**How the outcomes would steer the plan:**

| Outcome | Next |
| --- | --- |
| Took; the bottleneck adds to expert iteration | Keep it in later rounds. A rendering decision for maps beyond 8 x 8 becomes worth making. |
| Took; no detectable difference | The latents hold more of the map and the routes don't change. On FrozenLake the gap is in choosing moves; drop perception work here. |
| Took; the bottleneck harms | Look at the masked phase's loss and the harm checks before anything else. |
| Did not take | The mask did not change what the latents hold at these sizes. A label-free alternative is predicting the image from the latents (not in scope). |
| Round 2 shows no gain, or harms, on larger maps | Look at the frontier and the yield. If search rarely solves 7 x 7 and 8 x 8 maps, expert iteration needs exploration help there. |

## Precondition: batched policy reads (CPU, before any GPU job)

Round 1's batched reads differed from single passes by up to 0.049. The cause is not
established; the CPU tests used a tiny stand-in model, not the real vision encoder or
real image sizes.

**The check:** a CPU job on PACE (`ice-cpu`) loads E's weights. It reads 16 requests from
different maps, sizes 3 x 3 to 8 x 8, in one batch and one at a time, twice:
- in full precision;
- under bfloat16 autocast, the GPU's arithmetic.

**Readings:**
- **Full precision agrees within 1e-4, and bfloat16 shows a gap near round 1's:** the code
  is correct and the gap is rounding. The fix is to read batches in full precision on the
  GPU, checked again in the job by validity check 1.
- **Full precision disagrees:** it is a bug. Fix it, and test again on CPU.
- **Neither:** search with one request per pass, at the higher budget.

### Result (CPU job 6989964, 26 minutes on one 12-core `ice-cpu` node)

**The code is correct and the gap is rounding.**

| | Batched against single, largest difference | Four equal-length rows, no padding |
| --- | --- | --- |
| Full precision | 2.4 × 10⁻⁶ | 4.8 × 10⁻⁷ |
| bfloat16 autocast | 0.030 (one 5 x 5 request); 0.006 or less for the other 15 | 0.0028 |

- **Padding does not predict the gap.** The 3 x 3 rows, the most padded, were exact under
  bfloat16. The largest gaps fell on mid-sized maps.
- **The full-precision fix, as written, would not pay.** True FP32 matmuls on an H200 run
  at roughly a fifteenth of bfloat16's rate, so batches of 16 would be no faster than
  round 1's single reads.

**The fix adopted: the FP32 weights with TF32 matmuls for the search's reads.**
- **Why TF32:** its rounding is about eight times finer than bfloat16's, at roughly half
  bfloat16's throughput. The batched gap should therefore fall well under 0.01, but this
  is a prediction, not a measurement.
- **The in-job check is at the same precision.** The single-pass reference is one request
  per pass, also in TF32.
- **The fallback is round 1's reads:** one request per pass under bfloat16 autocast. It
  runs if either in-job check fails. The step-0 check fails before any search; the
  cached-read check fails after the frontier search, which is then searched again.
- **Only the search's policy reads use TF32.** Every greedy answer, state dump and
  training step stays under bfloat16 autocast. The settings record the read precision.
- **The tolerance stays at 0.01.** Measuring how far single bfloat16 reads are from full
  precision would be needed to justify any other number.

## Result (jobs 6992798–6992800)

Evidence: [docs/experiments/expert-round2-6992798/](experiments/expert-round2-6992798/README.md).
8,323 GPU-seconds (139 GPU-minutes) by Slurm's elapsed time, against about 125 for the
fallback case.

**Validity: all checks passed.**
- **TF32 batched reads failed their check by 0.0007:** 0.0107 at step 0 of the 52 maps,
  and 0.00054 on cached reads. The job fell back to round 1's reads and searched the
  frontier again. There, the 52 maps matched exactly.
- **Two departures from the plan above, both costs, not validity:**
  - **When the code acts:** it acts on the step-0 check only after the frontier search,
    not before any search as written above. So the frontier was searched twice, about 15
    extra GPU-minutes.
  - **TF32 was not faster:** its frontier pass took about 15 minutes against 12 for the
    one-at-a-time pass.
- **Loading:** E matched round 1's outcomes on all 240 maps.
- **Routes:** all 248 accepted routes pass the reference evaluator.
- **Sets:** labels 900 of 900, and no shared layouts.

**The search.**
- **Frontier:** 47 of the 174 maps solved (5 of 13 at 4 x 4, 17 of 56 at 5 x 5, 25 of
  105 at 6 x 6).
- **New pool:** 201 of 500 solved (107 of 231 at 7 x 7, 94 of 269 at 8 x 8).
  - **Routes:** 108 are the label, 76 are another route of the same length, and 17 are
    longer.
  - **Wall bumps:** 9 of the routes move into a wall.

**Manipulation check: did not take.**
- **The scores:** the hole probe at layer 12 scores 0.657 for EB2 and 0.653 for E2,
  against 0.649 for E.
- **The difference:** 0.004, with a 95% interval of −0.004 to 0.011. The margin was 0.05.

**Primary, descriptive only:** EB2 solves 78 and E2 84 of the 400 new held-out maps (20
only EB2, 26 only E2, p = 0.46).

**Secondary: round 2 helps on larger maps.** E2 solves 84 and E 54: 39 maps only E2
solves, 9 only E, p = 1.5 × 10⁻⁵.
- **By size:** 51 against 35 at 7 x 7 (p = 0.0037), and 33 against 19 at 8 x 8
  (p = 0.0026).
- **For scale:** L solves 17.

**The outcome-table rows that apply:**
- **"Did not take":** the mask did not change what the latents hold at these sizes. The
  next step it gives, predicting the image from the latents, is outside this spec.
- **The secondary reading:** round 2 helps on larger maps.

**Descriptive.**
- **Masked greedy:**

  | | Unmasked | Masked | Gap |
  | --- | --- | --- | --- |
  | E | 54 | 48 | 6 |
  | E2 | 84 | 66 | 18 |
  | EB2 | 78 | 73 | 5 |

  EB2 practised under the mask, and the probe shows no added map content. So EB2's
  smaller gap is not read as perception.
- **Round 1's 240 held-out maps:** E2 102 and EB2 100, against E's 88 (p = 0.013 and
  0.029). Round 2 did not cost the smaller maps.
- **Benchmark against LIVR** (LIVR 328 overall, 9 on detour maps, 319 on no-detour maps):

  | | Overall | Detour maps | No-detour maps |
  | --- | --- | --- | --- |
  | E2 | 322 (p = 0.48) | 19 (p = 0.0020) | 303 (p = 0.017) |
  | EB2 | 325 (p = 0.73) | 17 (p = 0.0078) | 308 (p = 0.043) |

  Neither arm differs detectably from E.
- **Holes still cause most failures:** E2 walks into a hole on 295 of the 400 new
  held-out maps.
- **Seen maps:** both arms solve none of the 53 pool maps the search failed.
## Result (jobs 7065168–7065176)

Evidence: [docs/experiments/expert-iteration-round2-7065168/](experiments/expert-iteration-round2-7065168/),
with `comparison.json` holding every reading below. The latent states (`states.pt`, 43 MB
per arm) and the arms' weights are not committed.

**What changed from the plan.**
- **E is the rerun's E (7063729),** not job 6961875. Its weights were not available.
  See the [round 1 rerun](EXPERT_ITERATION_SPEC.md#rerun-jobs-70631567063729). So the
  frontier is the 162 maps the rerun's search failed, not 174.
- **L's greedy answers came from a separate job (7063726),** run before the search. The
  search job copied them in.
- **The arms' final weights were not saved.** Nothing reads them after evaluation, and
  scratch had room for one 15 GB save, which E needed.

**Jobs.** 7,236 GPU-seconds (2 h 1 min) by Slurm's elapsed time, against an estimate of
about 125 minutes. L's greedy answers took 241, the search 3,529, E2 1,718 and EB2 1,748.
The CPU batching check took 1,595 seconds on `ice-cpu`.

**Precondition: the batching check gave the first reading.**
- **Full precision:** batched and single reads agree within 2.2 × 10⁻⁶.
- **bfloat16 autocast:** they differ by up to 0.014.
- **Meaning:** the code is correct, and the gap is rounding.
- **Not acted on:** the search was submitted at one request per pass before the check
  finished. It took 59 minutes, inside the one-request budget.

**Validity: all checks passed.**
- **Policy reads:** one request per pass, so the step-0 and cached-read differences are 0.
- **Sets:** 900 of 900 labels accepted, and no shared layouts.
- **Loading:** E's greedy outcomes on round 1's 240 held-out maps match the rerun on 240
  of 240.
- **Both arms' set checks pass.**

**Search: 199 of the 500 new pool maps,** above the variety floor of 100.
- **By size:** 108 of 231 at 7 x 7 (47%) and 91 of 269 at 8 x 8 (34%). LIVR's search
  solved 29% in the JEPA judge run.
- **Policy calls:** median 16 on solved maps.
- **Routes:** 112 are the label, 17 are longer, and 2 bump a wall.
- **Frontier:** 49 of the 162 maps LIVR's search failed. By size: 3 of 8 at 4 x 4, 17 of
  53 at 5 x 5 and 29 of 101 at 6 x 6.

**Manipulation check: took, by the fixed rule. But EB2 did not gain over E.**
- **The reading:** at layer 12, EB2 scores 0.664 and E2 0.609, a difference of 0.055. The
  95% interval is 0.047 to 0.063. The margin is 0.05.
- **The starting point scores higher than both:** E scores 0.669. Unmasked training
  lowered the latents' hole reading, and training under the mask kept it at E's level.
  The mask preserved what was there; it did not add to it.
- **The same pattern holds for the start cell:** E 0.76 and 0.57 (7 x 7, 8 x 8), E2 0.57
  and 0.42, EB2 0.78 and 0.59.
- **At layers 20 and 28** the three arms are within 0.02 of each other.
- **The interval is narrow because it resamples maps only.** It does not include
  training-run variance, and there is one run per arm.

**Primary: no detectable difference.** On the 400 new held-out maps, EB2 solves 76 and E2
72: 19 maps only EB2 solves, 15 only E2, p = 0.61. By size: 45 against 43 at 7 x 7 and 31
against 29 at 8 x 8.

**Secondary: round 2 helps on larger maps.** E2 solves 72 and E 56: 33 maps only E2
solves, 17 only E, p = 0.033. Neither size alone is significant: 43 against 35 and 29
against 21.

**The outcome-table row that applies:** "Took; no detectable difference". Its next step is
to drop perception work on FrozenLake. Because the check took only by protecting E's
level, the reading is weaker than the row assumes. The run shows that holding the latents
at E's level does not change success. It does not show what latents that hold more would
do.

**Descriptive.**
- **Held-out success (400 new maps):** L 16, E 56, E2 72, EB2 76. E against L: 40 maps
  only E solves, 0 only L, p = 1.8 × 10⁻¹².
- **Masked greedy:** E 51, E2 69, EB2 73. Each arm loses 3–5 maps without direct image
  access. The answers already use the image almost only through the latents.
- **Training loss:** EB2's masked phase starts at 0.113 against E2's 0.102 and ends at
  0.072 against 0.068. The mask had little to force.
- **Failures:** walking into a hole ends 307–310 of the 400 attempts for E, E2 and EB2,
  and 361 for L.
- **Hole walks (layer-12 probe).** For each failure, does the probe read the hole that
  ended the attempt as a hole?
  - E 124 of 310 (40%), E2 82 of 308 (27%), EB2 114 of 307 (37%).
  - **Holes the route passed beside without entering** are read as holes about as often:
    E 40%, E2 31%, EB2 42%.
  - The probe's reading does not separate the holes the model walks into from the ones it
    avoids. Either the decoder does not use what the probe reads, or the probe misses
    what the decoder uses (the latent battery found the probe understates it).
- **Round 1's 240 held-out maps:** E2 98 and EB2 103, against E's 92 (p = 0.43 and
  0.099). Neither arm harms the smaller maps.
- **Benchmark:**
  - EB2 ties E exactly, 325 and 325. Against LIVR: 325 to 323 overall, 16 to 7 on detour
    maps (p = 0.012) and 309 to 316 on no-detour maps (p = 0.30).
  - E2 scores 318: against LIVR, 13 to 7 on detour maps (p = 0.070) and 305 to 316 on
    no-detour maps (p = 0.071).
- **Seen maps (first 98 of the new pool):** the search solved 37. E2 solves 32 of those
  and EB2 28; of the other 61, they solve 1 and 2.

## Limits

- **One training run per arm.** McNemar treats the map as the unit, not the run.
- **Selection.** Training maps are those E's search solves, so they skew easy within each
  size.
- **E's no-detour harm carries forward.** Continuing from E keeps the harm it showed on
  no-detour maps; the benchmark check measures it.
- **The masked phase starts from an unmasked-trained model.** E may partly bypass its
  latents, and the unmasked phase has 188 steps to repair any damage.
- **The probe measures every hole,** while the mask pushes in what the routes need. The
  probe also understates what the model uses (latent battery). So the check can miss a
  real change.
- **Sizes stop at 8 x 8.**

## Scope

- **Generated maps** are approved for this line of work.
- **No layout-derived training targets.** Both arms train on the released answers and
  on routes the verifier accepted. The layouts score the probe only.
- **No step simulator or judge.** As in round 1.

## Budget

| Job | Contents | One request per pass | TF32 batched (if the in-job check passes) |
| --- | --- | --- | --- |
| Search | Loading E and L; checks; E's greedy on 738 maps and masked greedy on 400; E's states on 900 maps; search of 174 frontier maps and 500 new maps; L's greedy on 400 | about 65 minutes | about 30–35 minutes |
| E2 | Two-phase fine-tune (375 steps); greedy on 738 maps, masked greedy on 400; states on 900; benchmark | about 30 minutes | the same |
| EB2 | The same, with the mask in phase 1 | about 30 minutes | the same |

- **Total:** about 125 GPU-minutes at one request per pass, or about 90–95 with TF32
  batched reads. If the in-job check fails after the frontier, the search job adds about
  5 minutes to the higher figure.
- **TF32 read time is a guess:** about 0.02 s per read in batches of 16, against 0.073 s
  for round 1's single reads.
- **The search estimate:**
  - **New pool:** assumes E solves about 35% of the 500 new maps, at a median of 15 calls,
    and the unsolved maps use all 65. That is about 23,500 reads.
  - **Frontier:** most of the 174 maps are likely to use all 65 calls: about 11,000 reads.
  - **Read time:** about 0.085 s, against round 1's 0.073 s, for the larger images.
- **Generation time:** round 1's arms took about 0.3 s per answer on 3 x 3 to 6 x 6 maps.
  7 x 7 and 8 x 8 answers are about twice as long, so 0.4 s per answer is assumed overall.
- **Storage:** each arm's final weights take about 15 GB; the first phase's weights are
  not saved. Scratch has about 62 GB free. Deleting M's weights from round 1 is advisable
  but not needed.

## Implementation (new code)

1. **`training.train_stage`:** an option not to save the final weights (phase 1).
   **`expert_iteration.batched_decide`:** a read precision, defaulting to round 1's.
2. **`spara/expert_round2.py`:**
   - a check that the checkpoint is E: round 1's recipe in its `complete.json`;
   - the batching check for the precondition;
   - the search stage: in-job single-pass checks, the frontier, then the new pool;
   - training rows from both rounds, shuffled and split into the two phases;
   - the arms, the evaluation populations, masked greedy and the state dumps;
   - a comparison that applies the readings above, with the probe and the bootstrap.
3. **The two sets,** generated with `spara.fresh_maps`.
4. **The rest, as in round 1:** CLI, a settings file, job scripts, and CPU tests with
   the stand-in model.

## Running

```bash
python -m spara.cli expert-round2 --stage batching-check --checkpoint /path/to/E/stage2/final \
  --data-root data/vsp --output /path/to/batching                      # CPU
python -m spara.cli expert-round2 --stage search --checkpoint /path/to/E/stage2/final \
  --livr-checkpoint /path/to/livr/stage2/final --round1-routes /path/to/round1-search/routes.jsonl \
  --round1-greedy /path/to/E/greedy.jsonl --data-root data/vsp --output /path/to/search
python -m spara.cli expert-round2 --stage train --arm eb2 --checkpoint /path/to/E/stage2/final \
  --round1-routes /path/to/round1-search/routes.jsonl --search-run /path/to/search \
  --data-root data/vsp --output /path/to/eb2                           # and --arm e2
```

On PACE, [scripts/pace/expert_round2.sbatch](../scripts/pace/expert_round2.sbatch) with
`SPARA_STAGE` (`batching-check`, `search` or `train`). Search needs
`SPARA_LIVR_CHECKPOINT`; train needs `SPARA_ARM` and `SPARA_SEARCH_RUN`. `SPARA_BATCH` and
`SPARA_READ_PRECISION` override the search's reads; `SPARA_BATCH=1 SPARA_READ_PRECISION=bf16`
gives round 1's. `expert_round2.compare` applies the readings.

The sets were generated with the renderer pinned in `requirements/fresh-maps.txt`:

```bash
python -m spara.fresh_maps --name detour-heldout-78 --sizes 7,8 --detour-count 400 --weights 1,1 \
  --seed 205 --exclude probe,detour-pool,detour-heldout,jepa-train
python -m spara.fresh_maps --name detour-pool-78 --sizes 7,8 --detour-count 500 --weights 1,1 \
  --seed 204 --exclude probe,detour-pool,detour-heldout,jepa-train,detour-heldout-78
```

## Not in scope

- **Maps larger than 8 x 8.** They need a rendering decision: beyond 512 pixels, the
  renderer shrinks cells below the released 64 pixels.
- **A labels control (M) for round 2.** Round 1 answered that comparison.
- **Other perception objectives,** possible later:
  - predicting the image's features from the latents (label-free);
  - map grounding or step grounding (layout-derived targets).
- **Perception benchmarks** (BLINK, MMVP, V*) and other problems.
