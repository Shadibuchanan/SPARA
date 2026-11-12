# Expert iteration round 2 on larger maps, with LIVR's bottleneck

Status, 2026-11-12: specified and being built. Nothing is generated or submitted.

- **Sizes:** 7 x 7 and 8 x 8 only. Carried over from the 2026-11-11 discussion; the user
  did not object, but has not confirmed it explicitly.
- **The GPU budget** is the user's decision, after the CPU precondition below settles
  which line of the [budget](#budget) applies.
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
1. **Policy reads.** If reads are batched, they must agree with single passes computed in
   the same job, within 0.01.
   - **When:** at step 0 of 52 pool maps before searching, and on 32 cached reads after
     the frontier search.
   - **Single passes are computed in the job.** Round 1's reference reads were LIVR's, so
     they don't apply to E.
   - **Fallback:** one request per pass, as in round 1, with the frontier searched again.
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

| Job | Contents | One request per pass | Batched (if the precondition holds) |
| --- | --- | --- | --- |
| Search | Loading E and L; checks; E's greedy on 738 maps and masked greedy on 400; E's states on 900 maps; search of 174 frontier maps and 500 new maps; L's greedy on 400 | about 65 minutes | about 30 minutes |
| E2 | Two-phase fine-tune (375 steps); greedy on 738 maps, masked greedy on 400; states on 900; benchmark | about 30 minutes | the same |
| EB2 | The same, with the mask in phase 1 | about 30 minutes | the same |

- **Total:** about 125 GPU-minutes at one request per pass, or about 90 batched.
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
`SPARA_LIVR_CHECKPOINT`; train needs `SPARA_ARM` and `SPARA_SEARCH_RUN`. `SPARA_BATCH`
overrides the reads per pass. `expert_round2.compare` applies the readings.

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
