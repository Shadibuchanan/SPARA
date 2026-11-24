# Latent JEPA: label-free perception for LIVR's latents

Status, 2026-11-16:
- **Gate 1 (CPU job 7104083):** passed. LIVR's image tokens show every hole: a linear probe on one
  cell's image-token state reads hole or not at balanced accuracy 1.00, at every layer from 4 to 36
  and every size from 3 x 3 to 6 x 6. See [Gate 1 result](#gate-1-result-cpu-job-7104083).
- **Built and tested on CPU:** `spara.perception_jepa`, the `perception-jepa` command and
  [scripts/pace/perception_jepa.sbatch](../scripts/pace/perception_jepa.sbatch).
  - **Dry run of `compare`** on round 2's real outputs, with EB2 standing in for EJ2: it reproduces
    round 2's probe scores (0.649, 0.653 and 0.657), E2's first-mistake counts below, EB2 against E2
    (78 against 84, p = 0.46), and EB2's benchmark (325 against LIVR's 328).
- **Gate job (7106850):** gates passed, 1,148 GPU-seconds. Target layer 4; β = 2.47; E's swapped
  answers follow the donor's route on 102 of 200 maps. A first submission (7105700) failed after 2
  seconds on a quoting error in the job script. See
  [the gate's evidence](experiments/jepa-gate-7106850/README.md).
- **EJ2 (job 7107309), 1,871 GPU-seconds:**
  - **Did not take:** the hole probe moved 0.657 → 0.664 (margin 0.05), so the primary is
    descriptive: EJ2 68 against EB2 78 of 400, p = 0.15.
  - **Secondary:** EJ2 solves fewer than E2 (68 against 84, p = 0.036).
  - **Post hoc, why:** the predictions are position codes that do not depend on the latents. The
    learned target projector found a shortcut. See [Result](#result-jobs-7106850-and-7107309).
  - **Budget:** 3,021 of the 3,300 approved GPU-seconds used.
- **Retry with fixed targets (EJF):** approved 2026-11-16 with an allowance of 2,400 GPU-seconds.
  - **Features (job 7116225):** 489 GPU-seconds.
  - **Gate 3 (CPU job 7116395): failed at 3c.** The targets passed. The warm-started heads explained
    nothing, so EJF was not run.
  - **Post hoc:** the context projector's layer norm left the predictor almost the same input for
    every map, which also applies to EJ2.
  - See [the retry's result](#result-jobs-7116225-and-7116395).
  - **Amended the same day** with the user's approval: the latents are standardised before the
    context projector, and the allowance is raised to 2,600 GPU-seconds. See
    [the amendment](#amendment-after-gate-3-2026-11-16).
  - **Amended gate 3 (CPU job 7117800): passed.**
    - Shuffled gap 0.278 at 7 x 7 and 0.027 at 8 x 8.
    - Evidence: [jepa-fixed-gate-7117800](experiments/jepa-fixed-gate-7117800/README.md).
  - **EJF, first job (7117889):** failed after 135 GPU-seconds. Map IDs repeat within the released
    training split, so a row was given another map's target. Maps are now keyed by image; see
    [the amendment](#amendment-after-gate-3-2026-11-16).
  - **EJF (job 7118300), 1,994 GPU-seconds: did not take.**
    - Hole probe 0.655 against EB2's 0.657.
    - Descriptive: 76 against EB2's 78 of 400 (p = 0.84). No detectable difference from E2 (76
      against 84, p = 0.28).
    - The term took hold (shuffled gap up, validation error down a little), but the latents did not
      gain target content that a linear readout can see; EB2's latents do as well.
    - See [EJF's result](#result-ejf-job-7118300).
  - **Budget:** 2,618 of the 2,600 approved GPU-seconds used, 18 over. Slurm killed the EJF job 18 s
    past its limit, after the run had finished.
- **Region latents, EJR (job 7120944), 1,832 of 2,400 GPU-seconds: did not take.**
  - Count-matched check: 0.667 against EB2's 0.689.
  - Descriptive: 80 against EB2's 78 (p = 0.88). No detectable difference from E2 (80 against 84,
    p = 0.62).
  - The sixteen tokens remain near-copies. See [EJR's result](#result-ejr-job-7120944).
- **Global + region latents, EGR (job 7122822), 1,781 of 2,100 GPU-seconds: took, and routes did not
  change.**
  - The masked block latents read their blocks at 0.934 against EB2's 0.689, and are fully
    separated (similarity −0.03).
  - Unmasked: 75 against EB2's 78 (p = 0.74).
  - Masked: 68 against 73 (p = 0.46).
  - By the pre-registered table, the limit is downstream of the latents, and the latent-perception
    line closes. See [EGR's result](#result-egr-job-7122822).
- **The swap test (job 7123991), 391 GPU-seconds:** the answer ignores EGR's block latents (93 of 100
  routes unchanged when they are averaged). The route follows the global latents (92 of 100 take a
  donor's route). See [its result](#result-the-swap-test-job-7123991).
- **EGD (jobs 7124258 and 7125522), 2,063 + 436 GPU-seconds:** the answer now reads the block latents
  (98 of 100 routes follow a donor's), but routes do not improve. Masked: 78 against EGR's 68
  (p = 0.11); unmasked: 78.
  - It still steps into holes its latents flag, so the limit is the decision process.
  - See [EGD's result](#result-egd-jobs-7124258-and-7125522).
  - **EGR's final weights were deleted** on 2026-11-16 at the user's request; EGD's are kept.
- **Position tracking on EGD (job 7128341), 1,019 GPU-seconds: inconclusive, because the probe is too
  weak.**
  - Routes reproduced EGD's on 400 of 400 maps in both modes.
  - The position probe reads the cell on 23% (unmasked) and 16% (masked, globals hidden) of solved
    steps, against the 50% floor.
  - [Step grounding](STEP_GROUNDING_SPEC.md#result) had already weighed against tracking as the
    limit. See [the result](#result-position-tracking-on-egd-job-7128341).
- **Global + patch latents, for images without a grid:** built and tested on CPU, not run. See
  [the design](#global--patch-latents-for-images-without-a-grid-built-not-run).

The readings below are fixed before any GPU run.

## Question

**Does a label-free objective on LIVR's latents, predicting the image's own cell embeddings, make
the latents hold the map, and does that change routes?**

This is the "did not take" next step of [round 2](EXPERT_ITERATION_ROUND2_SPEC.md#result-jobs-69927986992800):
LIVR's bottleneck did not change what the latents hold (hole probe 0.65 for E, E2 and EB2).

## Why

**The holes are lost between the image tokens and the latents.**
- **Image tokens:** 1.00 balanced accuracy for holes at every probed layer (gate 1, LIVR's
  checkpoint, 3 x 3 to 6 x 6).
- **Latents, same maps and checkpoint:** 0.75 to 0.88 at layer 12 ([latent battery](LATENT_BATTERY_RESULTS.md#probes)).
  At 7 x 7 and 8 x 8, E's latents give 0.65.
- **Where moves are chosen:** hole-ahead AUROC 0.60 to 0.81 for LIVR ([JEPA judge](experiments/jepa-judge-6951675/README.md)).

**The route is decided in the latents.** Given another map's latents, LIVR writes that map's route
in 73 to 99% of cases by size. So an objective at the latents acts where the plan is formed. Step
grounding put map information where moves are written and did not change routes.

**LIVR's own signal is sparse.** The bottleneck makes the latents carry what one route per map needs.
Direct fine-tuning without latents matched LIVR, so on these maps the answer can be produced without
per-cell perception. Predicting every cell's embedding is a dense signal, and it needs no labels.

**Failures are mostly local.** E2's first mistake on its 316 failed held-out maps (7 x 7, 8 x 8):
- **Into an adjacent hole, where a safe shortest move toward the goal existed:** 79. This is the
  cleanest perception failure.
- **Into an adjacent hole at a point where the route must turn away from the goal:** 124.
- **A safe move off every shortest route:** 91.
- **Stops short, a wall bump, or moves past the goal:** 22.

## The objective

- **Context:** the four latent tokens' states at decoder layer 12 (the round 2 probe's layer), through
  a context projector (parameter-free layer norm, then a 2-layer MLP to 256 dimensions).
- **Targets:**
  - each map cell's overlap-weighted mean of the image-token states at the target layer
    (the latent battery's cell pooling);
  - detached from the language model;
  - centred by the per-(size, row, column) mean over the training and pool maps, computed from E's
    features in the gate job;
  - through a target projector of the same shape.
- **Why centring:** without it, a learned target could encode position, which the predictor already
  knows. After centring, position decodes at chance (gate 1, below).
- **Predictor:** one hidden layer (512, GELU, dropout 0.1) from the four projected latents in order
  plus Fourier features of the cell's normalised position, to the cell's projected target. It is kept
  small so it cannot memorise training layouts from a map-identifying context.
- **Loss, as LeJEPA** ([arXiv 2511.08544](https://arxiv.org/abs/2511.08544); SIGReg as in
  galilai-group/lejepa MINIMAL.md, commit c293d29):
  - JEPA term = (1 − λ) · squared error + λ · mean of SIGReg on the context and target embeddings;
  - λ = 0.02, inside LeVLJEPA's stable range of 0.005 to 0.04;
  - no stop-gradient through the projectors. The target states are detached, so the language model
    is trained only through the latents.
- **SIGReg sample:** training reads one map per forward. Each statistic pools the current embeddings
  with queues of recent detached ones (512 latent vectors, 1,024 cells). Gradients reach only the
  current ones. No batch norm anywhere.
- **Total:** answer loss + β · JEPA term. β comes from the gate job's rule.
- **Heads:** AdamW at 1e-4, weight decay 0.01, on the language model's schedule. They are fitted
  first on E's frozen features (the warm start), so the first language-model updates do not follow a
  random predictor.
- **Clipping:** the heads are clipped at 1.0 on their own, outside the language model's clipping
  norm. Their gradient therefore cannot shrink the language model's updates against EB2's. Each
  step logs both norms.
- **Inputs:** the targets come from the image itself. No layout labels or helper images. The grid
  size is used to pool image tokens into cells.

**Error measure.** The target projector is learned, so raw squared error is not comparable over
training. Every reported error is the share of the targets' variance left unexplained (1 − R²).

## Gate 1 result (CPU job 7104083)

[scripts/probe_image_token_holes.py](../scripts/probe_image_token_holes.py) on the latent battery's
saved features (LIVR 6189457, unmasked):
- **Probe:** one ridge probe per size, from a cell's image-token state to hole or not; fitted on
  training maps, scored on test maps.
- **Threshold:** 0.95, set in the script before the run.
- **Compute:** one `ice-cpu` node, 3 minutes 56 seconds. No GPU.

| Layer | 3 x 3 | 4 x 4 | 5 x 5 | 6 x 6 |
| --- | --- | --- | --- | --- |
| 4, 8, 12, 16, 32 | 1.00 | 1.00 | 1.00 | 1.00 |
| 20, 24, 28, 36 | 1.00 | 1.00 | 0.999 | 1.00 |

- **No miss and no false alarm** on the 1,463 test holes at layers 4 and 12.
- **Position after centring (descriptive):** row and column decode at chance (0.11, 0.06, 0.04 and
  0.03 for 3 x 3 to 6 x 6, against 1/9, 1/16, 1/25 and 1/36).

Evidence: [experiments/image-token-holes-7104083](experiments/image-token-holes-7104083/README.md).

## Gate job (one GPU job, before the arm)

From E (round 1's fine-tune, `expert-train-search-6961875`), unmasked:

1. **Features:** one prompt pass per map:
   - the training maps, as round 2 builds its rows;
   - the 500 pool maps;
   - the 400 held-out maps;
   - 128 validation maps: pool maps that round 2's search did not solve, beyond the 98 seen maps.
2. **Gate 2a, the image tokens at 7 x 7 and 8 x 8:** gate 1's probe at layers 4, 8, 12 and 16, fitted
   on pool maps and scored on held-out maps.
   - **The target layer** is the earliest layer at 0.95 or above for both sizes.
   - **If none qualifies,** stop: the frozen vision side would be the limit.
3. **Centring means** at that layer, per size from 3 x 3 to 8 x 8, from the training and pool maps
   (never the held-out maps).
4. **Warm start:** 30 epochs on E's training-map features, batches of 32 maps, AdamW at 1e-3. The
   epoch with the lowest validation error is kept.
5. **Gate 2b, the loss weight:** on the first 16 rows of phase 1, under the Stage 1 mask, the norms of
   the answer loss's and the JEPA term's gradients into the language model.
   - **The rule:** β is the ratio of their medians, so the two terms start with equal gradient norms.
     This guards against the Scaffolding run's failure, where one term carried 26,000 times the
     other's gradient.
   - **Outside 0.01 to 100,** hold for a decision.
6. **Gate 2c, the route follows E's latents:** on the first 100 held-out maps of each size, greedy:
   - each map is answered with its own latents and with another map's (same size, different layout;
     every layer's states at the four latent positions, as the latent battery's swap);
   - **the check:** the swapped answer is the donor's own route on at least 50% of maps. Otherwise
     stop, because changing the latents would not change routes;
   - **identity check:** re-inserting a map's own states must reproduce its answer on the first four
     maps, or the gate is withheld;
   - **descriptive:** how often the swapped answer's first two moves are the donor's. E's routes at
     these sizes are long and mostly wrong, so exact equality can sit near the line for reasons
     unrelated to whether the latents set the route.
7. **E2's swap:** the same check, descriptive. E2 leans more on the direct image pathway (84 unmasked
   against 66 masked).

## The arm: EJ2

EJ2 is round 2's EB2 with the JEPA term in both phases:
- **Rows and order:** the same 3,000 rows in the same order.
- **Phase 1:** 187 steps under the Stage 1 mask.
- **Phase 2:** 188 steps unmasked.
- **Evaluation:** the same as round 2 (greedy on every population, masked greedy, latent states),
  then the benchmark.
- **Monitored:** the error on 32 validation maps and 32 training maps at the start and after each
  phase. Training error falling while validation error stays flat is memorisation.

**Validity (readings are withheld if any fails):**
1. The gate job passed.
2. **Matched start:** EJ2's first optimiser step's answer loss equals EB2's (0.100) within 0.1%. The
   JEPA term does not change the forward pass.
3. Round 2's set checks.

**Manipulation check:** round 2's latent hole probe at layer 12 (fitted on pool maps, scored on
held-out maps, balanced accuracy averaged over 7 x 7 and 8 x 8).
- **Took:** EJ2's score is at least 0.05 above EB2's, and the 95% paired-bootstrap interval of the
  difference lies above zero. This is round 2's margin.

**Primary: EJ2 against EB2** on the 400 held-out maps. Greedy, paired by map, exact McNemar, two-sided.
- **The latent JEPA adds to expert iteration:** the check took, and EJ2 solves more, p < 0.05.
- **The latent JEPA harms expert iteration:** the check took, and EJ2 solves fewer, p < 0.05.
- **No detectable difference:** the check took, and neither of the above holds.
- **Did not take, no reading on perception:** the check failed. The result is reported as descriptive.

**Secondary: EJ2 against E2,** with the same test. It does not depend on the manipulation check.

**Descriptive (no verdicts):**
- **First mistakes** on failed held-out maps, for E, E2, EB2 and EJ2. The count to watch is the step
  into a hole where a safe toward-goal move existed: E2's 79.
- **Participation ratio** of the latents at layer 12 (centred, channel-standardised), by size.
- **The JEPA's errors:** validation against training maps at the start and after each phase, and the
  logged shuffled-context gap (error with another map's latents minus own, as a share of target
  variance).
- **Round 2's other readings:** the probes at layers 20 and 28; masked greedy; the harm check on
  round 1's 240 held-out maps; the benchmark against LIVR overall, on detour maps and on no-detour
  maps; and failure types.
- **From the gate:** the warm start's validation error, start against best. This is how much of the
  cells' content E's frozen latents already predict.

**Power.** As round 2: with 400 maps, a 2:1 split reaches p < 0.05 at about 42 discordant maps.

**How the outcomes would steer the plan:**

| Outcome | Next |
| --- | --- |
| Took; adds | The latents' perception limited routes. Scale it: JEPA-only rows on the maps search cannot solve, then the 3 x 3 to 6 x 6 benchmark. |
| Took; no detectable difference | The latents hold more of the map and routes do not change. On FrozenLake the limit is in choosing moves; keep the objective as a perception method and test it elsewhere. |
| Took; harms | Look at β, the gradient norms and the harm checks before anything else. |
| Did not take | Check whether the prediction depended on the latents (shuffled gap) and whether validation error fell. Options: a larger β, a longer phase, more latent tokens. |
| A gate fails | Record it. 2a: the vision side is the limit. 2c: the latents do not set E's routes. |

## Result (jobs 7106850 and 7107309)

**Gate job 7106850: gates passed** (1,148 GPU-seconds).
- **2a:** E's image tokens separate holes at 1.00 at 7 x 7 and 8 x 8, at every candidate layer. The
  target layer is 4.
- **2b:** β = 2.47.
- **2c:** E's swapped answers are the donor's route on 102 of 200 maps; their first two moves are the
  donor's on 141; one keeps its own route.
- **A first submission,** job 7105700, failed after 2 seconds on a quoting error in the job script.
- Evidence: [jepa-gate-7106850](experiments/jepa-gate-7106850/README.md).

**EJ2, job 7107309** (1,871 GPU-seconds). Evidence:
[jepa-ej2-7107309](experiments/jepa-ej2-7107309/README.md).
- **Validity:** passed. The first-step answer loss equals EB2's.
- **Manipulation check: did not take.** The hole probe at layer 12 is 0.664 against EB2's 0.657
  (difference 0.007, interval −0.0002 to 0.015; the margin was 0.05).
- **Primary: no reading on perception.** Descriptive: EJ2 68 against EB2 78 of 400 (15 against 25,
  p = 0.15).
- **Secondary: EJ2 solves fewer than E2:** 68 against 84 (18 against 34, p = 0.036).
- **The perception count rose:** steps into a hole beside a safe toward-goal move, 90 against E2's 79.
- **Benchmark:** 323 against LIVR's 328, with the same trade as round 2's arms (detour 17 against 9,
  no-detour 306 against 319).

**Why it did not take (post hoc).** The outcome table's check, whether the prediction depended on the
latents:
- **It did not.** The predictions on held-out maps are position codes: under 0.04% of their variance
  differs between maps at the same cell, and holes read from them at chance. This held already for
  the warm-started heads on E's frozen latents.
- **Validation error fell anyway** (0.65 to 0.11), because the learned target projector made the
  targets mostly a nonlinear function of position. The predictor is told position, so it could hit
  the targets without the latents.
- **What the language model got:** the JEPA term was mostly SIGReg on the latents (85 to 98% of its
  value). That spread the latents (participation ratio 26–30 against 21–24) without adding hole
  content.
- **A second cause, found later** ([gate 3](#result-jobs-7116225-and-7116395)): only 0.003% to 0.04%
  of E's latents' squared norm differs between maps. The context projector's parameter-free layer norm
  leaves that unchanged, so the predictor saw almost the same input for every map. The latents could
  not have helped the prediction, with or without the shortcut.

**For a next attempt:**
- Gate on latent dependence at warm start: the predictions' map-dependent share, and holes read from
  the predictions. These need only CPU.
- Remove the position shortcut by construction: fixed targets, or per-position centring in embedding
  space.
- Weigh SIGReg against the prediction term by gradient.

## Retry with fixed targets: EJF

Approved by the user on 2026-11-16, after EJ2. Everything not listed here is as EJ2: context layer
12, the context projector, the predictor (512, dropout 0.1, Fourier position features), target layer
4, the cell pooling, the heads' optimiser and separate clipping, the warm start, the β rule, the
rows and phases, the evaluation.

**What changes:**
- **Fixed targets, computed once from E.**
  1. Each cell's layer-4 state from E's prompt pass, centred by the per-(size, row, column) mean over
     the training and pool maps, as EJ2.
  2. Each channel divided by its standard deviation over those centred cells. Qwen's residual stream
     has a few channels far larger than the rest.
  3. Projected on the top 256 principal components of those standardised cells: one PCA over all
     sizes. Not whitened, so the loss weighs the directions in which cells differ most.
  4. One scale factor, so the average target variance per component is 1 on those cells.
- **Stored per map** (split, map ID) with the heads. The arm reads them, so the targets cannot drift
  as layer 4 trains. Image tokens precede the latents, so the prompt pass sees the same states as the
  training pass.
- **Why this closes EJ2's shortcut:** after centring, every position's mean target is zero, and the
  steps after centring are linear. Position alone predicts zero, so any explained variance must come
  from the latents.
- **No SIGReg** (λ = 0). The targets are fixed, so they cannot collapse, and in EJ2 SIGReg made up
  most of the term. The JEPA term is the squared error alone.

**Job 1, features (GPU):** from E, unmasked, one prompt pass on EJ2's 2,185 maps (training, pool,
held-out and validation). It saves the layer-12 latents and the layer-4 cells. Nothing is read from
them on the GPU.

**Gate 3, on CPU (an `ice-cpu` job):**
- **3a, the targets show holes:** a ridge probe from a cell's target to hole or not, fitted on pool
  maps and scored on held-out maps, at least 0.95 at 7 x 7 and 8 x 8. Otherwise stop.
- **3b, the loss weighs holes:** the hole label explains at least 10% of the targets' variance on
  held-out maps at 7 x 7 and 8 x 8 (between-class over total sum of squares).
  - **If 3b fails,** recompute with 32 components and recheck 3a and 3b.
  - **If it still fails,** stop.
- **Warm start:** as EJ2, on E's latents and the fixed targets.
- **3c, the predictions depend on the latents:** the warm-started heads on held-out maps, at 7 x 7 and
  8 x 8.
  - **Shuffled gap of at least 0.02:** the share of target variance left unexplained with a same-size
    donor map's latents, minus the share with the map's own.
  - **Holes from the predictions at least 0.55:** a ridge probe, fitted on pool maps' predictions.
    E's latents read holes at 0.65.
- **Descriptive:**
  - the variance the components keep;
  - the targets' between-position share on held-out maps (zero by construction, up to sampling);
  - the warm start's validation error, start against best;
  - the predictions' within-position share, EJ2's measure;
  - the linear ceiling: one ridge per size from E's four latents, flattened, to every cell's target
    at once, scored as in 3c.

**If gate 3 fails:**

| Failure | Reading | Next |
| --- | --- | --- |
| 3a | The projection lost the holes | Stop. |
| 3b at 256 and at 32 components | Layer 4's cell variance is mostly not about holes | Stop; a deeper target layer is the user's call. |
| 3c; the linear ceiling passes | The predictor is the limit | Stop; the user decides. |
| 3c; the linear ceiling fails too | E's latents carry too little cell content for the term to grip at the start | Stop; the user decides. |

**Job 2, EJF (GPU), only if gate 3 passes.**
- **Starts with the β rule:** gate 2b's 16 phase-1 rows. Outside 0.01 to 100, it stops before
  training.
- **Then EJ2's arm,** with the fixed-target term.
- **Gate 2c is not rerun.** E's swap result (102 of 200) is a property of E, unchanged.
- **Monitored:** at the start and after each phase, the unexplained share on 32 validation and 32
  training maps, and the shuffled gap on the validation maps. Position predicts zero, so the gap and
  the validation error can fall only through the latents.

**Readings:** EJ2's, with EJF in its place: the same validity checks, manipulation check (margin
0.05), primary (EJF against EB2), secondary (EJF against E2), descriptive readings and outcome
table. Added descriptive: holes from the predictions and the shuffled gap on held-out maps, from the
saved latent states and final heads.

## Result (jobs 7116225 and 7116395)

Evidence: [jepa-fixed-gate-7116395](experiments/jepa-fixed-gate-7116395/README.md).

**Features, job 7116225:** 489 GPU-seconds, 2,185 maps.

**Gate 3, CPU job 7116395: failed at 3c.** EJF was not run.
- **3a and 3b passed.** Holes read from the targets at 1.00 at 7 x 7 and 8 x 8. The hole label
  explains 21% and 23% of the targets' variance; position explains under 1%.
- **3c failed.** The warm-started heads left 100% of target variance unexplained. Their shuffled gap
  was 0.000004 against a threshold of 0.02, and holes from their predictions were 0.50 against 0.55.
- **The linear ceiling also fell below the holes threshold:** it explains 16% and 7% of target
  variance, and holes read from its predictions are 0.54 and 0.53. By the table above: stop; the user
  decides.

**Why (post hoc, not fixed before the run):**
- **The context path was dead.** E's latents have a few large constant channels (means up to 285),
  and only 0.003% to 0.04% of their squared norm differs between maps. After the parameter-free layer
  norm, the predictor saw almost the same input for every map. It applies to EJ2 too (above).
- **With each latent channel standardised first,** the heads use the latents:
  - shuffled gap 0.26 at 7 x 7 and 0.038 at 8 x 8;
  - target variance explained 13% and 1.3%;
  - holes from the predictions still 0.52 and 0.50.
- **The 0.55 holes threshold was not reachable** through these targets. It was set against round 2's
  latent probe (0.65), which regresses holes directly, and even the linear ceiling reads 0.53 to 0.54
  here.
- **In calibrated units,** the share of the targets' hole-direction variance explained on held-out
  maps:
  - standardised heads: 0.107 at 7 x 7, 0.008 at 8 x 8;
  - linear ceiling: 0.121 and 0.061;
  - gate 3's heads: zero.
- **E's latents start with little per-cell hole content.** The heads reach most of the linear
  ceiling at 7 x 7 and little at 8 x 8, where 94 of the 1,486 training maps are.

**Budget:** 489 of the 2,400 approved GPU-seconds used.

## Amendment after gate 3 (2026-11-16)

The user approved running EJF with the fix below, on the post hoc numbers above, and raised the
allowance to 2,600 GPU-seconds. The original gate 3 stays recorded as failed.

**The fix: standardise the latents before the context projector.**
- **The map:** per latent position and channel, subtract the mean and divide by the standard
  deviation of E's latents over the training maps.
- **Fixed:** it is computed once in gate 3 and stored in the heads, so the arm applies the same map.
- **Where:** it comes before the parameter-free layer norm. Nothing else in the heads changes.
- **A consequence, stated before the run:** most channels' standard deviation is about 0.02, so this
  scales them up about fifty-fold. The term's gradient into the latents therefore concentrates on
  channels that now barely vary between maps. β sets the gradient's size, not its direction. This is
  the intended effect, since those channels are where map content can be added.

**Gate 3 rerun, on the saved features, from the project code:**
- **3a and 3b:** unchanged. The targets do not depend on the fix.
- **3c, amended:** the shuffled gap is at least 0.02 at 7 x 7 and 8 x 8, as before.
- **The holes-from-predictions threshold is dropped as a gate.** It was unreachable through these
  targets (the linear ceiling reads 0.53 to 0.54).
- **Its replacement, descriptive:** the share of the targets' hole-direction variance that the
  predictions explain, beside the linear ceiling's. The post hoc figures were 0.107 against 0.121 at
  7 x 7 and 0.008 against 0.061 at 8 x 8.
- **What the rerun checks:** that the project code reproduces the diagnosis. It is not a new test of
  holes, because the user decided on these numbers.
- **If the amended 3c fails,** stop and report.

**EJF:** as specified above, with these heads.
- **Time limit:** 35 minutes (2,100 s), so the job cannot pass the 2,111 GPU-seconds left in the
  allowance.
- **Expected:** about 1,900 s (EJ2's 1,871 plus the pilot rows).

**A second fix, the same day: how maps are keyed.**
- **The failure:** the first EJF job (7117889) failed after 135 GPU-seconds, at phase 1's first
  rows. A row's stored target belonged to another map.
- **The cause:** map IDs are unique within each fresh split, but in the released training split they
  repeat across levels: 174 of its 1,000 rows share an ID with another row.
- **Its reach:** `distinct_maps` kept one map per repeated ID, so the features job saved 1,486
  training maps instead of all of them. The stored targets were keyed the same way.
- **The fix:**
  - maps are keyed by split and image (`perception_jepa.map_key`);
  - the saved features were re-keyed on CPU (`scripts/rekey_features.py`). Each saved map is found
    again by its ID and layout, and the tensors are unchanged;
  - the arm computes targets for the training maps the features job dropped before any update
    (`add_targets`), with the same prompt pass, centring means and projection;
  - gate 3 was rerun on the re-keyed features. Its fit is unchanged, because the dropped maps were
    never in it.
- **For EJ2:** its gate's centring means and warm start used the same 1,486 maps. EJ2 computed its
  targets from the hidden states, so no row was trained against another map's target.
- **Budget:** 624 GPU-seconds used, leaving 1,976. EJF's time limit is set to that remainder
  (32 min 56 s).

## Result: EJF (job 7118300)

Evidence: [jepa-ejf-7118300](experiments/jepa-ejf-7118300/README.md). Gate 3's re-keyed heads (job
7118222); 88 merged training maps got targets at the start; β = 0.027.

- **Validity:** passed. The first-step answer loss equals EB2's.
- **Manipulation check: did not take.** The hole probe at layer 12 is 0.655 against EB2's 0.657
  (difference −0.002, interval −0.010 to 0.006; the margin was 0.05).
- **Primary: no reading on perception.** Descriptive: EJF 76 against EB2 78 of 400 (11 against 13,
  p = 0.84).
- **Secondary: no detectable difference** from E2: 76 against 84 (17 against 25, p = 0.28).
- **The perception count rose,** as for EJ2: steps into a hole beside a safe toward-goal move, 91
  against E2's 79.
- **Benchmark:** 318 against LIVR's 328 (p = 0.18), with round 2's trade (detour 16 against 9;
  no-detour 302 against 319).
- **The outcome table's check:**
  - the prediction did depend on the latents. On validation maps, the shuffled gap rose from 0.08 to
    0.15 and the unexplained share fell from 0.96 to 0.93;
  - on held-out maps at 8 x 8, the gap rose from 0.03 (E, gate 3's heads) to 0.10;
  - the targets' hole direction stayed at 0.11 at 7 x 7 and moved from 0.005 to 0.02 at 8 x 8.
- **Post hoc, the latents themselves:** a linear readout of the targets from the saved latents does
  no better for EJF than for EB2, which was trained without the term. Share unexplained: 0.832 and
  0.945 for EJF, against 0.824 and 0.917 for EB2. What improved was mainly the co-trained heads.
- **Across arms:** the layer-12 hole probe reads 0.649 to 0.664 for E, E2, EB2, EJ2 and EJF, under
  three pressures: the bottleneck, a learned JEPA and a fixed-target JEPA that took hold. Both JEPA
  arms raised the latents' participation ratio and the perception-failure count (90 and 91, against
  79 and 80) without moving the probe.

**Budget:** 2,618 of the 2,600 approved GPU-seconds, 18 over. The run had finished; Slurm killed the
job 18 s past its 32 min 56 s limit, during shutdown.

## Region latents: EJR

Proposed by the user on 2026-11-16 ("an adaptive number of latent tokens"), in the form recommended
after the CPU check below; approved the same day ("sgtm"). Allowance 2,400 GPU-seconds.

**Why (CPU, on saved latent states).** E's and EB2's four latents are copies of one another, not
slots:
- **One token reads holes as well as all four.** For E at 7 x 7, all four read 0.657 and token 0
  alone 0.689. EB2 and 8 x 8 look the same.
- **Their per-cell maps of hole information are 85% to 88% alike.** All four carry the same few cells,
  mostly the top-left corner, top row and left column.
- **So more tokens alone would most likely add copies.** Each token needs its own job.

**The design.**
- **Tokens:** one latent per 2 x 2 block of cells, in row-major block order. That gives 4 tokens at
  3 x 3 and 4 x 4, 9 at 5 x 5 and 6 x 6, and 16 at 7 x 7 and 8 x 8.
  - Tokens 0 to 3 are LIVR's own, so prompts at 3 x 3 and 4 x 4 are unchanged.
  - Tokens 4 to 15 are new. Each new token's embedding starts as a copy of token k mod 4's, plus
    Gaussian noise at the model's initializer range.
- **The objective:** for each token, a single linear readout (shared by all tokens and sizes)
  predicts its own block's fixed targets from its standardised layer-12 state. The targets are EJF's:
  E's layer-4 cell states, centred per position and projected by the frozen PCA.
  - Cells outside the map, in edge blocks at odd sizes, are masked out of the loss.
  - **Why the stored targets still apply:** the image tokens come before the latents, so the
    layer-4 cell states do not depend on how many latents follow.
- **Total:** answer loss + β · mean squared error over the tokens' cells. β comes from the
  equal-gradient rule.
- **Rows, phases, recipe and evaluation:** as EB2, from E. The readout is clipped on its own, as the
  JEPA heads were.

**At the start of the arm, in this order:**
1. **Before extending:** E's answer loss on 8 phase-1 rows at 3 x 3 or 4 x 4 and 8 at 5 x 5 or above,
   under the Stage 1 mask.
2. Extend the tokens.
3. **Again on the same rows.**
   - **Validity:** the 3 x 3 and 4 x 4 rows' final-layer states equal E's, within 1e-4 relative.
   - **Why states, not losses:** the embeddings are tied, so the 12 new rows also enter the output
     softmax, and the small rows' answer loss can move slightly. The stand-in model's 40-token
     vocabulary showed this; that loss change is reported.
   - **Descriptive:** the larger rows' change in answer loss, "insertion disturbance".
4. **One unmasked pass over 350 training maps:** all 201 at 7 x 7 and 8 x 8, 100 at 5 x 5 and
   6 x 6, and 49 smaller ones. It gives each token's per-channel standardisation (fixed afterwards)
   and a ridge warm start of the readout.
5. **The β rule** on phase 1's first 16 rows. Outside 0.01 to 100, it stops before training.

**What one arm can and cannot show.**
- EJR changes two things at once: more tokens and the region objective. A "took" cannot be split
  between them.
- **If it takes,** the follow-up is a control arm with 16 tokens and no objective.

**Validity (readings are withheld if any fails):** the start check (step 3), round 2's set checks,
and gate 3 (job 7118222), whose targets EJR reuses.

**Manipulation check, count-matched.** Each block's holes are read by ridge from a single token's
2,048 features:
- for EJR, the block's own token;
- for EB2, token 0 for every block, its most informative token in the check above.

Scored on held-out maps (balanced accuracy, mean of 7 x 7 and 8 x 8), fitted on pool maps.
- **Took:** EJR is at least 0.05 above EB2, and the 95% paired-bootstrap interval of the difference
  lies above zero.
- **Why not round 2's probe:** it reads all latents at once, and extra tokens that carry nothing
  lower it. Padding EB2's states with 12 noisy copies of its tokens took it from 0.657 to 0.651,
  0.591 and 0.548 at noise of 0.1, 0.5 and 1 times the across-map spread. Fixed-size readouts did no
  better: 256 principal components read chance, and the best single token per cell fell from 0.652
  to 0.587.
- **Choosing token 0 for EB2 favours EB2:** it was picked after seeing held-out scores.
- **Baseline, computed before the run:** EB2 scores 0.689 (0.699 at 7 x 7, 0.679 at 8 x 8), so
  "took" needs EJR at 0.739 or above. For reference: E 0.681, E2 0.694, EJF 0.675.

**Primary: EJR against EB2; secondary: EJR against E2.** These are the same tests and outcome table as
EJ2's.

**Descriptive:**
- round 2's all-tokens probe, with the padding bound above;
- the probe on tokens 0 to 3 alone;
- insertion disturbance;
- the readout's unexplained share and shuffled gap at the start and after each phase, on 32
  validation and 32 training maps;
- first mistakes, participation ratio, masked greedy, the harm check and the benchmark.

**Budget:** expected 2,050 to 2,150 GPU-seconds: EJF's 1,994, plus the start block and slightly longer
prompts. The Slurm limit is 39 minutes, so the limit plus Slurm's kill grace stays inside the 2,400
s allowance.

## Result: EJR (job 7120944)

Evidence: [region-ejr-7120944](experiments/region-ejr-7120944/README.md). 1,832 GPU-seconds of the 2,400
allowance.

- **The start block:**
  - the 3 x 3 and 4 x 4 rows' states were unchanged by the new tokens;
  - insertion disturbance +0.041;
  - 88 targets added;
  - β = 0.25.
- **Validity:** passed.
- **Manipulation check (count-matched): did not take.** Each block read from EJR's own token gives
  0.667, against EB2's token 0 at 0.689 (difference −0.021, interval −0.030 to −0.013; the margin was
  +0.05).
- **Primary: no reading on perception.** Descriptive: EJR 80 against EB2 78 of 400 (24 against 22,
  p = 0.88).
- **Secondary: no detectable difference** from E2: 80 against 84 (16 against 20, p = 0.62).
- **Descriptive:**
  - round 2's all-tokens probe rose to 0.677 from EB2's 0.657 (interval 0.013 to 0.028);
  - the linear ceiling to the targets improved (hole-direction share 0.20 and 0.14, against 0.16
    and 0.09);
  - the tokens barely specialised: between-token similarity 0.83 against EB2's 0.85 to 0.86, own
    block 0.44 against other blocks 0.38 at 7 x 7;
  - the perception count rose to 92, as in EJ2 (90) and EJF (91);
  - the harm check is no longer clear of E: 95 against 88, p = 0.19.
- **Reading:** giving each token a block did not break the copies. Together, the latents carry
  slightly more of the map, and routes did not change.

**Across the four perception arms** (EB2, EJ2, EJF, EJR):
- **The latents' hole content barely moved.** Round 2's probe ranges from 0.649 to 0.677.
- **None solves more held-out maps than E2** (84).
- **Each objective that acted on the latents raised the perception-failure count** (90 to 92,
  against 79 and 80).

## Global + region latents: EGR

Proposed on 2026-11-16 after EJR, as the one design in which the latents must differ, and approved
by the user the same day ("global + region latents make sense"). Allowance 2,100 GPU-seconds;
time limit 34 minutes.

**Why.**
- **EJR's sixteen tokens stayed near-copies** (similarity 0.83), with or without an objective
  asking them to differ.
- **Nothing in the architecture separates the latents.** Every latent sits after the image and sees
  all of it.
- **EGR makes the separation structural instead.** A null result then rules something out: whether
  holding the map block by block in the latents changes routes.

**The design.**
- **Tokens:**
  - E's four global latents come first, unchanged.
  - Then one new region latent per 2 x 2 block (`<|livr_region_k|>`; 4, 9 or 16 by size). Each
    starts as a copy of global token k mod 4 plus noise at the initializer range.
  - The prompt up to the global latents is token-identical to E's.
- **The mask, in both stages:** a region latent attends only to the prompt before the image, its own
  block's image tokens and itself.
  - **A block's image tokens:** those with any overlap with the block's cells, so tokens on a
    boundary are shared.
  - **What it cannot see:** the text after the image, the global latents and the other region latents.
    All of them have seen the whole image.
  - **Everything else is unchanged:** Stage 1's bottleneck (the answer sees no image tokens, only
    latents) and Stage 2's full view of the image.
- **The limit of the separation:** inside the language model, image tokens attend causally to earlier
  image tokens. A block's image tokens can carry some context from blocks above and to the left, so
  the separation is at the latents, not perfect.
- **No objective.** The language model is trained on the answer loss alone, with EB2's rows, phases
  and recipe, so there is no β.
  - A linear readout from each region latent to its block's fixed targets trains on detached states,
    as a monitor only.
- **Not run:** the 3 x 3 to 6 x 6 benchmark, to save GPU time. Every earlier arm showed the same
  trade against LIVR.

**What one arm can and cannot show.** EGR adds tokens and the mask together. EJR (extra tokens, no
mask, with an objective) is the nearest comparison.

**Start block:** E's answer loss and final-layer states on 16 phase-1 rows (8 at 3 x 3 or 4 x 4, 8 at
5 x 5 and above), then the extension, then again.
- **Validity:** the final-layer states over the shared prefix, through the global latents, equal E's
  on all 16 rows (1e-4 relative).
- **Descriptive:** insertion disturbance, the mean change in answer loss over all 16 rows.

**Readings.**
- **Validity:** the start check, round 2's set checks, and gate 3 (job 7118222), whose targets the
  monitor uses.
- **Manipulation check: "took" needs both of these.**
  - The count-matched block probe: each block read from its own region latent, at least 0.05 above
    EB2's token 0 (0.689), with the 95% interval above zero. The mask makes this likely by
    construction.
  - The region latents actually separate: the mean between-token similarity of their per-cell hole
    maps is below 0.6 (EJR 0.83, EB2 0.85 to 0.86). At or above 0.6, the image tokens' causal mixing
    dominated and the mask did not separate the latents.
- **Primary: EGR against EB2, unmasked greedy;** **secondary: EGR against E2, unmasked.** These are as
  before.
- **Discriminating: EGR against EB2 under masked greedy** (only the latents see the image; EB2 73 of
  400), exact McNemar.

| Outcome, given "took" | Reading |
| --- | --- |
| Unmasked greedy improves | The latents' perception limited routes |
| Masked improves, unmasked flat | The latents help only when the direct path is cut; the direct path already supplies what routes use |
| Both flat | Holding the map block by block in the latents does not change decisions, even when it is the only path. The limit is downstream, and the latent-perception line closes |
| Did not take | The mask did not separate the latents (or the probe failed); recorded |

**Descriptive:**
- round 2's all-tokens probe;
- the global latents alone (tokens 0 to 3) against EB2's 0.657, which shows whether the globals
  changed;
- the region latents alone;
- the monitor readout's unexplained share and shuffled gap;
- insertion disturbance;
- first mistakes, participation ratio and the harm check on round 1's held-out maps.

**Budget:** expected 1,850 to 2,000 GPU-seconds. That is EJR's 1,832, plus an explicit mask in
Stage 2 (which loses the fast causal path), less the benchmark (about 130 s) and the β pilot.

## Result: EGR (job 7122822)

Evidence: [region-egr-7122822](experiments/region-egr-7122822/README.md). 1,781 GPU-seconds of the
2,100 allowance.

- **The start block:** E's prefix states were unchanged on all 16 checked rows. The block latents
  raised the answer loss by 0.004 (small maps) and 0.037 (larger maps) before training; training
  recovered.
- **Validity:** passed.
- **Manipulation check: took.**
  - Count-matched block probe: 0.934 against EB2's 0.689 (difference +0.246, interval 0.237 to
    0.254).
  - Separation: mean similarity between block latents −0.03, against a threshold of 0.6. Own block
    0.80 and 0.73, other blocks 0.15 and 0.11.
- **Primary: no detectable difference.** 75 against EB2's 78 (17 against 20, p = 0.74).
- **Secondary: no detectable difference.** 75 against E2's 84 (19 against 28, p = 0.24).
- **Discriminating, masked greedy: no detectable difference.** 68 against EB2's 73 (12 against 17,
  p = 0.46).
- **Reading, by the outcome table ("took, both flat"):**
  - Holding the map block by block in the latents does not change decisions, even when the latents
    are the only path to the image.
  - The limit is downstream of the latents, and the latent-perception line closes.
  - **Caveat:** this holds for EB2's recipe of 375 steps from E.
- **Descriptive:**
  - the global latents are unchanged (probe 0.658 against EB2's 0.657);
  - the all-latents probe is 0.708;
  - the perception-failure count is 89 (E2 79, EB2 80);
  - the harm check is no longer clear of E (90 against 88, p = 0.82);
  - the monitor readout broke down as the block latents' states drifted from its fixed
    standardisation.

**Across the five perception arms** (EB2, EJ2, EJF, EJR, EGR):
- **Only EGR put the map into the latents,** by construction.
- **None changed routes.** No arm solves more held-out maps than E2 (84), unmasked or masked.

## EGR: does the answer read the block latents? (the swap test)

Approved by the user on 2026-11-16 ("yes this sounds good to me"), about 7 GPU-minutes. It runs on
EGR's saved weights. The question: EGR's latents hold the map, yet routes did not change. Does the
answer ignore the block latents, or read them without its moves changing?

**The test.** Masked greedy, so the answer sees the image only through the latents, on the first 50
held-out maps at 7 x 7 and the first 50 at 8 x 8. Four conditions:
- **Own:** EGR as trained.
- **Block swap:** every layer's states at the block latents come from a donor map of the same size
  with a different layout (the latent battery's donors and patching).
- **Block mean:** the block latents' states are replaced by their mean over the 50 maps of that size,
  at every layer. This removes the map-specific content.
- **Global swap,** for reference: the four global latents come from the donor.

**Validity:**
- Own must reproduce EGR's recorded masked routes on at least 95% of the maps.
- Re-inserting a map's own block latents must reproduce its route on the first four maps.

**Readings (per condition, against own):** the share of routes unchanged; maps solved; and for the
swaps, the share of routes that are exactly the donor's.

| Outcome | Reading | Next |
| --- | --- | --- |
| Block mean leaves at least 90% of routes unchanged, and block swap gives the donor's route on at most 10% | The answer ignores the block latents | Make it rely on them: train under the bottleneck with the global latents dropped out |
| Block swap or block mean changes more than 30% of routes | The answer reads the block latents, but their content does not shape its moves | Make the reward depend on it: expert iteration or RL from EGR |
| In between | Mixed | Report; the user decides |

**Budget:** allowance 480 GPU-seconds, time limit 7 minutes. Expected about 330 s: one model load and
about 400 masked answers.

## Result: the swap test (job 7123991)

Evidence: [region-swap-7123991](experiments/region-swap-7123991/README.md). 391 GPU-seconds of the 480
allowance.

- **Validity:** passed. Own routes reproduced EGR's recorded masked routes on 100 of 100 maps;
  re-inserting a map's own block latents reproduced 4 of 4.
- **The reading: the answer ignores the block latents.**
  - Block latents replaced by their mean: 93 of 100 routes unchanged.
  - Block latents swapped from a donor: 92 unchanged, and none becomes the donor's route.
  - Solved: 17 own, 18 swapped, 17 mean.
- **The route is set by the global latents.** With a donor's globals, 92 of 100 routes are the
  donor's, and none is solved.
- **Interpretation:** the global latents precede the block latents, so they never see them. The
  answer copies the route already formed in the globals. By the table above, the next step is to train
  under the bottleneck with the global latents dropped out.

## Global dropout from EGR: EGD

Proposed after the swap test and approved by the user on 2026-11-16 ("lets start with what you just
proposed"). Allowance 2,100 GPU-seconds, time limit 34 minutes.

**Why.**
- EGR's answer ignores the block latents and copies the route formed in the four global latents.
- EGD makes the block latents the answer's only map, by hiding the global latents from it.

**The design.**
- **Start:** EGR's final weights (job 7122822). The block latents already exist and were trained
  under the mask.
- **Hiding the global latents:** no token after the latents (the header, the answer) may attend to
  the global latents. The global latents themselves are unchanged, and the block mask is as in EGR.
- **Training:** round 2's rows and phases again, on the answer loss alone.
  - **Phase 1 (Stage 1 bottleneck):** the globals are hidden on every row. The answer's only
    access to the image is the block latents.
  - **Phase 2 (unmasked):** the globals are hidden on every other row.
- **What one arm cannot show:** EGD has one more pass over the rows than EGR. A control with that
  pass and the globals shown is not in the budget.
- **Evaluation, on the 400 held-out maps:**
  - masked greedy with the globals hidden (only the block latents carry the image);
  - masked greedy with the globals shown;
  - unmasked greedy;
  - the latent states;
  - the swap test (own, block swap, block mean) on the first 50 maps per size, masked with the
    globals hidden.
  - Not run, to fit the budget: round 1's held-out maps, the test detour maps, the seen maps and the
    benchmark.

**Readings.**
- **Validity:** round 2's set checks; the swap test's reproduction check (own routes equal EGD's
  recorded masked, globals-hidden routes on at least 95% of the maps) and identity check.
- **Does the answer use the block latents?** It does if, in the swap test, block mean changes more
  than 30% of routes or block swap gives the donor's route on at least 30%.
- **Comparisons,** exact McNemar on the 400 held-out maps:
  - masked, globals shown: EGD against EGR (68) and EB2 (73);
  - unmasked: EGD against EGR (75), EB2 (78) and E2 (84).
- **Descriptive:** masked with the globals hidden, against EGR's masked 68; the block probe and the
  latents' separation; first mistakes.

| Outcome | Reading |
| --- | --- |
| Uses the block latents, and masked or unmasked greedy improves (p < 0.05) on EGR | Making the answer use the latents helps routes |
| Uses the block latents, no improvement | Reading the map from the latents does not fix the policy; the limit is the decision process |
| Does not use the block latents | Hiding the globals did not force reliance on the block latents |

## Result: EGD (jobs 7124258 and 7125522)

Evidence: [region-egd-7124258](experiments/region-egd-7124258/README.md). 2,063 GPU-seconds of the
2,100 allowance.
- **The job hit its time limit at the start of the swap test.** A slow model load cost about 2.5
  minutes. Training, the evaluations, the states and the weights are complete.
- **Masked, globals shown:** 78, against EGR's 68 (p = 0.11) and EB2's 73 (p = 0.49).
- **Unmasked:** 78, against EGR's 75 (p = 0.75), EB2's 78 and E2's 84 (p = 0.44).
- **Masked, globals hidden** (the block latents are the only map): 63.
- **The swap test (job 7124258 timed out before it; run as job 7125522, 436 GPU-seconds):** the answer
  uses the block latents. With a donor's, 98 of 100 routes are the donor's. With them averaged, every
  route changes.
- **The reading, by the table: "uses the block latents, no improvement".**
  - Reading the map from the latents does not fix the policy; the limit is the decision process.
  - **Post hoc:** even with routes built from the block latents, the model steps into the holes they
    flag, at their general rate (92%). It reads them for where to go, not for what to avoid.
- **A wording error found in the swap test's table:** its "reads the block latents, but their content
  does not shape its moves" should have read only "reads the block latents". The routes do follow
  their content.

## Position tracking on EGD (the state probe)

Proposed after EGD and approved by the user on 2026-11-16 ("do the 10-15 minute probe on EGD"). No
training. Time limit 30 minutes; expected 15 to 20 H200-minutes.

**Why.**
- EGD's routes come from its block latents, and those latents flag 92% of holes. Yet it steps into
  holes they flag.
- To avoid a hole, the answer must know which cell it is in, then check the cell next to it.
- On LIVR (job 6887458), the planning probe's state probe found the agent's cell decoded less often
  just before fatal moves (16–17%, against 26–40% expected). The count of moves from the start was
  intact, but the start cell was only partly present. That probe was weak (39–49% on solved steps).
- Does EGD know where it is when it steps into a hole?

**The design.** The state probe of [the planning probe spec](PLANNING_PROBE_SPEC.md#2-state-tracking)
unchanged, on EGD's final weights (job 7124258), except:
- **Maps:** round 2's 400 held-out maps (greedy routes) and 500 pool maps, whose released answers are
  teacher-forced to fit the probes. All are 7 x 7 or 8 x 8. LIVR's run used the 400 test maps and
  1,000 training maps, at 3 x 3 to 6 x 6.
- **Two answer modes,** each with its own routes and probes:
  - unmasked, with the globals shown (EGD's 78 of 400);
  - masked with the globals hidden (EGD's 63), where the block latents are the answer's only map.
- **Not run:** the headroom test and the fresh set.

**Readings, per mode.**
- **Validity:** the routes equal EGD's recorded routes in that mode (`greedy.jsonl`,
  `masked_hidden.jsonl`) on at least 95% of the maps.
- **Probe strength:** position accuracy on solved test steps of at least 50%. Below that, the reading
  is "inconclusive: the position probe is too weak".
- **S1,** as in the planning probe spec:
  - tracking contributes: p < 0.05 and observed accuracy at least 15 points below expected;
  - tracking intact: less than 10 points below;
  - otherwise inconclusive.
- **Descriptive:** S2 and S3. Then, post hoc on CPU, the planning probe's exploratory checks
  ([scripts/analyze_planning_probe.py](../scripts/analyze_planning_probe.py)): the offset from the
  start, the start cell at the first move, and the absolute cell with a small MLP.

| Masked, globals hidden | Reading |
| --- | --- |
| Tracking contributes | EGD loses its place before fatal moves. Its latents hold the holes, but the answer does not know which cell it is leaving. The lever is position tracking. |
| Tracking intact | EGD knows where it is and still steps into a hole its latents flag. The failure is in using the map at that cell, not in tracking. |
| Inconclusive, or the probe is too weak | No reading on tracking. |

- **The unmasked mode** is reported beside it, and set against LIVR's result.
- **Caveat:** a probe on one token's state shows what is resident there, not what attention fetches
  when it is needed.

## Result: position tracking on EGD (job 7128341)

Evidence: [region-tracking-7128341](experiments/region-tracking-7128341/README.md). 1,019
GPU-seconds, inside the 30-minute limit.
- **Validity: passed.** The routes equal EGD's recorded routes on 400 of 400 maps in both modes (78
  and 63 solved).
- **Both modes read "inconclusive: the position probe is too weak".**
  - On solved test steps, the position probe reads the agent's cell 23% of the time unmasked and
    16% masked with the globals hidden. The floor was 50%.
- **Descriptive only:**
  - S1 shows no gap: fatal steps decode 17% of the time, against 19% expected unmasked and 14%
    masked.
  - **Post hoc:** the count of moves from the start is present at 93–95% of fatal steps. The
    absolute cell is barely present at any step (a small MLP reads it at most 21% of the time).
  - The state that writes a fatal move does not mark the hole ahead more than a safe cell.
- **Missed in the design:** [step grounding](STEP_GROUNDING_SPEC.md#result) (job 6926184) had
  already tested tracking on LIVR.
  - It trained the cell into the model's states, readable before 76–78% of fatal moves.
  - Walks into holes did not fall.
  - With EGD, a model that holds the map, or that holds its cell, still steps into holes. The
    evidence points at how a move is chosen from a state that has the information.

## Global + patch latents, for images without a grid (built, not run)

Requested by the user on 2026-11-16 ("let's make the patch variation for the perception images").
That followed a discussion of MMVP, V* and BLINK, the perception benchmarks that LIVR and Scaffolding
Minds both report.

**What it is:** EGD's design without a map grid.
- **The latents:** LIVR's four global latents, then one patch latent per tile of each image.
  - Each image's merged token grid is cut into 4 x 4 tiles of near-equal height and width.
  - A prompt may carry up to 4 images, BLINK's most. That is 64 patch tokens,
    `<|livr_patch_0|>` to `<|livr_patch_63|>`, image-major.
- **One token per (image, tile) slot:** in Stage 1 only a latent token's first occurrence may read
  the image, so one token cannot serve two images.
- **The mask, in both stages:** a patch latent sees its tile's image tokens, itself and text. It
  sees no other image, tile or latent.
- **Which text, a setting (`text`):**
  - "all", the default: every prompt token before it that is neither an image token nor a latent.
    It sees the question.
  - "prefix": only the text before the first image, as EGR's block latents.
  - On FrozenLake the two differ only by the fixed sentence after the map; on questions, by the
    question.
- **The global latents** are LIVR's. `livr_hide_globals` hides them from the answer, as in EGD.
- **Empty tiles:** a merged grid with fewer than 4 rows or columns leaves some tiles empty. Their
  latents see only text.
- **On FrozenLake:**
  - Every map gets 16 patch latents, whatever its size.
  - At 8 x 8, each tile is 2 x 2 cells, because the map fills the image. That matches EGR's blocks.
  - At other sizes, tiles cut across cells.

**Code:**
- `region_latents.extend_latents(..., layout="global+patch", mask=True, grid=4, max_images=4, text="all")`,
  with `patch_latents`, `patch_tiles`, `image_runs` and `restrict_patches`.
- `model.LivrQwen.forward` applies the patch mask without a grid size.
- `serialization.encode` takes rows with several images (`image_inputs`, paths relative to the
  split's folder). It inserts the globals, then one set of patch latents per image.
- **Single-image prompts are unchanged:** 60 FrozenLake prompts encode identically before and after.
  They cover every size, training and inference, and LIVR's, the region and the hidden-globals latents.

**Tested on CPU:**
- tiles and image runs;
- the mask row by row, and zero attention across tiles in both stages;
- generation, hidden globals and empty tiles;
- a two-image prompt through the real processor, with a save and load round trip.

**Not built:**
- Training and evaluation data for MMVP and BLINK: loaders, prompts and answer scoring.

### V* and the arms from base (built 2026-11-18)

Requested by the user on 2026-11-18: V* first, as the most relevant of the three, on `main`'s code.

**V\* evaluation** (`spara.vstar`, `spara vstar`, `scripts/pace/vstar.sbatch`):
- **Data:** the 191 released questions (115 attribute, 76 relative position) from
  `craigwu/vstar_bench` at a pinned revision, each file checked against the hub's hash.
- **Answers:** greedy, 16 new tokens. The option letter is read from the reply. A reply that is
  exactly one option's words also counts, and the summary reports how many were read that way.
  Anything else is wrong.
- **Pixel limit:** the image is resized once, before the processor, to a stated number of image
  tokens. Native size is a median of about 4,300 tokens and at most 10,577, under the processor's
  16,384. The evaluation is run at native size and at 1,280 tokens.
- **Prompt formats:** "chat", the processor's own template, for the base model; "livr", this
  project's serialization, for trained arms and as a second reading of the base model.

**Training data** (`spara.viscot`): Visual-CoT with answers only, as LIVR's Table 4 uses it.
- Each released row keeps its question and final answer. The bounding-box turn and the cropped
  second image are dropped.
- A seeded sample of the released list (404,120 rows): 500 validation rows, then 10,000 training
  rows on other images. LIVR trained on the whole set; this is a first, smaller comparison.
- Only the sample's images are taken from the 139 GB archive, which is read as one stream and not
  stored. Each image is resized to at most 1,280 image tokens and saved as JPEG.

**Arms** (`spara.perception`, `spara perception`, `scripts/pace/perception.sbatch`), each from base
Qwen2.5-VL-3B with `configs/livr.json`'s optimizer:

| Arm | Latents | Segment 1 (2 epochs) | Segment 2 (3 epochs) |
| --- | --- | --- | --- |
| `sft` | none | plain | plain |
| `livr` | LIVR's four | Stage 1 mask | unmasked |
| `patch` | four globals + 16 patch latents, globals hidden from the answer | Stage 1 mask + tile mask | tile mask |

- Every arm sees the same rows in the same order for the same steps, with the schedule restarted
  between segments. The 2:3 split follows LIVR's multi-task recipe.
- After training: exact-match accuracy on the 500 validation rows (a check that training took),
  then V* at both pixel limits in the "livr" format.
- **Training images are at most 1,280 tokens; V* at native size is larger.** The 1,280-token V*
  reading matches training; the native reading tests transfer to larger images.

**Not built:** a gate or decision rule for the comparison, and other benchmarks' loaders.

## Budget

From 2026-11-16, after EGD, the user set no per-run GPU allowance: each job gets a time limit with
headroom over its expected runtime, short enough to avoid waiting in the priority queue. GPU-seconds
are still reported for every job.

| Job | Estimate | Main costs |
| --- | --- | --- |
| Gate | 20–25 H200-minutes | Two model loads; about 2,300 prompt passes; about 800 greedy answers in the swaps |
| EJ2 | 30–35 H200-minutes | EB2 took 26 minutes; this adds all hidden states, the heads and about 200 monitor passes |

That is about 55 GPU-minutes in all. The gate writes small files only. EJ2 saves its final weights
(15 GB); scratch must be rechecked first.

**EJF** (approved 2026-11-16, allowance 2,400 GPU-seconds):

| Job | Estimate | Main costs |
| --- | --- | --- |
| Features | About 6 H200-minutes | One model load; 2,185 prompt passes (the EJ2 gate's dump took 253 s) |
| Gate 3 | No GPU | An `ice-cpu` job |
| EJF | 31–33 H200-minutes | As EJ2 (1,871 s), plus 16 pilot rows |

EJ2's weights were deleted at the user's request, which leaves room for EJF's 15 GB.

## Running

```bash
python -m spara.cli perception-jepa --stage gate --checkpoint /path/to/E/stage2/final \
  --e2-checkpoint /path/to/E2/phase2/final --round1-routes /path/to/round1-search/routes.jsonl \
  --search-run /path/to/round2-search --data-root data/vsp --output /path/to/gate
python -m spara.cli perception-jepa --stage train --checkpoint /path/to/E/stage2/final --gate-run /path/to/gate \
  --round1-routes /path/to/round1-search/routes.jsonl --search-run /path/to/round2-search \
  --data-root data/vsp --output /path/to/ej2
```

On PACE: [scripts/pace/perception_jepa.sbatch](../scripts/pace/perception_jepa.sbatch) with
`SPARA_STAGE=gate`, then `SPARA_STAGE=train SPARA_GATE_RUN=<gate output>`. `perception_jepa.compare`
applies the readings on CPU.

The retry with fixed targets:

```bash
python -m spara.cli perception-jepa --stage features --settings configs/perception_jepa_fixed.json \
  --checkpoint /path/to/E/stage2/final --round1-routes /path/to/round1-search/routes.jsonl \
  --search-run /path/to/round2-search --data-root data/vsp --output /path/to/features
python scripts/fixed_target_gate.py /path/to/features /path/to/gate3 --settings configs/perception_jepa_fixed.json
python -m spara.cli perception-jepa --stage train --checkpoint /path/to/E/stage2/final --gate-run /path/to/gate3 \
  --round1-routes /path/to/round1-search/routes.jsonl --search-run /path/to/round2-search \
  --data-root data/vsp --output /path/to/ejf
```

On PACE:
- **Features:** `SPARA_STAGE=features` with perception_jepa.sbatch.
- **Gate 3:** `SPARA_FEATURES_RUN=<features output>` with
  [scripts/pace/jepa_fixed_gate.sbatch](../scripts/pace/jepa_fixed_gate.sbatch), on `ice-cpu`.
- **EJF:** `SPARA_STAGE=train-fixed SPARA_GATE_RUN=<gate 3 output>`. It refuses a gate 3 that did
  not pass.
- **Readings:** `compare(..., arm="ejf")`.

## Assumptions (not tuned)

- Context layer 12; target-layer candidates 4, 8, 12 and 16; λ = 0.02.
- Head sizes, learning rates, dropout and queue sizes; the warm start's length.
- The equal-gradient rule for β. Its consequence: the language model's clipping norm now covers two
  terms of about equal size, so the answer term gets roughly half EB2's effective step. This is part
  of the design, recorded here before any result rather than read into a "harms" outcome afterwards.
- Pooling by the overlap of cells and merged image tokens, with the map filling the image, as the
  latent battery.

## Options, not in scope

- **JEPA-only rows** on the 299 pool maps and 127 frontier maps the search did not solve: the answer
  loss masked, so perception learns from maps the policy cannot yet solve.
- **More latent tokens** for larger maps.
- **Perception benchmarks** (BLINK, MMVP, V*) and other problems. The latents they would use are
  built; see [global + patch latents](#global--patch-latents-for-images-without-a-grid-built-not-run).
