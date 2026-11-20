# Latent distillation: search the latents, then train the model to produce the ones that worked

Status, 2026-11-19: **readings final; implemented; submitted.** Written after the offline checks in
[latent-learnability-offline](experiments/latent-learnability-offline/README.md), then critiqued
against code and data and adjusted (section "Critique and changes", at the end). The readings below
were fixed before any job was submitted.

## Why

**The midterm's main test is SPARA (latent changes learned with RL) against text-only RL.** Three
latent arms have run from LIVR against B′ (text-only RL, 92 and 96 of 240 held-out detour maps):

| Arm | Result on the 240 maps | What it showed |
| --- | --- | --- |
| C′: learned Gaussian heads at layer 16 | 77, 86: worse (p = 0.0016) | Mechanically broken: the latent loss swamped the gradient, and the heads could not move |
| C″: fixed noise at layer 24, c = 0.25, no heads | 98, 98: no detectable difference (p = 0.32) | Its extra signal was half rescues, half breaks |
| C‴ (learned heads on C″'s settings) | Not run | The offline checks show policy-gradient heads cannot learn here (below) |

**The gap is real; the mechanism is the limit.** Of LIVR's 170 greedy failures among the pilot
maps, 121 are never solved by 64 text samples. Random latent pushes at layer 24 solve 46 of
those 121. But only about 1 try in 40 succeeds, and the push that works is specific to the map:
no shared direction predicts success (AUC 0.50), and a per-map direction barely carries over
between rollouts (0.52). Heads trained by the policy gradient either do not move or go to their
bounds, and never learn even the one real signal (less noise on easy maps).

**So turn rare successes into targets, as expert iteration did for text.** Search the latents per
map with the verifier, keep the pushes that work, and train the model to produce those latent
states itself. This is the proposal's step 2 ("find the latent the model could use … distill it
back"), done without labels. The control trains on the same found routes without the latent
target, so the comparison isolates what the latent states add beyond the routes.

**What could go wrong, and the caveat.** Latent search reaches only 7 maps that final-verdict
(text) search does not (117 against 110 of 170), so most of what latents unlock is reachable by
text too. The question is whether supervising the latent state, not just the route, improves
held-out success. A null is possible and would be informative.

## The question and the arms

Both arms start from LIVR (`baseline-6253300/stage2/final`) and train on exactly the same rows in
the same order per seed.

| Arm | Trains on | Extra loss |
| --- | --- | --- |
| **S** (search routes) | The released 1,000 maps plus 2,000 rows drawn in turn from the maps the latent search solved, with the found route as the answer (expert iteration's composition) | None |
| **D** (distilled latents) | The same rows | On rows from found maps: pull the model's own layer-24 latent states toward the latent state that produced the success |

At test time neither arm gets any push: the model answers from its own latents, as LIVR does.

## Design

### 1. The latent search (one GPU job)

- **Maps:** the first 500 maps of `fresh/detour-pool`, the maps expert iteration's E searched, so
  E is available as context.
- **Policy:** LIVR, greedy decoding, `max_new_tokens` 128.
- **Per map:**
  1. Greedy with no push. If it succeeds, the map is found with a push of zero: its target is
     LIVR's own latent state.
  2. Otherwise up to 64 pushes in ascending size: 32 draws at scale 1, then 16 at scale 2, then 16
     at scale 4, in batches of 16 (one batched greedy generation per batch). Stop at the first
     success. Smaller pushes first, so targets stay as close to the model's own latents as possible.
     No scale 8 (see "Critique and changes").
- **Push:** δ = scale · Σⱼ uⱼ wⱼ εⱼ per slot, ε ~ N(0, 1), in C″'s confirmed basis
  (`latent-diag-sweep-L16-24-7148362/bases.pt`, layer 24, family `pca`, sha256
  `c848988a0ce8989af7191aaf18c5a5d5a53c036599632bd335d49f23ebd8418a`, checked at load), drawn
  from a generator seeded 42. Applied to the output of decoder layer 24 at the four latent
  positions (`LatentAdjuster.offset`).
- **Recorded per map:** row, solved, tries, scale, ε (4 × 16), route, answer text
  (`answer_text(route)`, E's `\\boxed{UP, RIGHT, …}` format, checked against the evaluator), ‖δ‖/‖z‖
  per slot, and in a tensor file LIVR's latent state z (4 × 2048, from a separate one-map prefill
  under the same bf16 autocast) and δ; the target is a* = z + δ.
- **Replay check:** every map found by a push is decoded again at once, alone (batch of one), with
  its recorded push. Maps whose route does not match are dropped and counted. The search is valid
  if at least 95% of pushed maps reproduce (batch-of-16 against batch-of-one bf16 kernels can flip
  near-tie greedy tokens; 99% was not grounded in any measurement).

### 2. Training (one job per arm and seed: four jobs)

- **Recipe (expert iteration's, `configs/expert_iteration.json`):** stage 2 (normal attention),
  one epoch over 3,000 rows (`extra_examples` 2,000), gradient accumulation 8 → 375 optimizer
  steps, learning rate 1e-5, cosine schedule with 5% warmup, max gradient norm 1.0.
- **Rows:** `expert_iteration.training_rows` with the found routes, so both arms get identical
  rows; the run records a sha256 of the row list, which must match between arms.
- **Seeds 42 and 43:** the data order (`epoch_order`) per seed; one search serves all four jobs.
- **Low memory:** a 48 GB Blackwell slice needs AdamW on the CPU (`runtime.offloaded_optimizer`,
  the same update; B′ and C″ used it for RL), via `train_stage(..., offload_optimizer=True)`, in
  both arms.
- **D's latent loss**, on rows from found maps only (released rows contribute nothing; the target
  is looked up by the row's `map_id`, not its position in the row list):
  - L_lat = mean over the four slots s of ‖P_s (h_s − a*_s)‖ / √E_s, where h is the model's own
    output of layer 24 at the latent positions in this forward pass (`out.hidden_states[24]`, no
    push), a* = z + δ the recorded target, P_s the projection on slot s's 16 basis directions (the
    subspace every push lives in), and E_s the expected ‖δ_s‖² of a scale-1 push: the distance to
    the target in units of a scale-1 push. At the start it is about the push scale on pushed maps
    and 0 on maps LIVR solved unpushed (their target is LIVR's own state, so for them the term
    anchors the latents). **Not squared**, so its gradient does not grow with the error and the
    calibrated λ keeps its meaning as the latents drift (smoke finding, below).
  - **Not cosine** (the draft's choice): see "Critique and changes".
  - **Weight λ, calibrated once, before training:** on the first 8 maps found by a push (pool
    order), compute the gradient of a training step (the mean over the 8 rows, one backward per row,
    before clipping) of the route loss and of L_lat separately, and set
    λ = 0.5 × ‖∇CE‖ / ‖∇L_lat‖. Record both norms and λ; λ is then fixed for the run. This keeps
    the latent term from swamping the update, which is what broke C′ (its latent term made the
    gradient about 200 times larger). 0.5 is a judgment: large enough to move the latents, below
    the route loss; in a real step only about a third of rows are pushed maps, so the latent share
    of the update is smaller still.
  - Plugged in through `train_stage`'s auxiliary-loss interface (`loss(hidden_states, index, batch)`,
    `weight`, `released_only = False`, `head = None`).
- **Logged per step:** route loss and gradient norm on both; on D also L_lat over all rows, over
  target rows, and the fraction of the push reproduced.

### 3. Evaluation (in each training run)

As E: greedy answers on round 1's 240 held-out detour maps (`fresh/detour-heldout` and the 40 test
detour maps), the 400 held-out 7 × 7 and 8 × 8 maps (`fresh/detour-heldout-78`), the first 98 pool
maps (descriptive), and the reference benchmark (400 test maps, one sampled answer each, seeded with
the run's seed). Greedy answers keep their decoded text for the CPU rescore. No latent push at test
time. LIVR's greedy answers on the same maps are the fair pilot's (`fair-pilot-7142592/greedy-livr.jsonl`);
its benchmark is `livr-benchmark-7145429`.

## Readings, fixed before the runs

**Primary: D against S on the 240 held-out detour maps.** Greedy successes per map summed over
the two seeds; exact sign test over maps where the sums differ.
- **"The latent target helps":** D ahead, p < 0.05, and D's count above S's in each seed.
- **"The latent target hurts":** S ahead, p < 0.05, and S's count above D's in each seed.
- **"No detectable difference":** otherwise.

**Validity (readings withheld if any fails):**
- the basis sha256 matches;
- at least 95% of pushed maps reproduce on replay;
- both arms' row lists have the same sha256 per seed;
- all four runs complete their 375 steps;
- the held-out sets share no layout with the pool (`check_sets`);
- every evaluation answer matches a CPU rescore with the reference evaluator;
- D's λ is finite and recorded, and L_lat is zero on every released row (unit test).

**Secondary:**
- D against S on the 400 transfer maps (the same test);
- the benchmark: each arm against LIVR (328), split into detour and no-detour maps (exact
  McNemar), and D against S;
- each arm against LIVR on the 240 maps.

**Diagnostics (descriptive):**
- **Did D learn the targets?** On every kept map, from one prefill before and after training: L_lat
  and the fraction of the push reproduced, (h − z)·δ / ‖δ‖² (0 for LIVR, 1 if learned), for D and
  for S (S is the baseline drift). If D's fraction does not rise above S's, the model could not
  produce the targets, and a null primary means "not learnable", not "not useful".
- **Search yield:** maps found with no push and at each scale; tries per found map; how far the
  targets are from LIVR's own latents (‖δ‖ relative to ‖z‖).
- **Context, not controlled:** E (final-verdict search, the same 500 maps, H200 stack) reached 88
  of 240 and 54 of 400.

**What it can claim:**
- **If D is better:** supervising LIVR's latent states with pushes found by search improves
  held-out route success beyond training on the same routes. That is a positive answer to the
  midterm's question through distillation rather than RL.
- **If no difference:** in this recipe, what the latents unlock is carried by the routes alone,
  or the model cannot learn to produce the targets (the push-fraction diagnostic separates the two).
- **Power:** two seeds on 240 maps detect a gap of about 7 points, as for B′ against C″.

## Risks carried over from C, C′ and C″

| Risk | Where it bit | Guard here |
| --- | --- | --- |
| A latent loss swamps the text gradient | C′ (about 200×) | A per-slot normalised loss; λ calibrated to half the route loss's gradient norm; norms recorded |
| Heads cannot learn from sparse rewards | C′, the offline simulation | No heads: the targets are supervised, one per found map |
| Replay changes what the update sees | C′ (absolute replay) | No replay in training; the targets are fixed states, and the search checks its pushes reproduce |
| Noise breaks maps the policy already solves | C″ (half its signal) | No noise in training or at test |
| Pushes encode "which map", not "which plan" | C′'s basis (90–99.9% map content) | The test is held-out success; the push-fraction diagnostic shows whether D's latents moved toward the targets |
| Large pushes give targets far from the model's states | New | Smallest scale first; no scale 8; target distances reported |
| Memory on a 48 GB slice | E ran on an H200 (55 GiB peak) | CPU AdamW; a two-step smoke run before the full jobs |

## Code changes

1. **`src/spara/latent_distillation.py`:** the search (reusing `latent_diagnostics.draw`, `capture`,
   `generate_rows` and `fair_rl.load_noise`'s sha check), the replay check, the latent-target loss
   object, the arm runner (reusing `expert_iteration.training_rows` and `detour_scaling.run_arm` or
   its pieces), and `compare`.
2. **`src/spara/training.py`:** an `offload_optimizer` option in `train_stage`.
3. **`src/spara/cli.py`:** `latent-distillation --part search|train --arm S|D --seed N`, and
   `--part compare` on the CPU.
4. **`configs/latent_distillation.json`:** the settings above.
5. **`scripts/pace/latent_distillation.sbatch`:** search or one training job, given variables.
6. **Tests** (`tests/test_latent_distillation.py`, CPU, tiny models as elsewhere):
   - the push recorded equals the push applied, and replay reproduces the route;
   - L_lat is 0 when h = a*, matches the formula otherwise, ignores drift outside the push subspace, and is 0 on released rows;
   - the calibration sets λ as specified;
   - both arms build identical rows;
   - `train_stage` with the offloaded optimizer matches the regular one for a step.
7. **`GLOSSARY.md`:** an entry for latent distillation.

## Critique and changes (2026-11-19, before any job)

Checked against the code and the recorded data, as the handoff asked:

| Check | Finding | Change |
| --- | --- | --- |
| `hidden_states[24]` is what `LatentAdjuster(model, 24)` edits | Yes: `hidden_states[L]` is the output of `layers[L-1]`; a push δ there makes `hidden_states[24]` at the latents z + δ (unit test on a tiny model) | None |
| Target capture | The latents come before the answer and attention is causal, so the inference prefill's z equals the training pass's; z is taken from a one-map prefill under bf16 autocast, as training runs | None |
| **Cosine loss** | At layer 24 the four slots' median norms are about 90, 164, 94 and **2,103**, and a scale-1 push is 10%, 41%, 10% and 1% of them. 1 − cos is about (‖δ‖/‖z‖)²/2: 0.005, 0.08, 0.005 and 0.0001. Slot 2 would dominate and slot 4's push would be invisible. Cosine is not scale-free across slots | Squared error in the push's own subspace, per slot, normalised by that slot's push energy: every slot counts equally, and drift outside the 16 directions is not penalised |
| Calibration | Per row is right; zero-push maps have zero latent gradient at the start, so calibrating on "the first 8 found rows" could divide by about 0 | Calibrate on the first 8 maps found by a push; the norm is of the accumulated step gradient before clipping, as training does |
| Scale-8 pushes | In Q6 at layer 24, 6 of the 78 solved maps were first solved at scale 8 (47 at scale 1, 13 at 2, 12 at 4). Scale 8 is 0.8 to 3.3 times ‖z‖ in three slots: targets far from the model's states | Cap at scale 4; spend the 64 tries as 32 at scale 1, 16 at 2, 16 at 4 |
| Replay threshold | 99% had no basis; batched against single bf16 decoding can flip near-tie routes | 95% of pushed maps, non-reproducing maps dropped |
| `train_stage` with CPU AdamW | Matches the regular optimizer after a step (unit test); memory and step time from the smoke job | `offload_optimizer` option |
| Rows | Targets are looked up by `map_id` on pool rows; released rows and maps without a target give 0 (unit test) | None |
| Answer format | `answer_text(route)`, E's format, checked with the evaluator | None |
| **Smoke job 7169636** (8 maps, two steps of D) | One real update at learning rate 1e-5 moved the layer-24 latents by 10–15% of their norm on every found map. The squared form of the loss went from about 5.6 to 89, mostly drift in slot 4's subspace (its norm is about 100 times its push size). A squared loss's gradient grows with that drift, so the latent term would soon be many times its calibrated half of the route gradient: C′'s failure | Unsquared distance (above): its gradient's size is fixed, so λ holds for the whole run. The smoke also measured: peak GPU memory 32 GiB with CPU AdamW; about 10 s per optimizer step; λ = 0.039 (route gradient norm 41, latent 520) |
| Evaluation | `detour_scaling.run_arm` does not evaluate the 400 transfer maps and keeps no answer text | Own runner: the 240, the 400, 98 seen maps, the benchmark; answer text kept for the rescore |

## Budget and running

From measured rates: the diagnostics' latent search ran about 6.6 tries per second on one
Blackwell slice, and E's fine-tune took 13 minutes of training on an H200.
- **Search:** at most about 425 failing maps × 64 tries, fewer with early stopping: about 1 hour on
  one slice (the smoke's 8 maps took 86 s).
- **Training:** four runs (S and D × seeds 42 and 43). From the smoke: 375 steps × about 10 s ≈ 1 hour,
  plus about 15 minutes of loading, checks, 738 greedy answers, the benchmark and saving: about
  1.25 hours each, 5 hours for four.
- **Total about 6 GPU-hours, so one job** (limit 10 hours).
- **One job** runs the search and then S42, D42, S43, D43 in order, unless the estimated total is
  above 8 GPU-hours, in which case seed 43's pair goes to a second job on the same search (the
  user's instruction, 2026-11-19).
- **Slurm, as the user requires:** `--gres=gpu:rtx_pro_6000_blackwell:1 --cpus-per-task=8
  --mem=128G` (training) or `--mem=64G` (search), `--time` and `--output` only; never `--qos`,
  `--account` or `--partition`.
- **Order:** smoke (8 maps of search; two training steps of D with CPU AdamW; a few greedy answers),
  then the full job (`scripts/pace/latent_distillation.sbatch`, `SPARA_STAGE=all`).

## Amendment, 2026-11-19: the search missed its replay bar (made after the search, before any training)

**What happened.** The search (job 7169732, `sft-run/results/ld-search-7169732`, 62 minutes) found
266 of 500 maps:
- 81 with no push;
- 185 by a push: 146 at scale 1, 25 at scale 2, 14 at scale 4, after about 20 tries each.

Replayed alone, 169 of the 185 pushed maps gave the same route: **91.4%, below the 95% bar fixed
above**, so the search is invalid under the pre-fixed rule. The job was cancelled before training.

**Diagnosis.** The 16 mismatches (11 at scale 1, 5 at scale 2):
- In 13, the replayed route matches the found one for 2 to 10 moves, then differs.
- In 3, it differs from the first move.
- No answer was rejected, and the basis sha256 matched.

A bookkeeping error (a recorded push other than the one applied) would break routes from the first
move on most maps, and the CPU test shows the recorded push is the applied one. The pattern fits
near-tie greedy tokens flipping between batch-of-16 and batch-of-one bf16 kernels. The 95% bar
(itself set on judgment earlier the same day) underestimated how often that happens on these
latent-pushed maps. Their successes are fragile by construction: about 1 in 20 tries.

**Decision (by the agent, without the user, who was away; to be confirmed or rejected by the user).**
- Train on the 250 kept maps (81 unpushed, 169 pushed and reproduced); the 16 are dropped, as
  fixed above.
- S and D use identical rows, so the drop does not favour either arm.
- `run_train --accept-search` records `search_accepted_below_replay_bar` in every run.
- `compare` still reports the **pre-fixed reading, which will be "withheld"** because the search
  check failed. Beside it, `reading_if_replay_amendment_accepted` applies the same readings with
  the replay check alone set aside (every other validity check still applies).
- **Whether to accept that second reading is the user's call**; it was not fixed before the search.
- **The user accepted the amendment** (2026-11-19, after the comparison), so the reading is
  `reading_if_replay_amendment_accepted`.

**Search descriptives.**
- Pushes were 10%, 37%, 11% and 1% of ‖z‖ in the four slots (median).
- 14 of the 266 found maps needed scale 4.

## Amendment 2, 2026-11-19: D's latent loss was swamping the update (made after D42's first steps, before any D result)

**What happened.** Job 7170079 completed S42 (375 steps, evaluated; valid and kept). It then
started D42 with λ = 0.39, calibrated as fixed above (route gradient norm 20.7, latent 26.5).
D42's step gradient norms were 10 to 40 times S42's on the same rows:

| Steps | S42 gradient norm | D42 gradient norm |
| --- | --- | --- |
| 1–9 | 7 to 13 | 55 to 413 |

Clipped to 1, the route signal was a small fraction of each update. The latent loss on target rows
rose (0.47 to 2.0) instead of falling. D42 was cancelled at step 9; no D answer was evaluated.

**Cause.**
- Maps LIVR solved unpushed already had a latent error of about 0.4 push units before any update.
  At step 7 every target row was unpushed, with latent loss 0.40 and gradient norm 169.
- That error is numerical mismatch: z was captured by the inference prefill, but h comes from
  the training forward pass (another sequence length, training mode, bf16). It is largest on
  slot 4, whose norm is about 100 times its push size.
- The unsquared distance gives any nonzero error a full-size gradient. So these noise directions
  got full-size gradients through the high-norm slot, far beyond what calibrating on pushed maps
  measured.

**Changes to D (S unchanged, S42 kept).**
1. **Only maps found by a push carry the latent loss.** The 81 unpushed maps no longer anchor
   LIVR's latents. D is now S plus the push targets on the 169 pushed maps.
2. **z is recomputed by the training forward pass** (training encoding with the found answer,
   training mode, bf16) on LIVR before any update. The starting error is then the push itself.
   The mismatch to the inference z is recorded per run.
3. **Huber in push units:** per slot e = ‖P_s(h_s − a*_s)‖ / √E_s; e²/2 below 1 and e − ½ above,
   averaged over slots. Errors well under one push unit (numerical noise) get small gradients;
   drift gets a gradient of fixed size.

- λ is still calibrated as fixed above (0.5, first 8 pushed maps), on the new loss.
- The target-fit diagnostic is unchanged (inference prefill, unsquared distance), so S42's
  diagnostic stays comparable.
- **Guard while running:** if D's early gradient norms are again several times S42's at the same
  steps, D is stopped again.
- The readings, rows and evaluation are unchanged. The design of D changed after S42's result was
  seen, but it was driven by D's training mechanics, not by any held-out outcome. The user should
  know both.

## Amendment 3, 2026-11-19: the latent gradient's share is set per step (after amendment 2's first steps, before any D result)

**What happened.** Job 7170831 ran D42 with amendment 2. λ was calibrated at 0.105 (route
gradient norm 20.7, latent 98.6), and the mismatch between inference and training z was 6% of the
push norm.
- Step 1 matched the calibration: total gradient norm 15.0, against 12.8 for S42.
- From step 3, on steps with pushed rows, D's norm was again 3 to 10 times S42's (for example
  87.7 against 6.8).
- Steps without pushed rows matched S42 (11.1 against 10.1; 5.2 against 4.7).
- The guard in amendment 2 applied; D42 was cancelled at step 22, with no D answer evaluated.

**Cause.** Once the weights move, the latent error moves into directions where the parameters'
effect is much larger than along the original pushes, which is what the calibration measured.
One λ fixed at the start cannot hold the ratio it was meant to hold.

**Change (D only).** Each optimizer step:
1. accumulates the route gradient over its 8 rows;
2. accumulates the latent gradient over the rows that have a target;
3. adds the latent gradient rescaled to **0.5 × the route gradient's norm**, then clips and steps
   as before.

- A step with no pushed rows is exactly S's step.
- This is what the fixed λ was meant to do, and it holds at every step (`training.normalised_step`;
  unit test against a direct computation).
- Logged per step: the route and latent gradient norms and the scale applied.
- λ is still calibrated and recorded, but not used. (After the runs, on 2026-11-19, the
  calibration and the fixed-λ path were removed from the code; D's runs record their λ.)
- Cost: a second forward and backward pass on the pushed rows (about 45% of rows).
- The readings, rows, S and the evaluation are unchanged.
