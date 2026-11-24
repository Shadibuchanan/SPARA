# Structured exploration on harder maps (B′ and C′)

Status, 2026-11-08: the probe ran (job 6884608; see [Probe result](#probe-result)).
No B′ or C′ run has been approved. The training code for structured noise (Gaussian
heads in the structured basis) is not built.

Amended 2026-11-17: after an audit of B and C, B′ and C′ are redesigned so the
comparison can show an effect if there is one. See
[Amendment: rerun under fair conditions](#amendment-rerun-under-fair-conditions-2026-11-17).
It replaces the reading and budget sections below. Built and tested on CPU on 2026-11-17
(see [Built](#built-2026-11-17)). Run on 2026-11-17 (jobs 7142592, 7143007, 7143008); see
[Result](#result-jobs-7142592-7143007-and-7143008).

## Why

[Condition C](LATENT_RL_RESULTS.md) added isotropic noise to LIVR's latent states and
scored the same as text-only RL (B). Two limits showed up in its training data:

- **Per-map success is bimodal.** On most maps the model succeeds in all eight
  rollouts or in none, and GRPO learns nothing from either. On the test maps, only
  28 of 400 had a success rate strictly between 0 and 1 over 16 samples, so new maps
  of the same sizes would not by themselves give more signal than the training maps.
- **Isotropic noise does not rescue failing maps on balance.** C succeeded at least
  once on 19 training maps where all of B's rollouts failed, and B on 18 where all
  of C's failed. Most of C's extra signal came from noise breaking maps the model
  already solved.

This plan changes both: noise only in the directions in which latents of different
maps actually differ, and training maps on which the model succeeds about half the
time.

## New maps

Every released image is a FrozenLake render: gymnasium 1.0.0's renderer, given the
row's `map_desc`, reproduces all 1,000 training and 400 test images pixel for pixel.
The released layouts fit a simple generator: start and goal are distinct cells drawn
uniformly, every other cell is a hole with probability about 0.2, and every map is
solvable. For example, a 3 x 3 start sits next to its goal in 34% of training maps;
uniform placement gives 33%.

[src/spara/fresh_maps.py](../src/spara/fresh_maps.py) draws layouts this way, rejects
unsolvable ones and any layout in the train or test split or already drawn, renders
them, and writes rows with the released prompt text. Each set (probe, training,
held-out) has its own seed and a manifest with the row file's SHA-256 and the
renderer versions ([requirements/fresh-maps.txt](../requirements/fresh-maps.txt)).
The training loop and probes read a set as split `fresh/<name>` under the data root.

## Structured noise

- **Basis:** the latent states at the output of decoder layer 16 for the 1,000
  training maps, from the LIVR rerun `6253300`, centred within each grid size. For
  each of the four latent slots, the top 16 principal directions and the standard
  deviation along each. The number 16 is an assumption, from the battery's
  standardised participation ratio of 10 to 18.
- **Adjustment:** δ = c Σⱼ uⱼ sⱼ εⱼ with εⱼ ~ N(0, 1): noise along each direction in
  proportion to how much maps differ along it, with c the scale in those units.
- If the training maps are 7 x 7 or 8 x 8, recompute the basis on those sizes before
  any training run and report the variance it captures.

## Probe (one GPU job, about 0.3 GPU-hours)

All on the LIVR rerun `6253300`, with a fresh probe set of 100 maps each of 5 x 5,
6 x 6, 7 x 7 and 8 x 8.

1. **Basis**, as above.
2. **Size sweep.** For every probe map, the greedy answer and eight samples with the
   RL generation settings (temperature 0.7, top-p 0.9). Per size: the share of maps
   with mixed sample outcomes (1 to 7 of 8 succeed), all failing and all succeeding;
   greedy success; the share of greedy failures where at least one sample succeeds
   (text sampling's own rescue rate, B's exploration); prompt length and peak memory.
3. **Noise probe.** On the 200 probe maps of 5 x 5 and 6 x 6: the greedy answer with
   no adjustment, and with 4 draws each of isotropic σ ∈ {1.2, 1.6, 2.0, 2.4} and
   structured c ∈ {0.5, 1, 2, 4, 8}. Per scale: route-change rate; success rate;
   rescue rate (draws that succeed on maps whose unadjusted answer fails); break rate
   (draws that fail on maps whose unadjusted answer succeeds); rescue share (rescued
   draws over route-changed draws); and maps with any rescue or any break, since
   draws on one map are not independent.

## Rules fixed before the probe

- **Scale:** for each noise family, the smallest scale that changes at least 10% of
  greedy routes and lowers success by at most 5 points (C's rule).
- **Noise for C′:** structured noise at its chosen scale, if one qualifies and its
  rescue share is at least that of the isotropic scale with the closest route-change
  rate. Otherwise isotropic noise at its chosen scale.
- **Training sizes:** every size whose mixed share is within 5 points of the best
  size's, with equal numbers of maps per size.

## Probe result

The full results and files are in
[experiments/explore-probe-6884608](experiments/explore-probe-6884608/README.md).

**Harder maps did not add signal.**
- **Mixed share is flat with size.** It is 11%, 12%, 8% and 10% at 5 x 5 to 8 x 8. On the larger grids, all-fail maps replace all-success ones: 53% and 65% all fail at 7 x 7 and 8 x 8.
- **Same as B's training.** That is the same signal density B had on the released training maps (9.9% of groups mixed).
- **Rule outcome.** The size rule therefore selects all four sizes.

**Structured noise passed its rule, but only just.**
- **What the rule chose:** c = 4. It changes 10.0% of greedy routes, exactly the threshold, and leaves success unchanged. Its rescue share is 0.21, against 0.14 for the isotropic scale with the same route-change rate.
- **Uncertainty:** the bootstrap interval on that difference includes zero.
- **Overlap with text sampling:** of 57 failing maps, structured noise rescues 11 and text sampling 12; the two together rescue 16.
- **Coverage:** the noise probe covered sizes 5 and 6 only.

**Open decision.** B′ as specified trains only on fresh maps of sizes 5 to 8. The
benchmark is sizes 3 to 6, so B′ against B would also change the training sizes,
not only the exploration. B′ against C′ is unaffected.
*Settled by the 2026-11-17 amendment:* B′ and C′ train on detour maps of 3 x 3 to
6 x 6, LIVR's training sizes.

## How B′ and C′ would be read

*Superseded on 2026-11-17 by [the amendment](#amendment-rerun-under-fair-conditions-2026-11-17),
which changes the maps, the sampler, the batches, the noise, the heads and the readings.
Kept as the record of the original plan.*

B′ and C′ would each train one epoch on the same 1,000 fresh maps, with B's and C's
recipes otherwise unchanged.

- **Primary:** C′ with δ = μ against B′ on the 400-map reference benchmark, exact
  McNemar test, different only if p < 0.05. The benchmark stays primary so results
  remain comparable with A, B and C ([ADR-0002](adr/0002-reproduce-livr-reference-protocol.md)).
- **Secondary:** both on 400 held-out fresh maps of the training sizes; each against
  B and the LIVR rerun on the benchmark; C′ with δ = μ against δ = 0.
- **Diagnostic:** the share of training groups with mixed rewards.

## Budget

*Superseded for B′ and C′ by [the amendment's budget](#budget-1).*

The probe is about 0.3 GPU-hours. B′ and C′ would take about 1 GPU-hour each, and
need separate approval after the probe.

## Running

```bash
python -m spara.fresh_maps --data-root data/vsp --name probe --sizes 5,6,7,8 --per-size 100 --seed 101
python -m spara.cli explore-probe --data-root data/vsp --checkpoint /path/to/stage2/final \
  --split fresh/probe --output /path/to/probe
```

On PACE, submit [scripts/pace/explore_probe.sbatch](../scripts/pace/explore_probe.sbatch)
with `SPARA_LIVR_CHECKPOINT`.

## Amendment: rerun under fair conditions (2026-11-17)

Agreed with the user on 2026-11-17, after an audit of conditions B and C. This
replaces [How B′ and C′ would be read](#how-b-and-c-would-be-read) and the budget above,
and settles the open decision on training sizes. Nothing here is built or run. The
readings are fixed here, before any B′ or C′ run.

### Why B against C could not answer the question

C did not differ from B (323 against 326 of 400, p = 0.63;
[LATENT_RL_RESULTS.md](LATENT_RL_RESULTS.md)). That result stands and stays reported.
But the setup the two arms shared left neither much room to change, so the null says
little about latent exploration. Each issue below was measured before this amendment.

| # | Issue | Arms | Evidence |
| --- | --- | --- | --- |
| 1 | **Memorised training maps.** 86.3% of groups had all eight rollouts succeed; 9.9% of rollouts had a nonzero advantage. 58 of B's 125 updates (28 of C's) had no group with mixed rewards at all. | B, C | [TEXT_RL_SPEC.md](TEXT_RL_SPEC.md), line 35; [TEXT_RL_RESULTS.md](TEXT_RL_RESULTS.md), lines 55–61; `zero_variance_groups` in [B's](experiments/text-rl-6256423/steps.jsonl) and [C's](experiments/latent-rl-6879761/steps.jsonl) step logs |
| 2 | **The sampler could not reach the missing move.** At 45 of 52 failed detour decisions, every correct move had probability exactly zero after top-p 0.9. | B, C | [configs/text_rl.json](../configs/text_rl.json#L11); [DETOUR_LEARNING_SPEC.md](DETOUR_LEARNING_SPEC.md), lines 218–220 |
| 3 | **Rollouts did not come from the distribution the loss differentiates.** `generate` also applied the checkpoint's repetition penalty (1.05) and top-k (50). The loss scores a plain softmax at the rollout temperature, and with one update per batch the ratio is 1, so nothing corrects for the difference. | B, C | [text_rl.py#L66-L67](../src/spara/text_rl.py#L66-L67), [#L37-L44](../src/spara/text_rl.py#L37-L44), [#L50-L53](../src/spara/text_rl.py#L50-L53); `generation_config` in every run manifest, e.g. [explore probe](experiments/explore-probe-6884608/manifest.json); [detour_probe.py#L105-L109](../src/spara/detour_probe.py#L105-L109); [DETOUR_LEARNING_SPEC.md](DETOUR_LEARNING_SPEC.md), lines 224–225 |
| 4 | **A small update.** 125 steps at learning rate 1e-6; KL to the start stayed at 0.004–0.04. Scaffolding Minds trains to 500 steps (Fig. 2); E took 375 steps at 1e-5. | B, C | [TEXT_RL_SPEC.md](TEXT_RL_SPEC.md), lines 32–38; [TEXT_RL_RESULTS.md](TEXT_RL_RESULTS.md), line 43; [EXPERT_ITERATION_SPEC.md](EXPERT_ITERATION_SPEC.md), lines 50–53 |
| 5 | **Isotropic noise.** σ = 1.6 in each of 8,192 dimensions, while maps' latents differ by 0.04–0.25 per dimension, in 10 to 18 effective directions. σ was chosen by how often it broke routes on memorised maps, and C's extra signal came mostly from breaking maps it already solved. | C | [LATENT_RL_SPEC.md](LATENT_RL_SPEC.md), lines 51 and 59–74; [LATENT_RL_RESULTS.md](LATENT_RL_RESULTS.md), lines 64–68 and 95–97 |
| 6 | **Heads that barely moved.** σ went from 1.600 to 1.621 and mean \|μ\| from 0 to 0.098, so C was close to B plus fixed noise. Scaffolding Minds' gain needs both a learned mean and a learned variance (Table 3). | C | [LATENT_RL_RESULTS.md](LATENT_RL_RESULTS.md), lines 70–73 |
| 7 | **A benchmark that can hardly move.** One sampled answer per map. Over 16 samples, 312 of 400 maps are always solved and 60 never; most failures are on the 40 detour maps. | Reading | [benchmark.py#L21](../src/spara/benchmark.py#L21); [TEXT_RL_RESULTS.md](TEXT_RL_RESULTS.md), lines 71–73; [DETOUR_LEARNING_SPEC.md](DETOUR_LEARNING_SPEC.md), lines 32–44 |
| 8 | **One run per arm.** Two LIVR runs with identical settings disagree on 31 of 400 maps. | Reading | [LATENT_RL_RESULTS.md](LATENT_RL_RESULTS.md), line 3; [TEXT_RL_RESULTS.md](TEXT_RL_RESULTS.md), lines 26–31 |

### The design

B′ and C′ differ in one thing: C′ adds a learned Gaussian adjustment of LIVR's latent
states (changes 5 and 6). Everything else is shared.

| | B′ and C′ |
| --- | --- |
| Start | The LIVR rerun's final Stage 2 weights (`baseline-6253300`), as for B, C and E |
| Training maps | `fresh/detour-pool`: 2,000 detour maps, 3 x 3 to 6 x 6, seed 202 |
| Map order | Fixed by the seed; both arms of a seed draw the same maps in the same order |
| Rollouts | G = 8; temperature 1.0, top-p 1.0, top-k 0, repetition penalty 1.0; at most 128 new tokens |
| Reward | Success under the reference evaluator, 1 or 0 |
| Batch | P prompts per step (P x 8 rollouts); P is set by the pilot and is the same for both arms |
| Loss | B's text term (Scaffolding Minds eq. 6), averaged over the rollouts in groups with mixed rewards |
| Optimizer | B's: AdamW at a constant 1e-6, gradient clip 1.0, KL 0.001 to the start |
| Length | 125 optimizer steps, so each arm uses 125 x P x 8 rollouts and as many verifier calls |
| Seeds | 42 and 43; each seed runs both arms |

### The changes, and why each is correct

**1. Training maps: unseen detour maps of LIVR's training sizes.**
- **Change:** GRPO trains on `fresh/detour-pool`
  ([DETOUR_SCALING_SPEC.md](DETOUR_SCALING_SPEC.md), line 40), not on the released
  1,000 training maps.
- **Why it is correct:**
  - GRPO learns only from groups whose rewards differ. On memorised maps nearly every
    group succeeds by replaying a taught route (issue 1). The reward then measures
    recall, not a decision the model could carry to a new map.
  - The failures this project targets are on detour maps
    ([DETOUR_LEARNING_SPEC.md](DETOUR_LEARNING_SPEC.md), lines 32–44). Training there
    spends the rollouts where exploration can matter.
  - The sizes are LIVR's own, so B′ changes the maps and not the sizes. This settles
    the open decision above.
  - The pool shares no layout with the released splits, `fresh/probe`, or either
    held-out set. This was checked for expert iteration
    ([round 1](experiments/expert-iteration-6961874/README.md), line 48;
    [round 2](experiments/expert-round2-6992798/README.md), line 52), and the pilot
    checks it again.
- **What it does not do:** no-detour maps are absent from training. The benchmark's
  360 no-detour maps are therefore read as a harm check; expert iteration lost some
  of them.

**2. Rollout sampling: the model's own, full distribution.**
- **Change:** temperature 1.0, top-p 1.0, top-k 0 and repetition penalty 1.0, all set
  in the RL config so nothing is inherited from the checkpoint; at most 128 new tokens.
- **Why it is correct:**
  - A policy gradient is an expectation under the same distribution whose
    log-probability is differentiated. The loss uses a plain softmax at the rollout
    temperature ([text_rl.py#L37-L44](../src/spara/text_rl.py#L37-L44)), with the
    ratio fixed at 1 ([#L50-L53](../src/spara/text_rl.py#L50-L53)). With these
    settings `generate` samples from exactly that softmax, so the estimate is the
    on-policy one GRPO assumes. Before, top-p, top-k and the inherited penalty
    reshaped the sampler without the loss seeing it (issue 3).
  - Full support makes every move reachable. The correct move has probability of at
    least 0.01 at 21 of the 52 failed detour decisions
    ([DETOUR_LEARNING_SPEC.md](DETOUR_LEARNING_SPEC.md), lines 208–211); top-p 0.9
    had set 45 of the 52 to zero.
  - The Scaffolding RL runs already sample this way
    ([configs/scaffolding.json](../configs/scaffolding.json#L36)).
  - **The token cap:** greedy answers never reached 128 tokens
    ([LATENT_BATTERY_RESULTS.md](LATENT_BATTERY_RESULTS.md), lines 60–61), and expert
    iteration used the same cap. A full-support sampler can occasionally fail to stop.
    The cap bounds that cost, a truncated answer is scored like any other, and the
    pilot reports how often it happens.
- **What it does not change:** evaluation. The benchmark keeps its reference decoding
  ([benchmark.py#L21](../src/spara/benchmark.py#L21), ADR-0002), and held-out maps use
  greedy decoding as before.

**3. Batches: enough prompts that most updates carry a gradient.**
- **Change:** P prompts per step, set by the pilot so that a step holds on average at
  least four groups with mixed rewards under B′'s settings. Both arms use that P. A
  step's loss is averaged over the rollouts in groups with mixed rewards. Groups whose
  eight rewards are equal skip the policy and reference passes. A step with no such group
  updates nothing but still counts, so both arms of a seed draw the same maps.
- **The map stream:** one seeded permutation of the pool per pass, run end to end and
  cut into batches of exactly P maps. Every step has P maps, and a pass's last batch
  continues into the next pass.
- **Why it is correct:**
  - A group with equal rewards gives every rollout an advantage of zero
    ([text_rl.py#L32-L34](../src/spara/text_rl.py#L32-L34)), so its clipped
    surrogate is zero. Skipping it removes only its KL term, at coefficient 0.001,
    and two forward passes per rollout.
  - Averaging over the informative rollouts keeps a step's size from depending on how
    many uninformative groups the batch happened to draw. Leaving out groups whose
    rewards are all equal is standard in GRPO training (DAPO: Yu et al., 2025,
    arXiv:2503.14476).
  - Before, 58 of B's 125 updates carried no policy gradient (issue 1).
  - Both arms draw the same maps in the same order, with the same number of rollouts
    and verifier calls. That is the proposal's "match rollout budgets and rewards".
- **Rejected alternative:** refilling each batch until it holds a fixed number of
  mixed groups (DAPO's dynamic sampling). The two arms would then see different maps
  and use different rollout budgets. Here, the number of informative groups is a
  measured outcome instead. Raising it is exactly what latent exploration is claimed
  to do (Scaffolding Minds, Table 8).

**4. Length: B's steps and learning rate, plus a check that the policy moved.**
- **Change:** none to the optimizer: 125 steps at 1e-6, as for B and C and as in
  Scaffolding Minds App. D.1. A step now holds about four informative groups instead
  of about 0.8.
- **Why it is correct:** with the optimizer held fixed, a difference from B and C
  comes from the signal, not from a larger or longer update. The policy-moved check
  under Readings guards against reading a run that changed nothing.

**5. C′'s noise: only in the directions in which maps' latents differ.**
- **Change:** C′ adjusts the four latent states at the output of layer 16 within a
  basis. For each slot, the u_j are the top 16 principal directions of the training
  maps' latent states, centred within grid size, and s_j is the standard deviation
  along each (`principal_basis`,
  [explore_probe.py#L44-L63](../src/spara/explore_probe.py#L44-L63)). The action is
  a = z + Σⱼ uⱼ sⱼ (μⱼ + σⱼ εⱼ), with εⱼ ~ N(0, 1). The basis is computed from the LIVR
  rerun on the 2,000 pool maps.
- **Why it is correct:**
  - Maps' latents vary in about 10 to 18 directions
    ([latent battery](LATENT_BATTERY_RESULTS.md), representation geometry), and 16
    directions per slot hold 78–94% of that variance
    ([explore probe](experiments/explore-probe-6884608/README.md), line 16).
    Isotropic noise put almost all of its power elsewhere (issue 5).
  - Scaffolding RL is a Gaussian policy over a residual latent action. Restricting
    the residual to a subspace keeps that definition: it is the same method with a
    64-dimensional action instead of an 8,192-dimensional one.
  - The basis comes from the maps the arm trains on, never from evaluation maps.
- **Scale:** σ starts at c, chosen by C's own rule: the smallest c in {1, 2, 4, 8}
  that changes at least 10% of greedy routes and lowers greedy success by at most 5
  points ([LATENT_RL_SPEC.md](LATENT_RL_SPEC.md), line 51). The pilot measures it on
  pool maps.
  - **Why the same rule:** keeping it means the scale is not tuned on outcomes.
  - **Why it now picks differently:** on memorised maps, nearly every route change
    was a break. On detour maps, where LIVR mostly fails, a change can also be a
    rescue. The pilot reports both.

**6. C′'s heads: a small policy that can learn.**
- **Change:** two-layer MLPs read each slot's RMS-normalised state, as C's do
  ([latent_rl.py#L31-L35](../src/spara/latent_rl.py#L31-L35)). They output 16 values
  of μ and 16 of log σ per slot. μ starts at zero and σ at c. The latent
  log-probability sums over the 64 coefficients, as in the paper's eq. 5. The head
  learning rate is 2e-5, as for C.
- **Why it is correct:**
  - At 8,192 dimensions the paper's sum swamped the text term, which is why C took
    a mean instead ([LATENT_RL_SPEC.md](LATENT_RL_SPEC.md), line 53). At 64
    dimensions the paper's form can be used as written.
  - Every output acts on a direction that changes routes, so a learned μ or σ can
    matter. C's heads spread a few hundred small updates over 8,192 outputs, mostly
    in directions the model ignores, and barely moved (issue 6).
  - With δ = 0, C′ is exactly LIVR, and μ starts at zero. So C′'s mean starts where
    B′ starts; only the noise differs at step 0.

**7. Readings: held-out maps first, two seeds, the map as the unit.**
- **Change:** the primary reading is greedy success on round 1's 240 held-out maps
  (the 200 of `fresh/detour-heldout` and the 40 test detour maps), pooled over two
  seeds. The benchmark becomes secondary.
- **Why it is correct:**
  - Greedy decoding has no sampling noise, so a difference comes from training.
  - These maps are unseen, come from the training distribution, and include the 40
    released test detour maps. L (42) and E (88) are already recorded on them with
    the same decoding ([expert iteration](experiments/expert-iteration-6961874/README.md),
    lines 92–94).
  - The benchmark can barely move (issue 7). It is still reported, with its
    reference protocol.
  - Two identical LIVR runs disagree on 31 of 400 maps (issue 8), so one run per arm
    cannot separate the method from the run. Summing each map's successes over the
    seeds keeps the map as the unit and averages over runs.

### Pilot (one GPU job, before the runs)

On the LIVR rerun:
1. **Basis** on the 2,000 pool maps; report the variance captured per slot.
2. **Sampler check** (Validity, below).
3. **Signal under B′'s settings:** eight rollouts for each of the first 200 maps in
   seed 42's order. Report the share of groups with mixed rewards, s, and the share of
   answers that reach the 128-token cap.
4. **C′'s scale:** on the same 200 maps, greedy decoding, then four draws for each c
   in {1, 2, 4, 8}. Apply the rule; report route changes, rescues and breaks per c.
5. **L's greedy answers** on every evaluation population: the 240 and 400 held-out maps,
   and the 200 pilot maps for the policy-moved check. They come from the same code and
   decoding as the arms' answers, so no recorded file from an earlier job is needed.

**Rules, fixed before the pilot:**
- **P** is the smallest multiple of 8 with P x s ≥ 4, at most 48.
- **If s < 0.05,** stop. Even full-support sampling then rarely finds a success on
  these maps, which is a reading in itself. The team decides between G = 16 and
  adding no-detour maps.
- **If more than 1% of answers reach the cap,** stop and raise it.
- **If no c qualifies,** stop; the team decides.

### Validity checks

All must pass before any reading:
- **The sampler is the loss's distribution:** on 8 pilot rollouts, at every generated
  position, the distribution `generate` samples from (its processed scores) and the
  softmax the loss differentiates (the raw logits at the rollout temperature) differ by a
  total variation of at most 1e-4.
  - **Why within one call:** comparing the two from the same forward pass removes
    cached-against-uncached and precision noise, so any difference is a processor the
    loss does not model. That cached and uncached logits agree is already tested
    ([tests/test_text_rl.py](../tests/test_text_rl.py), line 28).
  - **Why that test was not enough:** it compares raw logits, so it could not have
    caught issue 3.
  - **On CPU:** with a tiny model whose generation config sets a repetition penalty and
    top-k, the fair settings pass and B's old settings fail (`tests/test_fair_rl.py`).
- **Zero adjustment is LIVR:** C′ with δ = 0 reproduces LIVR's greedy routes on 20 pool
  maps.
- **Pairing:** within a seed, B′ and C′ log the same sequence of maps
  (`rollouts.jsonl`).
- **Rewards:** every rollout's reward matches a CPU rescore with the reference
  evaluator.
- **Sets:** the pool shares no layout with the released splits or either held-out set.

### Readings, fixed before any run

**Policy-moved check (each arm, each seed).**
- **The check:** greedy success on the pilot's 200 pool maps differs from L's on at
  least 16 of them (7.75%). That is the rate at which two identical LIVR runs disagree
  on the benchmark (31 of 400).
- **If any run fails,** all four runs are extended once, to 250 steps.
- **If one still fails,** there is no C′-against-B′ reading: RL did not move the
  policy at this budget.

**Primary: C′ with δ = μ against B′, greedy, on the 240 held-out maps.**
- **The statistic:** per map, d is C′'s successes over the two seeds minus B′'s
  (−2 to 2).
- **The test:** exact two-sided sign test on the maps with d ≠ 0.
- **The rule:** C′ is better only if p < 0.05 and C′ solves at least as many maps as
  B′ in each seed. "Worse" is the mirror rule. Otherwise: no detectable difference.
- **Power:**
  - **Simulation:** per-map success probabilities with mean 0.35, and 23%
    disagreement between runs (E and M disagreed on 17%).
  - **Result:** about 80% power for a gain of 7 points, 57% for 5 and 23% for 3.
  - **Against the paper:** Scaffolding Minds' latent RL beat text-only GRPO by 2.4
    and 4.6 points (Table 3).
  - **So:** a null here rules out large gains, not gains of the paper's size.

**Secondary:**
- **Transfer:** the same test on round 2's 400 held-out 7 x 7 and 8 x 8 maps
  (`fresh/detour-heldout-78`), the larger-map transfer the proposal planned.
- **Test-time adjustment:** C′ with δ = μ against δ = 0, and sampled δ (descriptive).
  This says whether a gain needs the learned adjustment at test time.
- **Benchmark:** the reference protocol, one sample, split into the 40 detour and 360
  no-detour maps; each arm against L.
- **Did RL help at all:** each arm against L on the 240 maps, per seed (exact McNemar).

**Diagnostics (no verdicts):**
- **Signal:** the share of groups with mixed rewards per arm.
- **Rescues and breaks:** paired by map within a seed, maps whose group is mixed only
  for C′ where B′'s group all failed (rescues the latent noise reached) and where it
  all succeeded (breaks).
- **Training:** verifier calls, KL, and μ and σ over training.

**Context, not a controlled comparison:** E solved 88 of the 240 maps and 54 of the
400. E differs in method (supervised updates on found routes), learning rate (1e-5),
steps (375), maps (the first 500 of the pool) and verifier calls (4,412). Report it
beside the arms, without a test.

**Outcome table:**

| Policy moved | Primary | Reading |
| --- | --- | --- |
| No, after the extension | — | RL did not move the policy at this budget; no reading on latent exploration |
| Yes | C′ better | Latent exploration adds value beyond text exploration on unseen detour maps. δ = μ against δ = 0 says whether the gain needs the learned adjustment at test time or came through training |
| Yes | No detectable difference | No gain of about 7 points or more from latent exploration in this recipe. The diagnostics say whether C′ produced more informative groups that did not help, or did not produce them |
| Yes | C′ worse | Latent exploration hurt in this recipe |

### What this comparison can and cannot claim

**Defensible:**
- **One difference.** Start, maps and their order, rollouts, verifier calls, reward,
  optimizer, length, evaluation sets and decoding are shared. Only the latent policy
  differs.
- **The fixes are not tuned on the comparison.**
  - Each answers an issue measured before this amendment (the table above).
  - Each applies to both arms.
  - None was chosen by looking at B′ against C′.
  - The readings are fixed here, and the first B-against-C null stays reported.
- **Two seeds,** with the map as the unit.

**Not claimed:**
- **Small effects.** The design detects gains of about 7 points, not 2 to 5.
- **Scaffolding RL in general.** This adapts it to a 3B model, to LIVR's latents (no
  scaffolding encoder) and to a 64-dimensional residual at layer 16.
- **Which fix mattered.** B′ against B changes several things at once; there is no
  ablation per fix.
- **Search against sampling.** E is context only.
- **Tuned settings.** Layer 16, 16 directions per slot, the head learning rate, the
  target for P and the step count are assumptions.

### ADR-0002

[ADR-0002](adr/0002-reproduce-livr-reference-protocol.md) requires the reference
evaluator for later SPARA comparisons. Every reading here uses it, and the benchmark
is still reported with its protocol. Making held-out maps primary departs from this
spec's earlier choice (How B′ and C′ would be read), not from the ADR, which fixes
the evaluator rather than which maps come first. If the team reads the ADR as making
the benchmark primary for every comparison, it needs revisiting for this one: under
sampling, the benchmark can change on only about 28 of its 400 maps (issue 7).

### Budget

From B's step log: about 24 s per step, of which about 5 s generated 64 rollouts and
the rest ran the policy and reference passes.

| Job | Estimate |
| --- | --- |
| Pilot | About 30 H200-minutes: one load; 2,000 prompt passes for the basis; 1,600 rollouts; about 3,400 greedy answers |
| Each of four runs | About 1.5 H200-hours: 125 steps of about 30–40 s (P = 32 if s ≈ 0.125); greedy on 640 held-out and 200 pool maps; the benchmark; C′ in three latent modes |
| Total | About 6.5 H200-hours. A third seed would add about 3 |

Each run's final weights take 15 GB. Evaluate in the job, keep the weights until the
readings are done, and recheck scratch before submitting.

### Built (2026-11-17)

Tested on CPU with the tiny stand-in model; not run on a GPU.

| Part | Where |
| --- | --- |
| Settings: B′, C′, and the pilot and readings | [configs/text_rl_fair.json](../configs/text_rl_fair.json), [configs/latent_rl_fair.json](../configs/latent_rl_fair.json), [configs/fair_rl.json](../configs/fair_rl.json) |
| The map stream, skipping groups with equal rewards, and logging completions and verifier calls | `stream_batch` and `train_text_rl` in [src/spara/text_rl.py](../src/spara/text_rl.py). Without `steps` and `skip_uniform_groups`, B's and C's loop is unchanged |
| C′'s heads over the basis, its action and summed log-probability | `BasisHeads`, `add_latent_heads` and `latent_objective` in [src/spara/latent_rl.py](../src/spara/latent_rl.py); reloading in [src/spara/model.py](../src/spara/model.py) |
| The pilot, the arms, their evaluation, and `compare` (the readings above) | [src/spara/fair_rl.py](../src/spara/fair_rl.py); `spara.cli fair-rl` |
| The job script | [scripts/pace/fair_rl.sbatch](../scripts/pace/fair_rl.sbatch) |
| Tests | [tests/test_fair_rl.py](../tests/test_fair_rl.py): the stream, skipping, the sampler check, the heads (identity at the start, actions in the basis, replay, save and reload, the objective), the rules, and the pilot, four runs and `compare` end to end |

**Safeguards added while building:**
- **The RL recipe must train on the pool the pilot measured.** The arm refuses to start
  otherwise; the two are named in different files.
- **The arm refuses a pilot that stopped or failed a check,** and so does the job script.
- **C′ checks its fresh heads before training.** The mean adjustment must leave LIVR's
  greedy routes unchanged on 20 pool maps; the run stops if not.
- **The benchmark for C′ uses δ = μ;** sampled δ is read with greedy decoding on the 240
  maps only (descriptive).

### Running

```bash
python -m spara.cli fair-rl --stage pilot --checkpoint /path/to/livr-rerun/stage2/final \
  --data-root data/vsp --output /path/to/pilot
# Read pilot/summary.json: "proceed" must be true. Then, for each arm (b, c) and seed (42, 43):
python -m spara.cli fair-rl --stage train --arm c --seed 42 --pilot-run /path/to/pilot \
  --checkpoint /path/to/livr-rerun/stage2/final --data-root data/vsp --output /path/to/c-42
```

On PACE: [scripts/pace/fair_rl.sbatch](../scripts/pace/fair_rl.sbatch) with
`SPARA_STAGE=pilot`, then `SPARA_STAGE=train SPARA_ARM=b|c SPARA_SEED=42|43
SPARA_PILOT_RUN=<pilot output>`; both need `SPARA_LIVR_CHECKPOINT`. The four runs are
independent and can run at the same time. `fair_rl.compare` applies the readings on CPU.

### Assumptions (not tuned)

- From B: G = 8, learning rate 1e-6, KL 0.001, 125 steps.
- From C: layer 16, head learning rate 2e-5.
- New: 16 directions per slot; a target of four informative groups per step; the
  16-map policy-moved threshold; seeds 42 and 43.

### Result (jobs 7142592, 7143007 and 7143008)

Evidence: run outputs on PACE scratch, not in git (`sft-run/results/fair-pilot-7142592`,
`fair-b-seed42-7142592`, `fair-b-seed43-7143008`, `fair-c-seed42-7143007`, `fair-c-seed43-7142592`),
with `fair-compare-7142592.json` holding every reading.

**What changed from the plan.** The runs used 48 GB Blackwell MIG slices, so they trained with
`fair-rl --low-memory` (AdamW state on the CPU, the same update; a bfloat16 KL reference) on
PyTorch 2.7.1 + CUDA 12.8; both arms identically. The four runs ran in parallel jobs from one
pilot. LIVR's benchmark was rerun on the same stack (328 of 400, as on H200).

**Validity: all checks passed,** including the CPU rescore of all 96,000 rollouts. Every run
moved from LIVR (63–73 of 200 pilot maps), so none was extended.

**Primary: C′ is worse.** On the 240 held-out detour maps, B′ solves 92 and 96, C′ 77 and 86,
LIVR 42. Summed over seeds, B′ is ahead on 34 maps and C′ on 12 (p = 0.0016), both seeds the
same way: "C′ worse: latent exploration hurt in this recipe."

**Secondary.**
- **Transfer (400 maps, 7 x 7 and 8 x 8):** B′ 73 and 67, C′ 61 and 54, LIVR 19; C′ worse
  (p = 0.010).
- **δ = μ against δ = 0:** one map apart per seed; C′'s heads barely moved (σ 4.0 to 3.86 and
  4.13; mean |μ| 0.11 and 0.15).
- **The benchmark:** every run gains on the 40 detour maps (B′ 18 and 17, C′ 15 and 13, LIVR 9)
  and loses on the 360 others (B′ 287 and 281, C′ 300 and 296, LIVR 319); overall B′ 305 and
  298, C′ 315 and 309, LIVR 328.

**What this says.** Under fair conditions text-only RL more than doubles held-out detour
success in 125 steps. Structured latent noise adds reward variance mostly by breaking routes
already solved; its heads do not learn and its text policy moves less. Training only on detour
maps costs the no-detour maps for both arms.

