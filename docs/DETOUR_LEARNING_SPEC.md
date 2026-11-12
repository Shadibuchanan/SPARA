# Detour learning: policy improvement through directed exploration

Status, 2026-11-11:
- **Step 0 (the reach check):** ran as job 6937009, 286 H200-seconds. The reading is
  withheld: a validity check failed because its premise was wrong, not the measurement
  (see [Step 0 result](#step-0-result-job-6937009)).
- **Step 1:** its search layer is built and tested on CPU, and step 1a is answered
  without a model (see [the guided-search spec](GUIDED_SEARCH_SPEC.md)).
  - **Step 1b:** expert iteration took its routes from final-verdict-only search with
    LIVR (jobs 6961874–6961876,
    [expert iteration spec](EXPERT_ITERATION_SPEC.md#result-jobs-69618746961876)).
  - **Its readings:** expert iteration helps (88 against 42 of 240 held-out maps), with
    no detectable difference between found routes and labels.
- **Step 2:** run as a JEPA judge in LIVR's latent space (job 6951675,
  [JEPA judge spec](JEPA_JUDGE_SPEC.md#result-job-6951675)).
  - **The judge:** it hurts search, and it misses three in four holes. LIVR's frozen
    states do not show them.
  - **Post hoc:** search with the final verdict only, with LIVR as the policy, solved 55
    of 97 unseen detour maps within 64 calls.

## Problem

Every condition run so far solves most maps whose shortest route heads toward the
goal, and few [detour maps](../GLOSSARY.md). On a detour map the shortest safe route
is longer than the grid distance, so it must at some point move away from the goal.

| Test maps, reference benchmark | No detour needed (360) | Detour map (40) |
| --- | --- | --- |
| Vanilla SFT, no latents (arms A and C, 6259811) | 315 and 318 | 5 and 5 |
| LIVR (6189457 and rerun 6253300) | 316 and 319 | 7 and 9 |
| Text-only RL, B (6256423) | 320 | 6 |
| Latent RL, C (6879761, δ = μ) | 315 | 8 |
| LIVR + step grounding, G (6926184) | 319 | 7 |

What the failures look like (LIVR's greedy routes, test maps; post hoc, from the
[step-grounding](STEP_GROUNDING_SPEC.md) analysis):
- **Fatal moves head toward the goal:** 68 of 68 do.
- **The detour gap is not map size.** On 6 x 6 test maps, LIVR solves 59 of 79 maps
  that need no detour and 1 of 21 detour maps.
- **About half the walks into holes happen where every shortest-route move leads away
  from the goal.** In the other half, a safe move toward the goal existed and the
  model took the toward-goal move into a hole.
- **Holding position is not the limit.** G holds the agent's cell before fatal moves
  (76% at layer 20 by a linear probe, against 16%) and fails just as often.

Exploration has not reached the missing behaviour:
- **Fresh 5 x 5 and 6 x 6 detour maps (22):** greedy decoding solves 6–7. Eight text
  samples add one map, and nine latent-noise settings from the
  [exploration probe](experiments/explore-probe-6884608/README.md) add two.
- **Test detour maps (40):** with 16 samples per map, at least one solves 10 for LIVR
  and 7 for B.
- **RL has only seen memorised detours.** B and C ran RL on the released training maps,
  which SFT had already fitted. On the 98 training detour maps, their rollouts succeed
  68% and 61% of the time by reproducing taught routes. B solves 6 of the 40 test
  detour maps.

GRPO reinforces only routes that sampling produces. So the policy has to change, and
noise in text or latents does not reach the change.

## Direction

Improve the route policy through directed exploration, without helper images. The
learning signal is the simulator's verdict on routes the model produces.

- **Step 0, reach check (this spec, below).** How much probability does the current
  policy give the correct move where a detour is needed? This decides whether search
  can be guided by the policy, or must branch over all four moves.
- **Step 1, expert iteration (sketch).** At training time, search over moves step by
  step, guided by the policy. The simulator's walk reports the move at which a route
  dies, so dead branches are pruned. Then train the policy on the routes the search
  finds, by supervised updates or GRPO. Use generated maps rather than the fixed
  1,000, for variety and volume. The test benchmark is unchanged. Designed in
  [GUIDED_SEARCH_SPEC.md](GUIDED_SEARCH_SPEC.md), as a problem-agnostic interface.
- **Step 2, JEPA transition model ([JEPA judge spec](JEPA_JUDGE_SPEC.md)).** Train a predictor of the next state's
  embedding from the current state and a move. It learns from the (state, move,
  outcome) transitions that step 1's search produces. At test time, when there is no
  simulator, the model would use it to imagine and score continuations. Its role is
  planning; making the state decodable is what G already showed does not change routes.
  The pointer target from the step-grounding spec (the reached cell's features in the
  original image) is allowed. Rendered images of the next state are not (see below).

**The main risk is generalization, not exploration.** Supervised training on the 98
released detour routes left the detour decision only partly fitted on those maps:
LIVR's last-epoch answer loss is 0.013, and at the first detour decision the released
move has median probability 0.82 (step 0's validity pass). LIVR solves 17–22% of
unseen detour maps. Routes found by search could be learned just as narrowly. Variety and
volume of generated maps are the planned mitigation. Step 1's results must be read on
unseen detour maps, never on training maps.

## Scope questions for the team

- **Generated training maps.** Steps 1 and 2 train on maps drawn by
  `spara.fresh_maps`, scored by the reference evaluator's rules. This goes beyond the
  released training set. No helper images are involved.
- **Per-step simulator feedback.** The walk reports the move at which a route fails,
  not only success or failure. Is that within "simulator rewards"?
- **Next-state targets for JEPA.** A rendered image of the map with the agent moved is
  an additional visual target, so it is a helper image under the glossary and is
  excluded. A pointer into the original map image is not.

If the team adopts generated training maps, record the decision in an ADR.

## Step 0: reach check (fixed before the run)

**Checkpoint:** the LIVR rerun's final Stage 2 weights (`baseline-6253300`), the
common starting point. One job.

**Routes:** LIVR's own greedy routes, already recorded (no generation):
- test maps, from [planning probe 6887458](experiments/planning-probe-6887458/README.md);
- the 400 `fresh/probe` maps, 5 x 5 to 8 x 8, from
  [exploration probe 6884608](experiments/explore-probe-6884608/README.md).

Both used the same checkpoint and decoder.

**Decision point.** Walk each greedy route from the start. At each step, the optimal
moves are those that keep the agent on a shortest safe route. Stop at the first step
of one of these kinds:
- **forced:** every optimal move leads away from the goal. This is where a detour must
  start.
- **deviation:** the move taken is not optimal.
- **stopped:** the route ends early with every move optimal.
- **chosen:** on a detour map whose greedy route is optimal throughout, the first move
  away from the goal.

Routes that are optimal throughout on maps needing no detour have no decision point.

**Measurement.**
- **Teacher-forcing:** the greedy prefix up to the decision point is fed back in the
  answer format. The next-token distribution gives each move's probability (each move
  word is one token).
- **Ending the route:** where the route could also end, the probability of continuing
  multiplies the move probabilities.
- **Two distributions:**
  - raw: the model's own softmax, at temperature 1;
  - under the RL rollout settings of B and C, as `generate` applies them: the
    checkpoint's repetition penalty (1.05), temperature 0.7, top-k 50 (Hugging Face's
    default, since the checkpoint sets none), then top-p 0.9, in that order.
- **Per point:** each move's outcome (optimal, hole, off the grid, or safe but not
  optimal), the total probability of the optimal moves, the rank of the best optimal
  move among the four, and the model's argmax.

**Primary population:** forced points on test and fresh maps where the greedy route
did not take an optimal move. These are the failures that need a detour.

**Primary measure:** the share of those points that are within reach, meaning the
optimal moves have a raw total probability of at least 0.01.

**Reading (rule):**
- **tractable** if at least 50% are within reach. Policy-guided search should find
  most forced detours, and step 1 can start from the current policy.
- **hard** if fewer than 20% are within reach. Search must branch over all four moves,
  and the policy must learn the move largely from scratch.
- **partial** otherwise.

**Validity (the reading is withheld if either fails):**
- **Teacher-forcing reproduces the model's greedy choices:** the argmax at the
  decision point matches the greedy route's next token on at least 90% of points. The
  argmax applies the repetition penalty, as the greedy decoding that produced the
  routes did.
- **Memorised answers score as memorised:** on the 98 training detour maps,
  teacher-forcing the released answer gives the released move a median probability of
  at least 0.9 at its decision point.

**Secondary (reported, no verdicts):**
- the share of primary points where the optimal moves have probability exactly zero
  under the RL settings, which RL could never have sampled;
- the rank and median probability, by split and grid size;
- at deviation points, on all maps: the probability of the optimal moves, and how
  often the move taken enters a hole;
- at stopped points, the probability of ending against continuing;
- whether the recorded test completions equal the canonical answer tokens of their
  routes.

**Budget:** one H200, about 5 GPU-minutes, mostly model loading. There are about 300
decision points plus 98 training answers, one forward pass each. Wall limit 20 minutes.

## Step 0 result (job 6937009)

Ran 2026-11-10: one H200, 286 GPU-seconds. There were 275 decision points on unseen
maps, plus the 98 training detour maps. Evidence and tables are in
[detour-probe-6937009](experiments/detour-probe-6937009/README.md).

**Reading: withheld by the rule.**
- **Teacher-forcing reproduces the greedy choices:** passed, 275 of 275.
- **Memorised answers score as memorised:** failed, with a median of 0.82 against 0.9.

**Why the second check failed (post hoc).** Its premise was wrong. This spec cited an
answer loss near 0.001 for the fitted training maps; that was G's (0.00078), and
LIVR's is 0.013. On the 98 training detour maps:
- the released move scores at least 0.9 on 40 and below 0.5 on 24;
- the argmax walks into a hole on 20.

The measurement checks out:
- **Same inputs as training.** The released answers are character-identical to the
  canonical answer string the probe teacher-forces.
- **Confident moves score high.** Where the greedy route took an optimal away move
  (33 forced points), that move has median probability 0.89 and ranks first on 31.
- **Greedy choices reproduce.** The first check's 100%.

The failure reflects the checkpoint, not the probe. Whether to read the primary measure
anyway is for the team.

**Primary measure (computed, unread).** There are 52 forced points where the greedy
route did not take an optimal move; 48 of them walk into a hole.
- **Within reach:** 21 of 52 (40%): test 7 of 21, fresh 14 of 31. Under the rule this
  would read as partial.
- **Median P(optimal):** 0.0076, just below the reach line. Descriptively, the share
  within reach is 65% at 0.001, 25% at 0.05 and 15% at 0.1.
- **The best optimal move never ranks first.** It ranks second at 27 points, third at
  22 and fourth at 3. The move taken has median probability 0.96.

**Secondary.**
- **RL could not sample the detour:** at 45 of 52 primary points, the optimal moves
  have probability exactly zero under B and C's rollout settings (top-p 0.9 after
  temperature 0.7).
- **Deviation points (147):** 71% within reach, median P(optimal) 0.085. The move
  taken enters a hole at 69%.
- **Stopped points (43):** the route ends with median probability 0.986.
- **Repetition penalty:** at 6 deviation points, the checkpoint's repetition penalty
  flipped greedy decoding away from a move the raw softmax ranked first.

**Exploratory (post hoc): with simulator pruning, the detour move is first or second.**
At the 52 primary points, remove the moves into a hole or off the grid. The policy's
top remaining move is then optimal at 38 and second at 14. This covers only the first
detour decision on each route; later decisions are unmeasured.

**What this suggests for step 1 (not a verdict).**
- Sampling-based exploration cannot reach the detour.
- Policy-guided search with simulator pruning reaches the first detour move within two
  branches at every primary point.

## Running

```bash
python -m spara.cli detour-probe --config configs/livr.json --data-root data/vsp \
  --checkpoint /path/to/stage2/final --test-routes /path/to/planning-probe/routes.jsonl \
  --fresh-routes /path/to/explore-probe/sweep.jsonl --fresh-split fresh/probe --output /path/to/run
```

On PACE, submit [scripts/pace/detour_probe.sbatch](../scripts/pace/detour_probe.sbatch)
with `SPARA_LIVR_CHECKPOINT`.

## Not in scope

- Steps 1 and 2, until step 0 is read and the team answers the scope questions.
- B, C or G checkpoints in step 0.
- Any change to the test benchmark or its decoding.
