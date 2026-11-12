# Planning probe: latent headroom and state tracking

Status, 2026-11-08: ran as job 6887458 (1,831 H200-seconds); see [Result](#result).
The reading rules below were fixed before the run and are left as written. No
training run is part of this plan.

## Why

LIVR's failures are mostly walks into holes. In the
[latent battery](LATENT_BATTERY_RESULTS.md), 66 of 72 failures entered a hole.
The latent hole probe's errors did not predict which holes, but that test could not
say whether errors start in reading the map or in planning from it.

[Condition C](LATENT_RL_RESULTS.md) and the
[exploration probe](experiments/explore-probe-6884608/README.md) then showed that
random exploration of the latents rarely finds better routes. Before building an
objective that targets planning (Phase 4, action-conditioned prediction), two
questions decide whether it has something to fix:

1. **Headroom.** Do latent states exist, close to the model's own, that make the
   frozen model write a correct route on maps it fails? If none exist even well
   beyond the natural spread of latents, the limit lies in the layers after the
   latents. Then neither a latent-side objective nor latent RL can reach it.
2. **State tracking.** While the model writes its route, does it still represent
   where the agent is when it makes the fatal move? A position error just before the
   fatal move is the failure an action-conditioned predictor would train against.

Both tests distinguish between explanations; neither confirms one on its own.
Headroom shows that better latents exist within a radius, not that any training
objective would learn to produce them.

## Setup

- **Model:** the LIVR rerun `6253300`, final Stage 2 checkpoint (the start point of B
  and C).
- **Decoding:** greedy, at most 128 new tokens.
- **Maps:** the 400 test maps are primary. The 200 fresh 5 x 5 and 6 x 6 maps of
  `fresh/probe` replicate the state probe. Headroom uses the test maps the model
  fails.
- **Route walk:** the reference evaluator's rules. Off-grid moves are ignored,
  entering a hole fails at once, and the route must end on the goal.

## 1. Headroom

**What is optimised.**
- **Latents:** the four latent states at the output of decoder layer 16, as in C.
- **Basis:** the exploration probe's basis (`basis.pt` from job 6884608): 16
  principal directions per slot, with spread sⱼ along each.
- **Adjustment:** δ = Σⱼ uⱼ sⱼ aⱼ per slot, with the coefficients a constrained to
  ‖a‖₂ ≤ r per slot.
- **Units:** a training map's latent sits on average at ‖a‖ = 4 from its grid size's
  mean (the 16 coefficients have unit variance), so r = 4 is about one typical
  deviation from the size mean. (The typical difference between two maps is √2
  larger. As first written, this line called r = 4 a map-to-map difference.)
- **Radii:** r ∈ {1, 2, 4, 8}.

**Targets, for each failed map.**
- **Correct target:** among the shortest successful routes (found by search, at most
  64 enumerated), the one with the lowest mean token NLL under the unadjusted model.
- **Wrong control:** a route of the same length that differs from the target in one
  move, fails, and is not the model's own route. Among those, the one whose
  unadjusted NLL is closest to the target's. If there is none, the map has no control
  and is reported.
- **Format:** both are written in the model's answer format,
  `\boxed{RIGHT, DOWN, ...}<|im_end|>`.

**Optimisation.**
- The model is frozen.
- Adam (learning rate 0.25) runs on a, starting at zero, for at most 40 steps. The
  loss is the target's mean token NLL, teacher-forced. After each step, a is
  projected back onto the ball of radius r.
- A row stops once the teacher-forced argmax equals the target at every answer
  token. That row then keeps its coefficients from that step.
- The model then answers greedily with the resulting adjustment.

**Measures, per radius.**
- **Fixed:** share of failed maps whose greedy route succeeds with the
  target-optimised latents.
- **Target reproduced / control reproduced:** greedy route equals the target
  (respectively the control).
- **NLL:** mean NLL before and after.

**Reading (fixed before the run).**
- **Headroom present:** fixed share at r = 4 of at least 50%.
- **No headroom:** fixed share at r = 8 below 20%.
- **Otherwise:** partial headroom.
- **Specificity**, compared at r = 4 on maps with a control, with an exact McNemar
  test paired by map:
  - *specific to correct routes* if target reproduced exceeds control reproduced
    by at least 20 points with p < 0.05;
  - *generic steering* if control reproduced is at least target reproduced.

  Generic steering means the later layers write whatever route the latents encode.
  Then finding the right latent is the whole problem, and no part of the model
  favours correct routes on its own.

## 2. State tracking

**States.**
- **Which position:** for move k of a route, the hidden state at the token before
  the token holding move k's letter. This is the state that predicts move k.
- **Layers:** every fourth decoder layer's output (4, 8, …, 36), as in the battery.
  The last of these is after the final norm.

**Probes**, ridge regression as in the battery.
- **Position probe:** maps a state to the agent's current cell, as one-hot row and
  one-hot column. The argmax is taken within the grid.
- **Hole-ahead probe:** maps a state to whether the neighbour in each direction is a
  hole.
- **Training data:** both are fitted on the 1,000 training maps, with their released
  answers teacher-forced, over every step.
- **Avoiding a shortcut in the hole-ahead probe:**
  - The state before a move encodes that move, and released routes never step into
    a hole. A hole-ahead probe could therefore learn "the direction I am about to
    take is safe".
  - To prevent this, it is trained only on directions other than the next move.
  - It is still secondary.

**Layer.** The layer with the highest position accuracy on a held-out 20% of training
maps, with the probe fitted on the other 80%. The final probes are refitted on all
training maps at that layer.

**Test steps.** The model's own greedy routes, every step up to and including the
fatal one. The fatal step is the state before the move that enters a hole. Steps
whose letter shares a token with another move are dropped and counted.

**Primary test (S1).**
- **Fatal steps:** those with k ≥ 2. Before the first move the agent is at the
  start, so a fatal first move cannot be a tracking error. Those are reported
  separately.
- **Expected accuracy:** for each fatal step, the position accuracy of the probe over
  steps in solved test routes of the same grid size and the same k. If that size has
  no solved step at that k, use the nearest k, and count how often this happens.
- **Test:** the observed number of fatal steps with the correct decoded position is
  compared with the expected number, using a one-sided Poisson-binomial lower tail.

**Reading (fixed before the run).**
- **Tracking contributes to failures:** p < 0.05 and observed accuracy at least 15
  points below expected.
- **Tracking intact:** observed accuracy less than 10 points below expected.
- **Otherwise:** inconclusive.

**Secondary.**
- **S2.** For fatal steps whose decoded position is wrong, the share where the fatal
  move would be safe from the decoded position. This is compared with the same share
  averaged over all other non-hole cells.
- **S3.**
  - **Fatal holes:** the share read as a hole (score above 0.5).
  - **Avoided holes:** the same share for hole neighbours not taken, in solved
    routes.
  - **Bias check:** on solved steps, the mean score for the direction taken (always
    safe) against other safe neighbours. A lower score for the direction taken means
    the probe still leans on the upcoming move, which biases fatal holes towards
    "safe".
- **Validity check:** position accuracy on solved test steps against held-out
  training accuracy.

## Result

The full results and files are in
[experiments/planning-probe-6887458](experiments/planning-probe-6887458/README.md).

**Headroom (rule): absent, but the test was too small to answer its question.**
- **Result:** at r = 8, 7 of 73 failed maps are fixed (9.6%), and most optimised rows
  ended on their norm bound.
- **The radii were a design error.** The exploration probe had already shown that
  random changes of this size move few routes: 5.9% at ‖a‖ ≈ 8, and 10% at ‖a‖ ≈ 16.
  So "well beyond the natural spread" above was wrong for r ≤ 8.
- **What it does show:** natural-sized changes to the layer-16 latents rarely fix a
  failure. Larger changes are untested.
- **Specificity:** uninformative. The controls (median NLL 1.13) were never matched
  in difficulty to the targets (0.41).

**State tracking (rule): the test maps read "tracking intact", the fresh maps
"tracking contributes".**
- **The test reading is on the boundary:** a gap of 9.9 points against the 10-point
  cutoff, with p = 0.063.
- **Both splits point the same way:** before the fatal move, the cell decodes 16–17%
  of the time, against 26–40% expected.
- **The probe is too weak for a verdict:** 39–49% on solved steps.

**Post hoc (not pre-declared).**
- **The move count is intact.** The agent's offset from the start decodes on almost
  every step, including 91 of 93 fatal ones. That is bookkeeping on the written
  moves.
- **The start location is weak.** It is only partly present at the writing position:
  45–68% at the first move.
- **The drop is specific to fatal steps.** Position decodes normally at earlier
  steps of failed routes and at solved steps next to a hole.

## Budget

About 0.3 GPU-hours on one H200:
- greedy answers and hidden states for 400 test and 200 fresh maps;
- one hidden-state pass over the 1,000 training maps;
- the headroom optimisation for roughly 70 failed maps, with eight rows per map
  (four radii, target and control).

The ridge probes run in the same job.

## Running

```bash
python -m spara.cli planning-probe --data-root data/vsp --checkpoint /path/to/stage2/final \
  --basis /path/to/explore-probe/probe/basis.pt --fresh-split fresh/probe --output /path/to/run
```

On PACE, submit [scripts/pace/planning_probe.sbatch](../scripts/pace/planning_probe.sbatch)
with `SPARA_LIVR_CHECKPOINT`.
