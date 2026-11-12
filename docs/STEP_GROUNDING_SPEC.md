# Step grounding: per-move cell and move-outcome prediction in Stage 2

Status, 2026-11-10: ran as job 6926184 (3,486 H200-seconds, including a 4-second
failed first submission); see [Result](#result). The reading rules below were fixed
before the run.

## Why

The [planning probe](PLANNING_PROBE_SPEC.md#result) pointed at how the model writes
routes, not at its latents.
- **Position is weak where moves are written.** The agent's cell decodes on only
  39–49% of solved steps (linear probe, layer 20), and the start cell on 45–68% at the
  first move.
- **The move tally is intact.** The agent's offset from the start decodes on almost
  every step, fatal ones included.
- **Fatal steps are worse.** Just before the move into a hole, the cell decodes 16–17%
  of the time, against 26–40% expected from matched solved steps.
- **Latent changes have not helped.** Natural-sized changes to the layer-16 latents
  rarely fix a failure, and [condition C](LATENT_RL_RESULTS.md) did not beat B.

Those findings are post hoc and come from a weak probe. They support a hypothesis, not
a cause: at the positions where moves are written, the model does not hold the agent's
cell or what each move from it would do. Training it to hold them should reduce walks
into holes, which are 68 of LIVR's 73 greedy test failures.

**Relation to the earlier JEPA plan.** An action-conditioned predictor predicts the
next state from the current state and an action. Step grounding is the
label-supervised version of that idea:
- **What it predicts:** the outcome of every action from the current state, with
  labels from the map layout, instead of embeddings from a target encoder.
- **Why it goes first:** it needs no new tokens, attention masks or target encoder, has
  no collapse risk, costs about one GPU-hour, and tests the probe's hypothesis
  directly.
- **The JEPA variant would change only the target.** It would point at the
  image-token features of the reached cell, contrastively over the cells of the same
  image. The [battery](LATENT_BATTERY_RESULTS.md#image-token-cells) showed those cells
  are distinguishable. It is the natural second arm if step grounding shows that the
  model uses position once it is trained in.

## Objective

**Where it applies.**
- **Decision states:** for move k of the teacher-forced released answer, the hidden
  state at the token before the token holding move k's letter. This is the state that
  predicts move k, as in the planning probe (`move_tokens`). Steps whose letter shares
  a token with another move are dropped; there are none in the training set.
- **Layer:** the output of decoder layer 20 (`hidden_states[20]`), the layer the
  planning probe selected for position.

**What it predicts.**
- **Cell:** the agent's cell before move k, as a row and a column (logits beyond the
  grid size are masked).
- **Move outcomes:** for each of U, D, L and R from that cell, one of safe, hole, off
  the grid, or goal.
- **Labels:** computed from the training row's map layout (`map_desc`) and released
  answer, by the reference evaluator's walk. The model never receives them.

**Head.** A parameter-free layer norm followed by one linear map from 2,048 to
8 + 8 + 4 × 4 logits, initialised to zero. With a zero head the first update equals
plain Stage 2's, and no random direction is pushed into the model.

**Loss, per map.**

```
L_ground = ½ · mean(CE_row, CE_col) + ½ · mean over U, D, L, R of CE_outcome
           (averaged over the map's decision states)
L        = L_answer + λ · L_ground
```

- **λ = 1, no sweep.** Both terms are mean cross-entropies over discrete targets, so
  λ = 1 weighs one grounding prediction like one answer token.
- **Head optimiser:** the same AdamW and schedule, with learning rate 1e-3 and no
  weight decay. A fresh linear head at the model's 1e-5 would barely move in 1,250
  steps; 1e-3 is a usual linear-probe rate. Gradient clipping covers the head too.
- **A known shortcut.** The decision state already encodes the move it is about to
  write, and released routes never enter a hole. So the outcome for the direction
  about to be taken is nearly free (always safe or goal). The informative targets are
  the cell and the other three directions. This is acceptable for a training signal,
  but it is why the head's outcome accuracy is not reported as evidence.

## Training

- **Start:** the final Stage 1 weights of the LIVR rerun, `baseline-6253300/stage1/final`.
- **Recipe:** [configs/livr.json](../configs/livr.json) unchanged: 10 epochs of the
  1,000 training maps, learning rate 1e-5, effective batch 8, 5% warmup and cosine
  decay, seed 42, the same data order.
- **What differs from the LIVR rerun's Stage 2:** only the grounding loss, and the
  backward pass's non-determinism. Two LIVR runs with identical settings disagree on
  31 test maps.
- **Settings:** [configs/step_grounding.json](../configs/step_grounding.json).
  The final checkpoint records them under `grounding`, and the head is saved
  beside it (`grounding_head.pt`).

The condition is called *LIVR + step grounding* (G). It is not one of A, B or C.

## Scope assumption (needs the team's confirmation)

SPARA trains on action labels and simulator rewards and excludes helper images.
- **What the labels are not:** images, or anything the model receives as input.
- **What they add:** supervision richer than route answers and rewards: the cell
  before each move, and which neighbours of the route are holes, edges or the goal.
  LIVR, B and C do not get this.
- **Proposal:** report G separately, labelled as using layout-derived training
  targets, as the Scaffolding Minds baseline is reported separately
  ([ADR-0003](adr/0003-scaffolding-minds-helper-supervised-reference.md)). If the team
  adopts it beyond this pilot, record the decision in an ADR.

## Measures and reading (fixed before the run)

**Primary: benchmark success.**
- **Protocol:** the reference benchmark (`spara.cli evaluate`), with sampled decoding
  (temperature 0.7, top-p 0.9, seed 42), on the 400 test maps, as for A, B and C.
- **Comparison:** G against the LIVR rerun `6253300` (328/400), map by map, with an
  exact McNemar test.
- **Reading:**
  - *improves* if G solves more maps with p < 0.05;
  - *harms* if it solves fewer with p < 0.05;
  - otherwise *no detectable difference*.
- **Power:** with one seed, only a net change of about 12–13 maps (3 points) can reach
  p < 0.05, since two identical LIVR runs already disagree on 31 maps. Smaller effects
  cannot be told apart from run-to-run variation.

**Secondary (reported, no separate verdicts).**
- **Greedy test routes:** G against LIVR's greedy routes (327/400, from the
  [planning probe](experiments/planning-probe-6887458/README.md)), with McNemar.
- **Greedy fresh routes:** the 400 maps of `fresh/probe` (100 each of 5 x 5 to 8 x 8)
  against LIVR's greedy routes (217/400, from the
  [exploration probe](experiments/explore-probe-6884608/README.md)), with McNemar, by
  size. The exploration probe's routes are pairable: they come from the same
  checkpoint, decoder and 128-token cap, and all 200 maps shared with the planning
  probe give identical routes.
- **Failure types:** walks into holes against routes that end off the goal, for each
  comparison above. If grounding works as intended, the hole share should fall.
- **Training fit:** the mean answer loss over the last epoch, against the LIVR rerun's
  0.013. More than double that would mean the grounding loss interfered with fitting
  the answers.

**Manipulation check: did position become decodable on maps not trained on?** The
state-tracking part of the planning probe (the same code, without headroom) runs on
G's final checkpoint. Stage 2 fits the training maps almost exactly (answer loss
0.013), and it will fit the grounding labels there too. So the check uses test and
fresh steps.
- **Took:** the ridge probe's position accuracy at layer 20 on solved test steps is at
  least 10 points above LIVR's 39%.
- **Reading pairs:**
  - *took, and the primary improves:* consistent with the hypothesis (one seed, so not
    proof);
  - *took, and no detectable difference:* position became decodable but did not change
    routes. Either the model does not use it, or tracking position is not the
    bottleneck. This outcome is worth stating in advance as a result.
  - *did not take:* the objective did not change how test maps are represented, so the
    run says nothing about the hypothesis.
- **Also reported:**
  - S1 at fatal steps against LIVR's 16% (test) and 17% (fresh 5 x 5 and 6 x 6);
  - held-out training accuracy per layer;
  - the trained head's own accuracy on the decision states of G's test and fresh
    routes, solved steps against fatal ones.

## Result

The full results and files are in
[experiments/step-grounding-6926184](experiments/step-grounding-6926184/README.md).

**Combined reading (rule): took, and no detectable difference.** Position became
decodable but did not change routes. Either the model does not use it, or tracking
position is not the bottleneck.
- **Primary:** 326 of 400 against the LIVR rerun's 328. The two disagree on 30 maps
  (14 for G, 16 for LIVR), p = 0.86. Two identical LIVR runs disagree on 31.
- **Manipulation check:** position at layer 20 on solved test steps is 89%, against
  LIVR's 39%. Before a fatal move it is 76%, against 16%.
- **Walks into holes did not fall:** 68 against 69 on the benchmark, and 67 against 68
  on greedy test routes.

**Not anticipated (secondary, no verdicts).**
- **Tighter training fit:** the last epoch's answer loss is 0.00078, against LIVR's
  0.013.
- **Fresh maps lean worse:** greedy routes solve 203 of 400, against 217 (17 against 31
  discordant, p = 0.059). Every size from 6 x 6 up leans toward LIVR, and both hole
  and off-goal failures rose.
- **The probe's selected layer moved** from 20 to 24. The manipulation check is fixed
  at layer 20, so this does not change the reading.

**Post hoc (not pre-declared).**
- **G knows where it is when it fails.** On test maps and on fresh 5 x 5 and 6 x 6
  maps, the cell before a fatal move is read correctly 76–78% of the time by the ridge
  probe and 92–94% by the trained head. G still walks into holes as often as LIVR.
- **The state often marks the hole.** For about half of the fatal moves, a linear probe
  reads a hole in the direction about to be taken, against 8–11% false alarms on safe
  moves. LIVR's figures were 41–42% and 21–29%.
- **No transfer to larger maps.** On 7 x 7 and 8 x 8 maps, even cells inside the
  trained 0–5 range decode only 40–49% of the time.

**What this means for the plan.** With one seed, this weighs against the planning
probe's hypothesis that failures come from not holding position where moves are
written. It points instead at how a move is chosen from a state that already holds
the cell and often the hole ahead. Both follow-ups above were conditional:
- the JEPA pointer arm, on the model using position once trained in;
- the static-map control, on G improving.

Neither condition is met.

## Budget

About 1.1 GPU-hours on one H200, in one job (wall limit 2 hours):
- **Stage 2 with the grounding loss:** about 0.75 h. LIVR's Stage 2 took 0.69 h of
  steps, and the head adds little, since the hidden states are computed anyway. Loading
  and saving the model add a few minutes.
- **Benchmark evaluation:** about 8 minutes. The LIVR rerun spent about 11 minutes
  outside its training steps, including this evaluation.
- **State probe without headroom** (1,000 training, 400 test and 400 fresh maps):
  about 11 minutes. The planning probe took about 10 minutes before its headroom
  test, with 200 fresh maps.
- **Comparisons and head readout:** CPU, seconds.

The final weights (about 15 GB) stay on PACE scratch until the team decides whether
to keep them.

## Running

```bash
python -m spara.cli train --stage 2 --config configs/livr.json --grounding configs/step_grounding.json \
  --data-root data/vsp --checkpoint /path/to/stage1/final --output /path/to/run/stage2 --final-weights-only
python -m spara.cli evaluate --config configs/livr.json --data-root data/vsp \
  --checkpoint /path/to/run/stage2/final --output /path/to/run/evaluation
python -m spara.cli planning-probe --config configs/livr.json --data-root data/vsp \
  --checkpoint /path/to/run/stage2/final --radii none --fresh-split fresh/probe --fresh-sizes 5,6,7,8 \
  --output /path/to/run/probe
python scripts/analyze_step_grounding.py /path/to/run --data-root data/vsp
```

On PACE, submit [scripts/pace/step_grounding.sbatch](../scripts/pace/step_grounding.sbatch)
with `SPARA_LIVR_STAGE1`.

## Not in scope

- **Grounding during Stage 1**, under the bottleneck. It would force the latents to
  carry what tracking needs, but it doubles the cost.
- **More seeds or a λ sweep.**
- **A control with static map supervision,** such as predicting the hole map from the
  latents. It would separate the per-step structure from the extra layout
  supervision. It is the first follow-up if G improves.
