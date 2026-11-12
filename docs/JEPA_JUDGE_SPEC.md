# JEPA judge: a learned transition model in LIVR's latent space

Status, 2026-11-11: run as job 6951675, with readings fixed before the run.
- **Readings:** all validity checks passed. The learned judge hurts search, and it is not
  ready (below).
- **The judge's errors:** it misses three in four holes.
- **A post-hoc surprise:** with LIVR as the policy, search with the final verdict only
  solved 55 of 97 maps within 64 calls.
- **Post hoc, on CPU:** no threshold on the judge helps, and no layer of LIVR's frozen
  states separates holes well (AUROC 0.72–0.81).

The search setting comes from step 1a″, and the budget was revised before submission.
The user approved the JEPA model. This is step 2 of the
[detour-learning spec](DETOUR_LEARNING_SPEC.md).

## Question

Guided search finds detour routes cheaply when the exact step simulator judges every
move ([guided-search spec](GUIDED_SEARCH_SPEC.md), step 1a). It becomes hundreds of
times more expensive with the final verdict only (step 1a′). Most RL problems have no
step simulator. Can a transition model learned from LIVR's own states stand in for one?

- **The test:** on unseen detour maps, does search judged by the learned model solve
  more maps than search with the final verdict only, at the same budget?

## The model

- **State.** LIVR's hidden state after a route prefix: the decoder output at the
  position where the next move is chosen, read from the frozen LIVR rerun
  (`baseline-6253300`). No LIVR weight changes.
- **Predictor.** An MLP maps (state, move) to the next state's embedding. The target
  is LIVR's own state after that move is written, layer-normalised, with a cosine loss.
  - The targets come from the frozen model, so the embedding cannot collapse, and no
    EMA target encoder is needed.
  - Prediction stays in latent space. A rendered image of the next state would be a
    helper image, which the detour spec excludes.
  - [Agima](related_literature/2610.26845v1.pdf) imagines outcomes as images instead;
    this is the latent counterpart.
- **Outcome head.** It reads the *predicted* embedding and says alive, dead or done.
  The predictor and the head are trained together.
- **What is learned and what is given.**
  - **Given:** movement follows the rules the prompt states. A move changes the cell,
    and a move off the grid leaves the agent in place. The search tracks position by
    these rules, as in step 1a′.
  - **Learned:** what lies in each cell (hole, goal or safe).
  - **Arrivals are learned too.** Step 1a′'s sweep assumed the goal is always
    recognised, so its ε₉₀ does not cover arrival errors.
  - **A missed arrival loses the route:** the search walks past the goal and never
    submits it. A false arrival costs one refused answer. Both are counted.
- **Comparisons** (same data):
  - **probe:** the same capacity, outcome straight from (state, move), with no
    next-state prediction. It shows whether predicting the next state helps the judge.
  - **oracle:** the outcome read from LIVR's *true* state after the move. This needs a
    forward pass per move, which the judge exists to avoid. It shows what the frozen
    states hold.

## Data

- **Training maps:** `fresh/jepa-train`, 3,000 generated maps, 500 of each size from
  3 x 3 to 8 x 8 (seed 203).
  - They are disjoint from the released splits, `fresh/probe`, `fresh/detour-heldout`
    and `fresh/detour-pool`.
  - Every tenth map (index 9, 19, …) is held out for validation: layer choice and
    early stopping.
- **Walks per map:** four, each read in one batched teacher-forced pass:
  - the labelled shortest route;
  - three random walks over moves that stay on the grid. A walk steps into an adjacent
    hole with probability 0.2 (or when nothing else is left), and ends at a hole, at the
    goal or after 2n moves.
- **Labels.**
  - Every state where the agent is alive gets the outcome of each move onto the grid.
  - Every move taken gives a (state, move, next state) triple for the predictor.
- **Evaluation maps, never trained on:**
  - the 97 unseen detour maps of step 1a (40 test, 57 `fresh/probe`);
  - the 200 maps of `fresh/detour-heldout`.

  They are walked the same way, with different seeds.

## Training

- **Candidate layers:** decoder layers 12, 20, 28 and 35. One JEPA is trained per
  layer.
- **Layer choice:** the layer with the lowest validation outcome loss. The probe and
  oracle use the same layer.
- **Settings:** AdamW, learning rate 1e-3, weight decay 0.01, batch 512, at most 40
  epochs, early stopping after 5 epochs without a better validation loss. Seed 42.

## Evaluation

**Judge accuracy on the evaluation maps' walks.**
- **Rates:** the miss rate (a hole judged alive or done) and the false-alarm rate (a
  safe or goal move judged dead), argmax decisions. They are given overall, per
  population and per size.
- **Same for the probe,** and the oracle on moves taken.

**Imagination.** The move taken 1, 2 and 3 steps ahead is judged two ways, on the same
moves:
- from a state imagined by rolling the predictor along the walk's own moves;
- from LIVR's true state there.

**Search on the 97 unseen detour maps.**
- **Shared setup:** LIVR is the policy in every arm, with its raw move probabilities
  and, for ending, its probability of closing the answer. The budget is 64 policy calls
  and the depth cap 64.
- **The arms:**

| Arm | Judge | Search |
| --- | --- | --- |
| Exact | The step simulator | Step 1a's search |
| JEPA | The JEPA's outcome head, from LIVR's state at each prefix | The setting step 1a″ selects (below) |
| Final verdict only | None | Step 1a′'s final-verdict-only search |

- **Verification:** finished answers in every arm are verified by the reference
  evaluator.
- **Forward passes:** one LIVR pass per distinct prefix serves both the policy and the
  judge. The number of passes is reported per arm.
- **A guard.** Each search also stops after 50,000 expansions, the guard of step 1a″'s
  sweep, and such stops are reported.
  - **Why not a smaller guard:** re-expansions reuse the parent's cached pass, so they
    cost CPU time, not GPU time.
  - **Measured on CPU** with the stand-in prior and a per-transition noisy judge:
    50,000 reproduces the sweep exactly.
    - It took 11 to 39 distinct prefixes per map on average (at most 288) at error
      rates from 0 to 0.5.
    - The 1,280 first planned lost one or two maps in several seeds.
- **The JEPA arm's search setting:** soft veto λ = 10 with re-opening (`veto` 10,
  `reopen` true). Recorded here before the run.
  - **How it was chosen.** Step 1a″'s registered rule tied three variants: re-opening,
    soft veto λ = 10, and both. All three have ε₉₀ = 0.05 within 1,000 calls and a
    median of 6 calls at ε = 0.
  - **The tie-break, added after the sweep:** the whole solved-within-1,000 curve for
    the same condition. Soft veto λ = 10 with re-opening is at least as high as the other
    two at every swept rate.
  - **The accuracy target:** its ε₉₀ for per-transition errors of both types. That is
    0.02 within 64 calls (this arm's budget) and 0.05 within 1,000.
  - **A caveat:** the rule was set at 1,000 calls, but this arm has 64. Within 64,
    re-opening gains little, and soft veto λ = 10 alone does slightly better at higher
    error rates (step 1a″ result).

## Readings (fixed before the run)

**Primary: JEPA arm against final verdict only,** maps solved within 64 policy calls,
paired by map, exact McNemar, two-sided.
- **The learned judge makes search work without the simulator:** the JEPA arm solves
  more, p < 0.05.
- **The learned judge hurts search:** it solves fewer, p < 0.05.
- **No detectable difference:** otherwise.

**The primary is a low bar.** At 64 calls, step 1a′'s final-verdict-only search solved
0–9 of the 97 maps with the stand-in priors. So "helps" can mean a handful of maps. The
informative numbers are:
- the 90% threshold below;
- the JEPA arm's solved count against the exact arm's;
- its miss and false-alarm rates against ε₉₀.

**Secondary: ready for expert iteration without the simulator,** if the JEPA arm solves
at least 90% of the 97 maps within 64 calls. This is step 1a′'s threshold. Otherwise,
*not ready*.

**Descriptive (no verdicts):**
- the JEPA's miss and false-alarm rates, set against step 1a″'s ε₉₀ for the selected
  setting;
- the same rates on the moves the search itself judged;
- JEPA against probe against oracle;
- imagination against true states, by horizon;
- results by size; the selected layer; and passes per arm.

**Validity (the readings are withheld if any check fails):**
1. **Disjoint:** the training maps share no layout with the released splits,
   `fresh/probe` or `fresh/detour-heldout`.
2. **States align:** on 20 sampled evaluation states, the batched walk state and a
   single teacher-forced pass after the same prefix have cosine at least 0.99.
3. **The policy matches step 0:** LIVR's move probabilities at step 0's 52 primary
   points are within 0.01 of the reach check's.
4. **Search works with an exact judge:** the exact arm solves at least 90% of the 97
   maps. Otherwise the comparison says nothing about the judge.

**How the outcomes would steer the plan:**

| Outcome | Next |
| --- | --- |
| Helps, and ready | Expert iteration (step 1b) can run with the JEPA judge where no simulator is available. Then test on a second problem. |
| Helps, not ready | The judge is worth improving: fine-tune the encoder with the JEPA loss, more data, or a soft judge using its probabilities. Compare its rates with ε₉₀ to see how far it is. |
| No difference or hurts | LIVR's frozen states do not carry what a judge needs. The oracle tells whether the information appears once the move is written. If it does, the predictor is the gap; if not, the encoder is. |

## Result (job 6951675)

Ran 2026-11-11 on one H200: 1,715 GPU-seconds, against the revised 30–45 GPU-minutes.
Evidence and full tables are in
[jepa-judge-6951675](experiments/jepa-judge-6951675/README.md).

**Validity: all four checks passed.**
- **Disjoint:** no layout shared.
- **States align:** minimum cosine 0.9997.
- **The policy matches step 0:** maximum difference 5.0e-7.
- **The exact arm** solved 97 of 97.

**Primary: the learned judge hurts search.**
- **Within 64 calls:** the JEPA arm solved 29 of 97 maps; final verdict only solved 55.
- **Paired:** 1 map solved by the JEPA arm only and 27 by final verdict only. Exact
  McNemar p = 2.2e-7.

**Secondary: not ready** (29 of 97).

**The judge.** Layer 28 was selected.

| Moves judged | Judge | Miss rate | False-alarm rate |
| --- | --- | --- | --- |
| Evaluation walks (26,560) | JEPA | 0.757 | 0.022 |
| | Probe | 0.775 | 0.021 |
| Moves taken (7,710) | JEPA | 0.785 | 0.025 |
| | Oracle | 0.966 | 0.002 |
| In search (10,285) | JEPA | 0.852 | 0.036 |

- **Misses against ε₉₀:** the target is 0.02, and the miss rate is about 40 times it.
- **Arrivals:** in search, 232 of 837 were missed and 443 invented.
- **Predicting the next state adds nothing over the probe.** Imagined states add false
  alarms with each step ahead: 0.066, 0.100 and 0.124 at 1, 2 and 3 steps, against 0.015
  from true states.

**Next, by the outcome table: "hurts".**
- **The oracle:** reading LIVR's true state after the move, it misses 97% of the holes
  taken. So the information does not appear once the move is written, and the encoder is
  the gap.
- **Caveat:** the oracle's training data is imbalanced (holes are 8% of moves taken).

**Post hoc: final verdict only is much stronger with LIVR than with the stand-ins.**
- **Count:** 55 of 97 within 64 calls, with a median of 15 calls on solved maps. In
  step 1a′ the stand-ins solved 0–9.
- **By size:** it falls with size, to 3 of 17 at 8 x 8.
- **The low bar was not low:** the comparator was much stronger than the spec assumed.
- **For step 1b:** this search could supply routes on small and medium maps with no
  simulator and no judge.

**Post hoc, on CPU: no threshold helps, and no layer holds the holes** (2026-11-11;
[evidence](experiments/jepa-judge-6951675/README.md#post-hoc-cpu-analysis-thresholds-layers-and-search-2026-11-11)).
- **Thresholds:** the JEPA separates holes with AUROC 0.81 and an equal-error rate of
  0.30. No threshold brings both error rates under 0.2.
- **The oracle:** its AUROC is 0.73, below the JEPA's. So the argmax reading of the oracle
  was not an artefact of the imbalance: the encoder is the gap.
- **Layers:** the probe and the oracle retrained at layers 12, 20, 28 and 35 stay between
  0.72 and 0.79.
- **Arrivals:** they separate almost perfectly (AUROC 0.997). The run's arrival errors
  came from argmax.
- **Search at points on the curve** (stand-in prior): at most about 46 of 97 maps within
  64 calls.
- **For step 3:** G, trained with an outcome loss, separates holes better than LIVR on
  unseen maps by the same ridge probe, with AUROC 0.86 against 0.67 on test maps. That is
  still far from what search needs.

## Limits

- **One training run.** The 97-map comparison treats the map as the unit.
- **Generated maps and per-step labels.** The judge learns from labelled walks on
  generated maps, the "learned simulator" version. A final-verdict-only judge is
  harder and not tested here.
- **Rules given, cells learned.** The judge does not learn the movement rules.
- **Size range.** Training maps reach 8 x 8, as the evaluation maps do. LIVR itself was
  trained only on 3 x 3 to 6 x 6.

## Scope

- **Training-time only.** The judge is a training-time tool for search. The benchmark
  and its decoding are unchanged.
- **Generated maps.** Training on them is approved by the user for this line of work.
  The question for reported conditions stays open in issue #8.

## Budget

| Item | Estimate |
| --- | --- |
| Load LIVR | about 2 minutes |
| States: 3,000 training and 297 evaluation maps, one batched pass each | about 6 to 14 minutes |
| Train 4 JEPAs, the probe and the oracle | about 5 minutes |
| Three search arms on 97 maps, about 9,000 LIVR passes | about 8 to 23 minutes |

- **Total:** about 20–25 GPU-minutes was the first estimate.
  - **Revised before submission:** about 30–45 GPU-minutes in one job on one H200, with
    a 60-minute wall limit. GPU-seconds are reported from Slurm.
  - **Why the revision:** step 0 measured about 0.17 seconds per single pass, including
    encoding (373 in 65 seconds).
  - **Passes per arm** (CPU stand-in counts, no more than one per policy call in the
    exact arm):
    - exact: about 1,000;
    - JEPA: about 1,000–4,000;
    - final verdict only: up to 64 per map, about 6,000, since it rarely solves.
- **Kept as they finish:** the interim summary before search, and each search arm
  (`search-<arm>.json`, `searches-<arm>.jsonl`). A run cut short by the wall still has
  the arms it completed. The arms run in the order exact, JEPA, final verdict only.
- **Storage:** about 1.5 GB of states and 50 MB of weights on PACE scratch. The states
  are kept so later analyses run on CPU.

## Running

```bash
python -m spara.fresh_maps --data-root data/vsp --name jepa-train --sizes 3,4,5,6,7,8 \
  --per-size 500 --seed 203 --exclude probe,detour-heldout,detour-pool
python -m spara.cli jepa-judge --config configs/livr.json --settings configs/jepa_judge.json \
  --data-root data/vsp --checkpoint /path/to/stage2/final \
  --probe-points /path/to/detour-probe/points.jsonl --output /path/to/run
```

On PACE, [scripts/pace/jepa_judge.sbatch](../scripts/pace/jepa_judge.sbatch) with
`SPARA_LIVR_CHECKPOINT`.
