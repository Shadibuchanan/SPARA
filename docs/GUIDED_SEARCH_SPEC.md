# Guided search: a problem-agnostic exploration layer for detour learning

Status, 2026-11-11:
- **Interface, search and the FrozenLake instance:** implemented and tested on CPU.
- **Step 1a:** answered by a CPU dry run with no model (below). The planned GPU pilot is
  withdrawn before any model run: the dry run already settles both of its readings.
- **Step 1a′, search without an exact step simulator:** run on CPU (below). The
  outputs were fixed before the run; the user approved building it.
  - With the final verdict only, no map is solved within 64 calls.
  - To keep 90% of maps solved within step 1a's budget, a judge may err on at most
    2–5% of moves (ε₉₀), or misread 10% of cells if its only errors are misses.
- **Step 1a″, a search that tolerates a noisy judge:** run on CPU (below). Soft veto
  λ = 10 with re-opening is carried into the [JEPA judge](JEPA_JUDGE_SPEC.md), after a
  tie-break added after the sweep.
  - Given 1,000 calls, it turns almost every frontier failure into a budget stop.
  - Its ε₉₀ for both error types per move is 0.05 within 1,000 calls, but 0.02 within
    64.
- **With LIVR as the policy** (post hoc, from the [JEPA judge run](JEPA_JUDGE_SPEC.md#result-job-6951675)):
  final-verdict-only search solved 55 of 97 maps within 64 calls; the stand-ins solved
  0–9. The learned JEPA judge solved 29.
- **Step 1b, expert iteration:** run (jobs 6961874–6961876,
  [expert iteration spec](EXPERT_ITERATION_SPEC.md#result-jobs-69618746961876)), with
  final-verdict-only search as the route source.
  - **Primary: expert iteration helps.** It solves 88 of 240 held-out maps, against 42
    without fine-tuning.
  - **Secondary: no detectable difference between found routes and labels** (88 against
    97).

This is step 1 of the [detour-learning spec](DETOUR_LEARNING_SPEC.md). Step 0 found
that, at the first forced detour on LIVR's failed routes, the correct move is never the
policy's first choice. Sampling at RL settings gives it zero probability at 45 of 52
points. Once the step simulator rules out moves into holes, the policy's top remaining
move is optimal at 38 of 52 points and second at the other 14.

## Why an interface

The method should carry to other RL problems, so the search knows nothing about grids
or images. A problem plugs in as a [search environment](../GLOSSARY.md) and a policy:

| Piece | What it does | FrozenLake |
| --- | --- | --- |
| `start(problem)` | Initial state | The start cell |
| `actions(problem, state)` | Finite list of candidate actions | U, D, L, R |
| `step(problem, state, action)` | Next state, and whether the attempt is alive, dead or done | A hole is dead, the goal is done, a move off the grid leaves the cell unchanged |
| `key(state)` | Identity for spotting repeats | The cell |
| `answer(actions)` | The answer text in the task's format | `\boxed{RIGHT, DOWN}` |
| `reward(problem, answer)` | The authoritative verifier | The reference evaluator |
| Policy | Log-probabilities of candidate actions after a prefix, for a batch of requests | One teacher-forced pass of the model per request |

A test-only counting puzzle, with no grid and no image, runs through the same search
(tests/test_search.py).

**Where the interface reaches its limits.**
- **No step simulator:** only `reward` is available. The search then reduces to
  sampling with a final check, which is what GRPO already does. Step 2's transition
  model is meant to fill this gap.
- **Free-text actions:** these cannot be listed. A proposer that samples candidate steps
  from the policy would replace `actions`. This is not built.
- **Closed-loop tasks,** where the environment returns an observation after each
  action, fit unchanged.
- **Small problems with a step simulator:** search is cheap even without a policy.
  FrozenLake up to 8 x 8 is one (step 1a).

## The search

Best-first over prefixes (partial answers):
- **Cost:** the negative cumulative log-probability under the policy, renormalised over
  the children the step simulator leaves viable (alive or done).
- **Pruning:**
  - dead children;
  - children whose state was already expanded. This covers revisits and moves off the
    grid, with no FrozenLake-specific rule;
  - children past a depth cap.
- **Forced moves cost nothing.** A node with one viable child needs no policy call.
- **Acceptance:** a done child's answer must pass `reward` before it counts. The search
  stops at the first accepted answer, when the policy-call budget runs out, or when the
  frontier empties.
- **Logging:** every expansion records its prefix, each child's outcome and the
  policy's raw scores. These (state, action, outcome) transitions are step 2's training
  data.
- **Batching:** searches over many problems run in lockstep, and each round sends all
  pending expansions to the policy in one call. The FrozenLake policy currently loops
  over them; batching inside it is a later optimisation.

Repeats are judged by state, but the policy conditions on the whole prefix. The search
keeps the most likely path to each state, judged by the scores seen when that state is
first expanded.

**Cost unit:** policy calls, one forward pass of the prompt plus prefix each.

**The search is a training-time exploration tool.** The benchmark never sees the
simulator, so "search solves N maps" is not a benchmark result.

## Step 1a result: CPU dry run, no model

**Setup.**
- **Maps:** the 97 unseen detour maps, the 40 of the test split and the 57 of
  `fresh/probe`.
- **Search:** a budget of 64 policy calls per map, and no binding depth cap.
- **Two model-free priors:**
  - the **uniform** prior, every candidate scored equally;
  - a **toward-goal** stand-in that prefers moves shortening the grid distance. This is
    LIVR's failure mode.

Reproduce with `scripts/search_dry_run.py`.

| Prior | Solved | Policy calls: median / 90th percentile / max | Routes longer than shortest |
| --- | --- | --- | --- |
| Uniform | 97 of 97 | 9 / 18 / 25 | 23 of 97 |
| Toward-goal stand-in | 97 of 97 | 6 / 11 / 20 | 7 of 97 |

**Simulator agreement.** Every alive, dead and done outcome both searches recorded
(6,892) matches the reference evaluator's verdict on the same route.

**The GPU pilot is withdrawn.** Its readings were fixed above in an earlier draft, and
both are settled without a model:
- **"Ready" if the guided search solves at least 90%.** Uniform search already solves
  100%. A model-guided search could fall short only through a broken adapter, which
  validity checks catch; it is not a reading.
- **"Guidance saves calls" if the guided median is at most half the uniform's (4.5).**
  The stand-in, with LIVR's failure mode, needs 6. Pruning removes the move that the
  failure mode is confident about, so sharper probabilities of the same kind gain
  little.

No new rules replace these.

**What this means for adaptability.** On FrozenLake up to 8 x 8, pruning by the step
simulator alone makes search cheap. The policy's prior decides which route is found,
not whether one is found: uniform search returns a longer-than-shortest route on 23
maps. Showing that policy guidance makes search feasible needs one of:
- **a second problem where uniform search is infeasible;** or
- **a FrozenLake ablation with only the final verdict and no per-step pruning.** This is
  what most RL problems offer. It needs a search mode that is not built, and it can be
  tested on CPU with the two priors before any model is used.

That choice is for the team.

**Still needing the model.** Two measurements cannot be made on CPU:
- that the policy adapter reproduces step 0's probabilities at its 52 primary points
  (within 0.01);
- the **departures**: the steps on each found route where the search overrides LIVR's
  top raw choice. These are what expert iteration would teach.

Both come out of step 1b's first search round, as its validity gate. As a standalone job
they would take about 2 GPU-minutes. The `guided-search` command and
[scripts/pace/guided_search.sbatch](../scripts/pace/guided_search.sbatch) run them
together with both arms. They are not scheduled.

## Step 1a′: search without an exact step simulator (CPU, fixed before the run)

**Question.** Step 1a's search was cheap because the exact step simulator judged every
move. Most RL problems have no such simulator, and step 2's transition model would be an
imperfect one. Two measurements:
- **Final verdict only:** how much does search cost rise with no step judgments at all?
- **Noisy judge:** how accurate must a learned judge be for search to stay as cheap as
  with the exact simulator?

The **judge** is whatever says alive, dead or done for each move during search. In
step 1a it is the exact step simulator.

**Shared setup.**
- **Maps:** step 1a's 97 unseen detour maps.
- **Priors:** step 1a's two model-free priors. No model and no GPU.
- **Depth cap:** 64 moves, as in step 1a.
- **Verifier:** the real one (the reference evaluator) still accepts or rejects every
  finished answer in every mode.

### Mode 1: final verdict only

The search may use only the candidate actions, the answer writer and the verifier's
accept or reject. It never calls the step simulator or reads the evaluator's status text
(which would reveal holes). It has no states, so it cannot spot repeats or prune.

- **Ending is a candidate.** The policy scores an extra candidate, *end the answer
  here*, alongside the moves.
- **Order.** Best first by the answer's probability under the policy. A popped prefix
  costs one policy call and pushes each extended prefix and the ended answer. A popped
  ended answer costs one verifier call. A prefix that reaches the depth cap is ended
  with no call.
- **What this mode is.** It is the de-duplicated, deterministic best case of sampling
  with a final check, the setting GRPO works in. Its cost is a lower bound on what
  sampling from the same prior would pay.
- **How each prior scores ending:**
  - **uniform:** ending and each move score equally (1/5 each);
  - **toward-goal:** each move scores −3 × the grid distance to the goal after it,
    and ending scores −3 × the distance now. Moves off the grid leave the agent in
    place. At the goal, ending is the most likely choice. Elsewhere, ending is e⁻³
    times as likely as the best move.
- **Budget:** 20,000 policy calls per map. Verifier calls are not capped separately.
  Each policy call queues one ended answer, or up to five at the depth cap.
- **Validity:** every accepted answer passes the reference evaluator, and popped costs
  never decrease.

### Mode 2: noisy judge

Step 1a's search, unchanged, with each alive or dead judgment replaced by a judge that
errs. Arriving at the goal is always judged correctly.

- **Error types:**
  - **misses:** a hole judged safe, at rate ε; the search walks on from the hole;
  - **false alarms:** a safe cell judged a hole, at rate ε;
  - **both:** each at rate ε.
- **Two noise structures:**
  - **per cell:** a misread map. Each hole is read as safe, or each safe cell other
    than the start and goal is read as a hole, with probability ε. Every move into
    that cell is then judged the same way. This is a perception error.
  - **per transition:** each (state, move) judgment errs independently with
    probability ε. Asking again gives the same answer. This is the error rate a
    transition model would be scored on.
- **Nested errors.** Each cell, or each (state, move), draws one random number per
  seed. So the errors at a given ε include those at every smaller ε.
- **Rates:** ε ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5}, each with 5 noise seeds. ε = 0
  is step 1a.
- **Anchor point:** per transition, misses at 0.5 and false alarms at 0.1. These are
  roughly the rates of the step-grounding linear probe on G's states. That probe flagged
  a hole ahead at 46–51% of fatal moves and at 8–11% of safe moves
  ([step grounding 6926184](experiments/step-grounding-6926184/README.md)). It is weak
  evidence: one probe, measured only on the moves the model took.
- **Budget:** 1,000 policy calls per map.
- **Why a search ends unsolved, when its frontier empties.** Every map is solvable, so
  either:
  - **blocked:** the judge rules out every truly safe route. This is the judge's error;
  - **poisoned:** a truly safe route survives the judge, but the search closed a state
    on it after reaching that state through a missed hole, so the safe route was pruned
    as a repeat. This is the search's error, not the judge's accuracy.

  The two causes are counted separately. The search is not changed to avoid poisoning.
- **Validity:**
  - at ε = 0, both structures reproduce step 1a exactly: the same routes and call
    counts;
  - repeated queries get the same judgment;
  - the order of expansions does not depend on the budget. So one run at the largest
    budget gives the results at every smaller one.

### Outputs (fixed before the run; descriptive, no verdicts)

- **Per mode, prior and condition:**
  - maps solved within 64 policy calls (step 1a's budget) and within the mode's cap;
  - for noisy judges, the mean and range over seeds;
  - policy calls, median and 90th percentile, with unsolved maps counted at the cap
    plus one;
  - verifier calls, median and 90th percentile over solved maps;
  - rejected answers;
  - stop reasons (solved, budget, blocked, poisoned).
- **Final verdict only:** also solved within 1,000 calls, and results by grid size.
- **One summary number per prior, structure and error type:** ε₉₀. It is the largest
  swept ε such that, at it and at every smaller swept rate, the noisy judge's search
  solves at least 90% of maps within 64 policy calls (mean over seeds). It is also
  given within 1,000 calls.
- **The anchor point's results,** set against ε₉₀.

**What the outcomes would mean.**
- **If final-verdict-only search with the toward-goal prior needs far more than step
  1a's median of 6 calls,** or fails within the budget, then FrozenLake without step
  judgments is the regime where the policy, or a learned judge, decides feasibility.
  That is step 2's target.
- **ε₉₀ is the accuracy a learned judge needs.** The anchor shows roughly where current
  latent states sit against it.

Reproduce with `scripts/search_sweep.py`.

### Step 1a′ result (CPU, 2026-11-10)

Evidence: [docs/experiments/search-without-simulator/](experiments/search-without-simulator/README.md).
42 seconds of CPU, no GPU. Every validity check passed. At ε = 0 both noise structures
reproduce step 1a exactly.

**Final verdict only: search cost rises by orders of magnitude.**

| Prior | Solved within 64 / 1,000 / 20,000 calls | Policy calls, median |
| --- | --- | --- |
| Exact judge, toward-goal (step 1a) | 97 / 97 / 97 | 6 |
| Uniform | 0 / 9 / 45 | over 20,000 |
| Toward-goal | 0 / 45 / 72 | 1,455 |
| *Toward-goal, off-grid moves ruled out (exploratory)* | *9 / 67 / 90* | *329* |

- **Why the exploratory row.** It was added after the run. The fixed priors spend much
  of their probability padding refused routes with moves off the grid, which change
  nothing.
- **Without that padding, the cost is still about 55 times step 1a's.**
- **So the first outcome above holds.** Without step judgments, FrozenLake is the
  regime where the prior or a learned judge decides whether search is feasible.

**Noisy judge: ε₉₀ is 0.02 to 0.1.** It is the same for both priors and both budgets.

| Structure | Misses | False alarms | Both |
| --- | --- | --- | --- |
| Per cell | 0.1 | 0.05 | 0.02 |
| Per transition | 0.05 | 0.05 | 0.02 |

- **Per-cell misses sit on the threshold.** Their ε₉₀ of 0.1 rests on means of 88.2
  and 87.6 maps against the 87.3 required. One more lost map in one seed would have made
  it 0.05 for the uniform prior.
- **Errors cost success, not budget.**
  - Every condition solves the same maps within 1,000 calls as within 64.
  - Every unsolved search emptied its frontier.
- **How often each failure happens.** By construction, misses can only poison and false
  alarms can only block. The finding is the frequency:
  - misses at ε = 0.5 poison about 40–46 of 97 maps;
  - per-transition false alarms at 0.1 block 16.6 maps per seed.
- **At the anchor** (per transition, misses 0.5, false alarms 0.1), the search solves
  40–43 of 97 maps. Of the 485 searches per prior:
  - 189–203 were poisoned;
  - 83 were blocked.
- **Against ε₉₀, current latent states are far off.** By the one weak probe, they sit
  well outside the accuracy a hard-pruning judge needs.

**What this means for step 2 (post hoc, not readings).**
- **A transition model used as a hard pruner must be very accurate,** wrong on about 2%
  of moves or fewer.
- **Most of the damage is the search's design, not the judge's accuracy.**
  - **Misses:** all their losses are poisoned closed states. A search that re-opened
    states after a refused answer would turn misses into extra verifier calls.
  - **False alarms:** they block because a dead judgment is a veto. Using the judge as
    a penalty on the move's score, with the final verdict as the check, would let search
    recover. It would also keep the final-verdict-only mode's gains with budget, which
    hard pruning loses.
- **Both changes were then built and measured:** step 1a″ below. The JEPA model is
  specified in the [JEPA judge spec](JEPA_JUDGE_SPEC.md).

## Step 1a″: a search that tolerates a noisy judge (CPU, fixed before the run)

The user approved building both changes. Step 1a′ found that misses lose maps only
through poisoned closed states, and false alarms only through vetoes.

**Re-opening (aimed at misses).**
- Each closed state remembers the prefix that closed it.
- A child dropped as a repeat is parked under its state, together with its parent.
- When a finished answer is refused, every state that the refused route closed is
  re-opened (except the start). The parents parked under it go back on the frontier at
  their original cost.
- A re-expanded parent costs a policy call like any expansion with two or more viable
  children.
- With an exact judge no answer is refused, so the search is unchanged.

**Soft veto (aimed at false alarms).** A move judged dead is deferred, not dropped.
- **Cost.** It is pushed at the parent's cost plus λ nats, plus −log q of the move.
  - If the node called the policy, q is the policy renormalised over the node's
    non-repeat children, dead ones included.
  - If the node was forced (no call), q is uniform over them. No extra policy calls
    are made.
- **Continuing.** The search continues from the state the step returned, as if the
  judge were wrong.
- **Unchanged parts.** Judged-viable children are ordered exactly as before.
  λ = ∞ is step 1a′'s veto.
- **Rates.** λ ∈ {1, 3, 10}.

**Variants.** Seven new ones, against step 1a′'s hard veto:
- re-opening alone;
- soft veto at each λ;
- soft veto at each λ with re-opening.

Each runs on step 1a′'s full grid of priors, structures, error types, rates and seeds,
and the anchor point. The budget is 1,000 policy calls and the depth cap 64.

**A guard.** Each search stops after 50,000 expansions. Re-expansions of forced nodes
cost no calls, so this caps the work. Any search that hits it is reported.

**Validity:**
- re-opening with an exact judge reproduces step 1a exactly;
- the default (λ = ∞, no re-opening) reproduces step 1a′;
- every accepted answer passes the reference evaluator;
- the expansion order does not depend on the budget (a test).

**Outputs (descriptive):**
- per variant: ε₉₀ within 64 and within 1,000 calls, for each prior, structure and
  error type;
- the anchor point's solved maps within 64 and 1,000 calls;
- policy calls at ε = 0, the cost of each variant when the judge is right;
- stop reasons, with frontier stops split into blocked and poisoned.

**The setting carried into step 2.**
- **Choice:** the variant with the highest ε₉₀ within 1,000 calls for both error types,
  per transition, with the toward-goal prior. This is the error type a transition model
  is scored on.
- **Ties:** broken by fewer median policy calls at ε = 0.
- **Use:** that variant is the search setting for the JEPA judge, and its ε₉₀ is the
  JEPA judge's accuracy target.

Reproduce with `scripts/search_sweep.py --modes robust`.

### Step 1a″ result (CPU, 2026-11-11)

Ran on CPU in 2 hours 11 minutes (7,832 seconds, almost all in the re-opening
variants). Evidence and full tables are in
[search-without-simulator](experiments/search-without-simulator/README.md#step-1a-a-search-that-tolerates-a-noisy-judge).

**Validity: all checks passed.**
- **The default reproduces step 1a′:** its summary is identical, timings aside.
- **Re-opening with an exact judge reproduces step 1a:** the same routes and policy
  calls on all 97 maps, for both priors. So does soft veto λ = 10, alone or with
  re-opening.
- **Every accepted answer passes** the reference evaluator.
- **The budget-independence test passes.**

**Toward-goal prior** (uniform is in the evidence). ε₉₀ is per transition, within
64 / 1,000 calls. The anchor is maps solved, mean of 5 seeds:

| Variant | ε₉₀, both errors | ε₉₀, misses | ε₉₀, false alarms | Anchor, 64 / 1,000 calls | Exact judge: solved; calls, median / p90 |
| --- | --- | --- | --- | --- | --- |
| Hard veto (step 1a′) | 0.02 / 0.02 | 0.05 / 0.05 | 0.05 / 0.05 | 39.8 / 39.8 | 97; 6 / 11 |
| Re-opening | 0.02 / 0.05 | 0.05 / 0.1 | 0.05 / 0.05 | 34.6 / 49.2 | 97; 6 / 11 |
| Soft veto λ = 1 | — | — | — | 39.0 / 39.0 | 82; 7 / 1,001 |
| Soft veto λ = 3 | — | — | — | 40.6 / 40.6 | 85; 7 / 1,001 |
| Soft veto λ = 10 | 0.05 / 0.05 | 0.05 / 0.05 | 0.1 / 0.1 | 43.0 / 43.0 | 97; 6 / 11 |
| λ = 1 with re-opening | 0 / 0.02 | 0.01 / 0.02 | 0.05 / 0.05 | 34.2 / 49.0 | 93 (90 within 64); 7 / 26 |
| λ = 3 with re-opening | 0.02 / 0.05 | 0.02 / 0.05 | 0.05 / 0.1 | 34.0 / 51.2 | 94 (93 within 64); 7 / 16 |
| λ = 10 with re-opening | 0.02 / 0.05 | 0.05 / 0.1 | 0.1 / 0.1 | 34.4 / 52.2 | 97; 6 / 11 |

"—": the search solves fewer than 90% of maps even with an exact judge, so there is no
ε₉₀.

**Stop reasons.** Over the 41,710 noisy searches of each variant: both priors, every
rate above 0, and the anchor; budget 1,000.

| Variant | Solved | Blocked | Poisoned | Budget | Expansion guard |
| --- | --- | --- | --- | --- | --- |
| Hard veto | 28,900 | 9,258 | 3,552 | 0 | 0 |
| Re-opening | 29,852 | 8,176 | 258 | 3,172 | 252 |
| Soft veto λ = 1 | 27,920 | 5,463 | 8,327 | 0 | 0 |
| Soft veto λ = 3 | 30,914 | 5,408 | 5,388 | 0 | 0 |
| Soft veto λ = 10 | 32,671 | 5,486 | 3,553 | 0 | 0 |
| λ = 1 with re-opening | 32,140 | 18 | 1 | 8,194 | 1,357 |
| λ = 3 with re-opening | 34,062 | 26 | 6 | 6,405 | 1,211 |
| λ = 10 with re-opening | 34,493 | 57 | 18 | 6,055 | 1,087 |

No accepted answer failed the reference evaluator.

**The setting carried into step 2: soft veto λ = 10 with re-opening.**
- **The registered rule tied three variants:** re-opening, soft veto λ = 10, and both.
  All three have ε₉₀ = 0.05 within 1,000 calls and a median of 6 calls at ε = 0.
  - The sweep script first printed "re-opening", the first of the three in its list.
    That was list order, not a choice, and the script now reports the tie.
- **The tie-break, added after the sweep** and recorded in the [JEPA judge
  spec](JEPA_JUDGE_SPEC.md) before its run: the whole solved-within-1,000 curve for
  the same condition. λ = 10 with re-opening is highest at every rate:

| ε | 0.01 | 0.02 | 0.05 | 0.1 | 0.2 | 0.3 | 0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Re-opening | 95.0 | 94.0 | 89.2 | 73.6 | 47.0 | 30.6 | 7.4 |
| Soft veto λ = 10 | 93.6 | 92.4 | 88.2 | 77.0 | 56.2 | 44.6 | 22.4 |
| λ = 10 with re-opening | 95.8 | 95.0 | 91.8 | 81.6 | 64.0 | 53.2 | 32.2 |

**Observations (descriptive, as registered):**
- **With enough calls, the two changes remove almost every frontier failure.** With
  λ = 10 and re-opening, 75 of 41,710 searches end blocked or poisoned, against 12,810
  with the hard veto. The rest become budget stops.
- **Re-opening costs calls.**
  - Within 64 calls it raises no ε₉₀ and lowers one: per-cell misses, from 0.1 to 0.05.
  - It lowers the anchor from 39.8 to 34.6 maps, because parked parents are re-expanded
    first after a refused answer.
  - Its gains appear within 1,000.
- **A cheap soft veto hurts even with an exact judge.**
  - With λ = 1 or 3, the search walks on through real holes. The cells beyond a hole
    are then closed by a route through it, so the true route is dropped as a repeat.
    This is the same poisoning that misses cause.
  - With an exact judge, it solves 82 and 85 of 97. Re-opening repairs this only in
    part: 90 and 93 within 64 calls, 93 and 94 within 1,000.
- **λ = 10 alone is the best variant within 64 calls.** It has the highest anchor
  (43.0 maps), ε₉₀ = 0.05 for both error types, and it changes nothing with an exact
  judge.
- **Re-opening variants hit the 50,000-expansion guard** in 0.6–3.3% of noisy
  searches. They re-expand forced nodes, which cost no calls.
- **The budget mismatch.** The rule was set at 1,000 calls, but the JEPA arm searches
  with 64. Within 64, λ = 10 with re-opening has ε₉₀ = 0.02, and λ = 10 alone 0.05.

## Step 1b: expert iteration (sketch)

Superseded by the [expert iteration spec](EXPERT_ITERATION_SPEC.md), which takes its routes
from final-verdict-only search and its control from the detour scaling check. The sketch
is kept as written.


- **Rounds.**
  1. Draw maps with `spara.fresh_maps`.
  2. Run guided search with the current policy.
  3. Fine-tune on the routes the search found, mixed with the released training maps.
  4. Search again with the updated policy.
- **Training targets are found routes only, never the BFS route.** The BFS route exists
  for FrozenLake and for few other problems.
- **Control: SFT on the same maps with shortest-route labels.** This is FrozenLake-only
  and separates "search helps" from "more maps help".
  - **Expect less separation than this suggests.** Guided search finds a shortest route
    on most maps (90 of 97 for the stand-in), so the two arms' targets will largely
    coincide.
  - **On FrozenLake, step 1b mainly tests generalization from generated maps.** The
    detour spec names that as the main risk.
- **Choosing among routes:** train on the first route found, or let the search collect
  several per map and keep the shortest or the most likely. This is open.
- **Reading:** on the 40 test detour maps (the benchmark's sampled decoding) and on a
  fresh set held out from every round. Never on maps used in training.
- **Data format:** found routes are written as a `fresh/<name>` set with `text_output`
  filled in. `train_stage` currently reads only the released training split, so it
  needs a split parameter.
- **Detour share:** the generator draws detour maps about 10% of the time, like the
  released sets. Enriching detour maps changes the training distribution, and is a
  decision for the team.

## Scope assumptions (issue #8)

- **Per-step simulator feedback.** The search prunes with the step simulator's verdict
  on each move.
- **Generated training maps.** Step 1b needs maps beyond the released training set.

## Running

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/search_dry_run.py --data-root data/vsp   # CPU
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/search_sweep.py --data-root data/vsp \
  --output docs/experiments/search-without-simulator            # CPU, step 1a′ (--modes in_grid: exploratory)
PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/search_sweep.py --data-root data/vsp \
  --output docs/experiments/search-without-simulator --modes robust   # CPU, step 1a″ (about 2 hours)
python -m spara.cli guided-search --config configs/livr.json --data-root data/vsp \
  --checkpoint /path/to/stage2/final --probe-points /path/to/detour-probe/points.jsonl \
  --fresh-split fresh/probe --output /path/to/run                                       # GPU, not scheduled
```

## Not in scope

- Proposing candidates for free-text actions, KV-cache reuse across expansions, and
  batching inside the FrozenLake policy.
- Other changes for a noisy judge, beyond step 1a″'s re-opening and soft veto: for
  example, a state key that remembers moves judged dead, or a soft judge that uses
  probabilities.
- Training of any kind. Step 1b is in the [expert iteration spec](EXPERT_ITERATION_SPEC.md).
- Any change to the benchmark or its decoding.
