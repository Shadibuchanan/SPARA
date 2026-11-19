# SPARA

SPARA studies whether adapting latent visual representations improves route
planning beyond reinforcement learning on output actions alone.

## Language

**LIVR**:
Latent Implicit Visual Reasoning, the published method that learns visual
representations from final-answer supervision without helper-image targets.
_Avoid_: LVIR

**LIVR baseline**:
The project's reimplementation of LIVR's published planning experiment, following
its documented method and reference benchmark before subsequent reinforcement
learning. It is the common starting point for the planned RL comparisons.

**Scaffolding Minds baseline**:
The project's adaptation of Scaffolding Minds (arXiv 2608.19669v2) to the LIVR
baseline's benchmark, backbone and evaluator. Its Stage 1 learns latent targets
from helper images, so it is a helper-supervised reference, reported separately
from LIVR and SPARA. See ADR-0003.

**Scaffolding RL**:
Reinforcement learning that samples Gaussian adjustments to latent states, with
learned mean and scale, and rewards those that lead to route success.

**Direct SFT**:
Supervised fine-tuning on map questions and route answers without LIVR's
latent-token method.

**Text-only RL**:
Reinforcement learning from the LIVR baseline that optimizes generated route
text against route success, without sampling or adjusting latent tokens.
Condition B in the planned comparisons.

**B′ and C′**:
Conditions B and C rerun under fair conditions. Both start from the LIVR rerun and
train on unseen detour maps (`fresh/detour-pool`), sampling from the model's own
distribution, with the maps per step set by a pilot. C′'s latent adjustment lies in the
principal basis of LIVR's latents. B′ against C′ is the main comparison. See
docs/STRUCTURED_EXPLORATION_SPEC.md, amendment of 2026-11-17.

**Step grounding**:
An auxiliary training objective for route writing. At each move of a teacher-forced
route, a head reads one layer's hidden state and predicts the agent's current cell
and the outcome of each possible move (safe, hole, off the grid, goal). The labels
come from the map layout and are used in training only, never as model input.
*LIVR + step grounding* retrains LIVR's Stage 2 with it. It is a separate condition,
not one of A, B or C. See docs/STEP_GROUNDING_SPEC.md.

**Latent JEPA**:
An auxiliary training objective for LIVR's latent tokens. Through a small predictor,
the four latents predict an embedding of each map cell's image tokens, with LeJEPA's
SIGReg against collapse. The targets come from the image itself: no layout labels
and no helper images. *EJ2* is round 2's EB2 trained with it. *EJF* is the retry with
fixed targets: E's cell states, centred per position and projected once by a frozen
PCA, with no SIGReg. See docs/PERCEPTION_JEPA_SPEC.md.

**Map layout**:
The arrangement of safe tiles, holes, the start, and the goal, independent of
how that arrangement is rendered as an image.

**Route**:
An ordered sequence of cardinal movement actions proposed for a map.

**Detour map**:
A map whose shortest safe route is longer than the grid distance from the start to
the goal, so any successful route must at some point move away from the goal. About
10% of the released training and test maps. See docs/DETOUR_LEARNING_SPEC.md.

**Step simulator**:
A program that applies one action to a problem's state and says whether the attempt
is still alive, has failed, or is finished. For FrozenLake, it walks the map one move at
a time under the reference evaluator's rules.

**Search environment**:
What guided search needs from a problem: its candidate actions, a step simulator, a
way to write an answer, and a verifier for finished answers. FrozenLake is the first;
the interface is meant to carry to other problems. See docs/GUIDED_SEARCH_SPEC.md.

**Guided search**:
Best-first search over partial answers. It expands the actions the policy rates most
likely among those the judge (by default the step simulator) leaves viable, and
accepts a finished answer only if the verifier does. It is a training-time exploration
method; the benchmark never uses it.

**Expert iteration**:
Training a policy on the successful answers its own guided search found, then
searching again with the updated policy.

**E2R**:
Expert iteration's round 2 arm E2 with a rebalanced training mix: half the found
detour rows, plus 1,000 new no-detour maps, for a detour share of 37% instead of 70%.
It tests whether the mix caused E2's losses on no-detour benchmark maps. See
docs/EXPERT_ITERATION_ROUND2_SPEC.md, amendment of 2026-11-17.

**Latent distillation**:
Searching LIVR's latent states per map (random pushes at one layer, kept when the
verifier accepts the answer), then fine-tuning the model on the found routes while
pulling its own latent states toward the pushed states that produced them (arm D),
against the same routes without that pull (arm S). See docs/LATENT_DISTILLATION_SPEC.md.

**Judge**:
Whatever tells guided search, move by move, whether an attempt is alive, has failed or
is finished. The exact step simulator is one; a learned transition model would be an
imperfect one. The verifier, not the judge, decides whether a finished answer counts.

**Final-verdict-only search**:
Search that sees only the verifier's accept or reject on complete answers, with no
judge. It verifies answers in order of their probability under the policy, so it is the
best case of sampling with a final check. See docs/GUIDED_SEARCH_SPEC.md, step 1a′.

**Route success**:
Acceptance of a proposed route by the reference benchmark evaluator: the agent
avoids holes and finishes at the goal after the complete action sequence.

**Helper image**:
An additional visual target or aid, such as a solved-route overlay or heatmap,
beyond the original unsolved map. SPARA excludes helper images from training and evaluation.
The Scaffolding Minds baseline is the one exception: it reads them during training
only (ADR-0003).
