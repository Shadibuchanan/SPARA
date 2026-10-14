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

**Map layout**:
The arrangement of safe tiles, holes, the start, and the goal, independent of
how that arrangement is rendered as an image.

**Route**:
An ordered sequence of cardinal movement actions proposed for a map.

**Route success**:
Acceptance of a proposed route by the reference benchmark evaluator: the agent
avoids holes and finishes at the goal after the complete action sequence.

**Helper image**:
An additional visual target or aid, such as a solved-route overlay or heatmap,
beyond the original unsolved map. SPARA excludes helper images from training and evaluation.
The Scaffolding Minds baseline is the one exception: it reads them during training
only (ADR-0003).
