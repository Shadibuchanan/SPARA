# SPARA: Spatial Planning with Adaptive Representation Adjustment

**Team:** Akash Singirkonda, Raymond Zeng, Aarya Singh, Shadi Buchanan, Pranav Somu  
**Updated:** October 11, 2026

## What we are trying to do

We want a vision-language model to look at a map and produce a safe path to the goal, without extra helper images during training or testing.

Our idea combines two methods:

- **LIVR** teaches the model to store useful visual information in internal representations called latent tokens, using final-answer supervision.
- **Scaffolding RL** explores adjustments to internal representations and rewards those that lead to successful answers.

SPARA uses LIVR to learn the initial representations, then adapts Scaffolding RL to improve them. Our main question is whether adjusting these representations improves planning beyond reinforcement learning on the output actions alone.

LIVR already demonstrates planning without helper images. Our project tests the combination with latent reinforcement learning and whether that extra step helps.

## Task and starting setup

We will use a deterministic FrozenLake-style environment: a grid with a start, a goal, safe tiles, and holes. Movement has predictable outcomes.

| Item | Plan |
| --- | --- |
| Input | An unsolved map image and route-planning instruction |
| Output | A complete sequence of `UP`, `DOWN`, `LEFT`, and `RIGHT` actions |
| Success | Reach the goal without entering a hole; any valid route counts |
| Supervised training | Learn from correct action sequences |
| RL training | Execute generated routes and reward success |
| Training maps | Begin with 3×3–6×6 grids |
| Evaluation | New layouts, plus 8×8 grids to test transfer |

The original map is always available. SPARA excludes solved-route overlays, heatmaps, and other helper-image targets, while still using action labels and simulator rewards.

Start with **Qwen2.5-VL-3B-Instruct**, LoRA adapters, a frozen vision encoder/projector, and four latent tokens. We have **PACE ICE access** and will measure GPU memory and runtime before scaling. LoRA for this planning task is an adaptation of the papers' setups.

## Implementation plan

### 1. Prepare the environment

Create or load maps, separate training/validation/test layouts, and implement a route scorer. Fix action parsing, boundary behavior, and maximum route length. Run a small supervised baseline to check the pipeline.

### 2. Train the LIVR baseline

First, use a bottleneck attention mask: latent tokens can read the image, while prompt and answer tokens cannot attend directly to it. Train on correct routes so useful image information must pass through the latent tokens.

Then restore normal attention and continue supervised training. Save this checkpoint as the common starting point for evaluation.

As checked on October 11, the official LIVR page still lists code/data as forthcoming. Plan to implement token insertion and attention masking unless an official release becomes available.

### 3. Add latent reinforcement learning

Adapt Scaffolding RL to sample small, input-dependent Gaussian adjustments to LIVR's latent states. Generate routes, score them in the simulator, and update the latent and action-output policies.

The main integration challenge is making these adjustments affect answer generation correctly. Verify that zero adjustment reproduces LIVR and that nonzero adjustments change downstream predictions. Keep rollout and training behavior consistent, including cached decoding.

### 4. Run controlled comparisons

| Condition | Training after the same LIVR checkpoint |
| --- | --- |
| A. LIVR | None |
| B. LIVR + text-only RL | Optimize action outputs without sampled latent adjustments |
| C. SPARA | Optimize both latent adjustments and action outputs |

**The main comparison is C versus B.** This tests whether latent exploration adds value beyond ordinary RL.

Match maps, rewards, rollout counts, and generation limits. Measure held-out route success, performance by grid size, failure types, GPU-hours, and peak memory. Repeat with multiple seeds if resources permit. Use validation for training choices and reserve the test set for evaluation.

A matched helper-supervised reference and larger maps are stretch goals. Preserve the text-only RL control if compute becomes limited.

## Timeline

These are planned milestones and course deadlines, not a record of completed work.

| Dates in 2026 | Planned work / milestone |
| --- | --- |
| Sep 18–Oct 14 | Prepare maps/scorer, profile PACE resources, train LIVR, check the latent interface |
| Oct 12 | Prepare midterm slides for the course's Monday submission rule |
| Oct 15 | Midterm: demonstrate completed components, report blockers, and show pilot results if available |
| Oct 16–Nov 16 | Run matched RL experiments, evaluate transfer, analyze failures and uncertainty |
| Nov 17 | Submit code and audit packet with reproducible configurations and agent-usage documentation |
| Nov 20–24 | Investigate the returned audit and submit the rebuttal by Nov 24 at 11:59 PM ET |
| Nov 30–Dec 4 | TA project/code interviews |
| Dec 1, 3, or 8 | Final presentation on the team's assigned date |
| Canvas deadline | Final report |

### October 15 midterm targets

**Committed:** dataset/scorer, PACE resource profile, a LIVR checkpoint with a tiny-data learning check, and zero-adjustment equivalence at the latent interface.

**Stretch:** a working latent-RL pilot and an initial comparison against text-only RL.

## Immediate next steps

- Confirm actual progress against the midterm targets and assign owners for remaining tasks.
- Verify PACE GPU memory and run a small end-to-end training job.
- Finish the LIVR baseline before scaling latent RL.
- Use measured pilot cost to set the full experiment budget.
- Save configurations, checkpoints, and per-map outputs so results are reproducible.

## Supporting papers

- [LIVR](related_literature/LIVR_2512.21218v2.pdf): learning visual representations without helper targets.
- [Scaffolding Minds v2](related_literature/Scaffolding_Minds_2608.19669v2.pdf): latent exploration and RL. Use v2 for implementation details; v1 is also saved.
- [VSP](related_literature/VSP_2407.01863v1.pdf): visual planning tasks and evaluation.
- [Mirage](related_literature/Mirage_2506.17218v1.pdf): related latent visual reasoning and planning setup.
