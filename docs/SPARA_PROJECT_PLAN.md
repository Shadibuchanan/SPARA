# SPARA: Spatial Planning with Adaptive Representation Adjustment

**Team:** Akash Singirkonda, Raymond Zeng, Aarya Singh, Shadi Buchanan, Pranav Somu  
**Updated:** October 11, 2026

**Implementation status:** The LIVR code and bounded PACE pilot are complete;
21 tests pass. PACE job `6189457` completed 10+10 epochs and reference evaluation,
scoring 323/400 (80.75%) in 1 hour 34 minutes. All saved responses were rescored
locally with matching outcomes; final Stage 2 checkpoint backup remains in progress.
See [the pilot report](LIVR_PILOT_REPORT.md) and [full-run record](LIVR_BASELINE_RESULTS.md).

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
| Output | Route answers in the released benchmark's format, preserving its prompt and answer conventions |
| Success | The released reference evaluator accepts the route: avoid holes and end at the goal |
| Supervised training | Learn from correct action sequences |
| RL training | Execute generated routes and reward success |
| Training maps | The released Mirage VSP planning training set used as the source for LIVR's comparison |
| Evaluation | The released benchmark test set and reference evaluator, shared across LIVR and later SPARA comparisons |

The first phase reproduces LIVR's published planning baseline as closely as the
available evidence permits. Preserve the released split and disclose its observed
train/test overlap rather than changing the benchmark. Custom maps, additional
direct-SFT experiments, and alternative scoring are outside this first phase.
See [the reproduction-scope decision](adr/0002-reproduce-livr-reference-protocol.md)
and [the data/evaluator audit](research/LIVR_DATASET_COMPARABILITY.md).

The original map is always available. SPARA excludes solved-route overlays, heatmaps, and other helper-image targets, while still using action labels and simulator rewards.

Use **Qwen2.5-VL-3B-Instruct**, full language-backbone fine-tuning, a frozen vision
encoder/projector, and four latent tokens. Follow the published planning recipe:
10 epochs of Stage 1 and 10 of Stage 2, AdamW learning rate 1e-5, effective batch
size 8, and separate 5% warmup/cosine schedules for each stage. We have **PACE ICE
access** and will profile memory and runtime before the full run. If resources
force a LoRA fallback, document it as a deviation; it is not the reproduction target.

The agent can operate through the current user's configured `ssh pace-ice` alias.
Use remote `$HOME/scratch/SPARA` for working files and outputs, with connection,
Slurm account, and path settings configurable for teammates. The initial profiling
budget is one GPU for at most one hour after setup; select compatible available
hardware rather than requiring a particular model. See the [execution context](agents/pace-ice.md)
and [implementation handoff for review](LIVR_IMPLEMENTATION_SPEC.md).

## Implementation plan

### 1. Prepare the environment

Pin the released data, model revision, preprocessing, dependencies, and reference
evaluator. Use the original input images and route answers, excluding helper-image
targets and simulator-only map metadata from model inputs. Check the LIVR pipeline
on a tiny subset before a full training run.

Use the released reference evaluator for both the baseline and later SPARA
comparisons. Its parser retains `U,D,R,L` characters; off-grid moves stay in place,
holes fail, and success is checked after the complete route. Preserve its behavior
and generation limits; add no custom action-count cap or stop-on-first-goal rule.
The released Mirage evaluator is the verified reference available to us; LIVR's
exact evaluation manifest, decoder, and checkpoint-selection practice remain
unverified and must be documented as such. For unspecified settings, use the pinned
Mirage implementation where applicable and record the assumption; explicit LIVR
settings take precedence. This applies to experimental details, not Mirage's
different latent architecture or helper-image supervision. Any remaining gap
without a relevant source must be identified before implementation.

### 2. Train the LIVR baseline

First, use a bottleneck attention mask: latent tokens can read the image, while prompt and answer tokens cannot attend directly to it. Train on correct routes so useful image information must pass through the latent tokens.

Then restore normal attention and continue supervised training. Save this checkpoint as the common starting point for evaluation.

Following the approved Mirage fallback policy, use the final Stage 1 weights to
start Stage 2 with fresh optimizer state, and report the final Stage 2 checkpoint.
Keep the released training set intact rather than adding a validation holdout for
checkpoint selection. These are documented assumptions about LIVR, supported by
the [pinned reference-configuration research](https://github.com/Shadibuchanan/SPARA/blob/9386caaa181b0905476a630b4fa434ae8309df5f/docs/research/LIVR_REFERENCE_SETTINGS.md).

As checked on October 11, the official LIVR page still lists its code/data as
forthcoming, but Mirage's VSP data and evaluator are public. Implement LIVR's token
insertion and attention masking from the paper unless its official implementation
becomes available. The published 3B planning accuracy is 66%; it is a comparison
reference, not sufficient by itself to establish implementation correctness.

### 3. Add latent reinforcement learning

Adapt Scaffolding RL to sample small, input-dependent Gaussian adjustments to LIVR's latent states. Generate routes, score them in the simulator, and update the latent and action-output policies.

The main integration challenge is making these adjustments affect answer generation correctly. Verify that zero adjustment reproduces LIVR and that nonzero adjustments change downstream predictions. Keep rollout and training behavior consistent, including cached decoding.

### 4. Run controlled comparisons

First evaluate the reproduced LIVR checkpoint with the reference protocol. The
additional direct-SFT experiment proposed earlier is superseded by the narrower
baseline scope. Then use the same reference evaluator for the planned RL comparisons.

| Condition | Training after the same LIVR checkpoint |
| --- | --- |
| A. LIVR | None |
| B. LIVR + text-only RL | Optimize action outputs without sampled latent adjustments |
| C. SPARA | Optimize both latent adjustments and action outputs |

**The main comparison is C versus B.** This tests whether latent exploration adds value beyond ordinary RL.

Match maps, rewards, rollout counts, and generation limits. Measure reference-evaluator
accuracy, performance by grid size, failure types, GPU-hours, and peak memory.
Repeat with multiple seeds if resources permit. Reserve test data for reporting.
The initial baseline uses the documented final-checkpoint fallback above; any
validation policy for later RL experiments belongs to that later planning phase.

A matched helper-supervised reference and larger maps are stretch goals. Preserve the text-only RL control if compute becomes limited.

## Timeline

These are planned milestones and course deadlines, not a record of completed work.

| Dates in 2026 | Planned work / milestone |
| --- | --- |
| Sep 18â€“Oct 14 | Prepare maps/scorer, profile PACE resources, train LIVR, check the latent interface |
| Oct 12 | Prepare midterm slides for the course's Monday submission rule |
| Oct 15 | Midterm: demonstrate completed components, report blockers, and show pilot results if available |
| Oct 16â€“Nov 16 | Run matched RL experiments, evaluate transfer, analyze failures and uncertainty |
| Nov 17 | Submit code and audit packet with reproducible configurations and agent-usage documentation |
| Nov 20â€“24 | Investigate the returned audit and submit the rebuttal by Nov 24 at 11:59 PM ET |
| Nov 30â€“Dec 4 | TA project/code interviews |
| Dec 1, 3, or 8 | Final presentation on the team's assigned date |
| Canvas deadline | Final report |

### October 15 midterm targets

**Committed:** dataset/scorer, PACE resource profile, a LIVR checkpoint with a tiny-data learning check, and zero-adjustment equivalence at the latent interface.

**Stretch:** a working latent-RL pilot and an initial comparison against text-only RL.

## Immediate next steps

- Review the completed LIVR implementation, [results and charts](LIVR_BASELINE_RESULTS.md),
  and [reproduction audit](LIVR_REPRODUCTION_AUDIT.md) before merging the baseline PR.
- Finish the verified Stage 2 checkpoint backup and keep the trained baseline fixed.
- Assign owners and prepare a separate specification for the latent-RL phase.
- Reuse the same benchmark and reference evaluator for the LIVR, text-only RL
  and SPARA comparisons; report the published LIVR score separately.
- Verify the zero-adjustment latent interface and agree on a new compute budget
  before launching additional experiments.

## Supporting papers

- [LIVR](related_literature/LIVR_2512.21218v2.pdf): learning visual representations without helper targets.
- [Scaffolding Minds v2](related_literature/Scaffolding_Minds_2608.19669v2.pdf): latent exploration and RL. Use v2 for implementation details; v1 is also saved.
- [VSP](related_literature/VSP_2407.01863v1.pdf): visual planning tasks and evaluation.
- [Mirage](related_literature/Mirage_2506.17218v1.pdf): related latent visual reasoning and planning setup.
