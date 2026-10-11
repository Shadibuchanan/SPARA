# LIVR planning baseline: implementation handoff

Status: accepted by the user, 2026-10-11. Scope, source precedence,
compute access and the pilot ceiling are agreed. This document consolidates those
decisions and the engineering checks needed before scaling. Implementation and
pilot outcomes are recorded in [the pilot report](LIVR_PILOT_REPORT.md); full
baseline training and evaluation completed as PACE job `6189457` within the
approved four-GPU-hour ceiling, scoring 323/400 (80.75%). Final backup is still
in progress; see [the run record](LIVR_BASELINE_RESULTS.md).

## Objective and boundary

Reimplement the documented Qwen2.5-VL-3B LIVR VSP planning experiment as closely
as available evidence permits. Use the released Mirage VSP data, full language-model
fine-tuning and one reference evaluator. Preserve the released split and disclose
its overlap. Report the difference from the published 66%; reaching that score
alone is not proof of implementation correctness.

The initial phase excludes a direct-SFT comparison, custom scoring, new maps,
helper-image targets, and latent RL. Later SPARA comparisons reuse the reference
evaluator and LIVR checkpoint.

Sources:

- [Scope and source policy](adr/0002-reproduce-livr-reference-protocol.md).
- [Dataset/evaluator audit](research/LIVR_DATASET_COMPARABILITY.md).
- [Pinned reference settings](https://github.com/Shadibuchanan/SPARA/blob/9386caaa181b0905476a630b4fa434ae8309df5f/docs/research/LIVR_REFERENCE_SETTINGS.md).
- [PACE research](https://github.com/Shadibuchanan/SPARA/blob/d29e74a6c4ef7bd7fccf783a1f5b9e0a7119638a/docs/research/LIVR_PACE_REQUIREMENTS.md).
- [Verified account and execution context](agents/pace-ice.md).

## Source precedence and reproducibility

Explicit LIVR settings take precedence. Fill relevant omissions from Mirage
commit `53f26de2682025e146781c5b198ec93bdfe4c4d6`, recording each as an assumption
about LIVR. Do not inherit Mirage's different latent architecture or losses.
Pin upstream Qwen model/tokenizer/processor revisions and software before the
pilot. Record engineering choices that neither source specifies separately;
material experimental changes reopen the scope decision.

Each run manifest records source/data hashes, revisions, dependencies, settings,
seed, added token IDs, trainable parameters, preprocessing, precision/backend,
hardware/job ID, checkpoint lineage, and effective generation configuration.

## Data and evaluation

- Preserve the pinned 1,000 training rows and 400 test rows, including their text
  and order. Verify archive hashes and retain immutable source files.
- Use released input images, questions and boxed answers. Exclude helper images
  and simulator map metadata from model inputs; retain metadata for scoring.
- Preserve source image placement and processor behavior. Record resolved sizes
  and visual-token counts; do not silently shrink images or truncate answers.
- Use the pinned reference simulator and boxed-answer extraction path. Pin and
  check the previously unversioned extractor dependency. Preserve character
  filtering, off-grid no-ops, immediate hole failure and final-position goal check.
  Add no action cap or stop-on-first-goal rule.
- Apply the relevant Mirage generation defaults: seed 42, file-order evaluation,
  one example at a time, sampling enabled, temperature 0.7, top-p 0.9, and 1,024
  maximum new tokens. Save inherited generation defaults too.
- Save raw responses, extracted answers, scores, failure information and map IDs.
  Report overall/per-grid results on the original split. Do not choose checkpoints
  or tune hyperparameters using test outcomes.

## LIVR model and training

- Qwen2.5-VL-3B-Instruct; four distinct added latent tokens with random embeddings;
  frozen vision encoder/projector; full language-backbone fine-tuning.
- Put latents after the prompt and before the assistant prefix/answer. Fix exact
  chat boundaries and answer/end-token labels in a token-level fixture and record
  assumptions. The first supervised answer prediction must come from an image-blocked
  assistant-prefix position, not directly from the last image-reading latent.
- Stage 1 retains causality/padding and blocks every nonlatent prompt, control,
  assistant-prefix and answer query from image keys. Latents retain access to the
  image and preceding prompt. Stage 2 restores ordinary causal attention and keeps
  the latents. Use answer-only next-token loss without auxiliary visual losses.
- Train 10 epochs per stage with AdamW LR 1e-5, weight decay 0.01, betas (0.9, 0.999),
  epsilon 1e-8, microbatch one and eight accumulation steps on one GPU. Each stage
  has its own 5% warmup/cosine schedule; seed 42.
- Use final Stage 1 model weights and a fresh optimizer/trainer for Stage 2; report
  the final Stage 2 checkpoint. These are approved Mirage-derived assumptions;
  retain the complete training set rather than adding a checkpoint-selection split.
- Begin with a verified full-mask attention backend. Compute multimodal positional
  indices from original sequence/padding/image-grid information; the inspected Qwen
  code bypasses automatic RoPE calculation with 4-D masks. Check the actual pinned
  upstream version rather than assuming equivalence to Mirage's modified fork.
- Record embedding initialization, old-embedding/output-head trainability, EOS
  supervision, precision and clipping. Framework defaults are implementation
  assumptions, not recovered facts about LIVR's original run.

## Implementation sequence

1. Establish pinned data, reference evaluation, environment and run manifests.
2. Implement serialization, latent embeddings, attention masks and trainability.
3. Pass the correctness checks below and a tiny-subset learning check.
4. Profile both stages on PACE-ICE within the agreed ceiling.
5. Report measured costs and propose the full-run resources.
6. Once those resources are settled, run both stages and report the benchmark result.

Keep data loading, reference evaluation, model/mask integration and training
configuration separable so each can be checked without a full experiment.

## PACE-ICE workflow

The agent may operate through the user's configured `pace-ice` alias. Parameterize
SSH target, remote root, Slurm account/QoS and GPU request for teammates. Use remote
`$HOME/scratch/SPARA`; `logs/` and `results/` already exist. Add run directories,
checkpoints and caches there as needed.

Use the login node for lightweight coordination and Slurm for GPU work. Choose
available, permitted, compatible hardware with adequate memory instead of fixing
one model. Inspect the allocated device; broad resource names can hide different
memory sizes. The first pilot is limited to one GPU and one allocated GPU-hour
after setup, including consumed time across retries. The subsequently approved
full run had a separate four-GPU-hour ceiling and completed in 5,664 GPU-seconds.

The pilot covers both attention stages, backward/optimizer work, save/reload and
short generation. Measure peak GPU/host memory, typical and longest-example costs,
step time, checkpoint size/time and runtime settings. Extrapolate full-training
cost from these measurements; inference alone does not establish training feasibility.
Preserve objective and effective-batch semantics during resource optimization.
Report inability to fit rather than silently changing the model, data or method.

## Correctness and completion checks

- Data matches the pinned records and supplies no helpers or simulator metadata to
  the model. Evaluator fixtures agree with the source, including malformed answers
  and routes that cross the goal and continue.
- Mask tests establish causality, padding, first-answer handling and exclusion of
  both direct image access and prompt-mediated image leakage in Stage 1. Stage 2
  restores the normal path.
- A training step updates intended language parameters and added embeddings, while
  frozen vision/projector parameters remain unchanged.
- Save/reload preserves token IDs, embeddings, configuration and predictions. A
  stage transition resets optimizer state; within-stage resume preserves it.
  Cached/uncached next-token logits agree within a declared numerical tolerance
  on fixed inputs, checked separately from stochastic decoding.
- A tiny training subset exhibits clear learning through the actual LIVR pipeline.
  Full-scale spending follows correctness evidence and a successful resource pilot.
- Final artifacts contain configurations, per-example outputs, reference scores,
  per-grid counts, resource measurements and deviations. Results below 66% are
  reported and investigated without tuning on test data or changing the evaluator.

Exact original environment/model revisions and LIVR's VSP test manifest remain
unverified. This handoff defines a reproducible attempt, not a guarantee of the
published score.
