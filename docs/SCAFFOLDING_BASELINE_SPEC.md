# Scaffolding Minds baseline: implementation specification

Status, 2026-10-14: implemented in `src/spara/scaffolding/` and checked on CPU with
tiny random Qwen models plus the real tokenizer and VSP records. The first PACE run
(job 6257067) failed in the Latent-Prediction Phase; see [First run](#first-run-job-6257067).
Two corrected variants are implemented but **have not yet run on PACE**.

## Objective and boundary

Run [Scaffolding Minds v2](related_literature/Scaffolding_Minds_2608.19669v2.pdf)
on the LIVR baseline's benchmark so the two scores are directly comparable:
Mirage VSP (1,000 train / 400 test, 3x3 to 6x6 grids), Qwen2.5-VL-3B-Instruct at
the same pinned revision, the same reference evaluator and the same decoding
settings. The paper's own FrozenLake suite (8x8 to 32x32, 4,550 examples) and 7B
backbone are not reproduced, so its published numbers are not targets.

Unlike every other method in this repository, this baseline reads helper images
during training. That makes it a helper-supervised reference, not a SPARA
condition. See [ADR-0003](adr/0003-scaffolding-minds-helper-supervised-reference.md).

## Method as implemented

Each trace has one latent block, as in the paper's FrozenLake setting:
`<user turn with map><|im_start|>assistant` + BOT + K=4 latent slots + EOT + answer +
`<|im_end|>`. The assistant header has no trailing newline, matching the LIVR
serialization. Latent vectors are scattered into the slot positions right after
the image merge, so multimodal positions still come from token IDs.

| Paper | Implementation |
| --- | --- |
| Helper image h (App. E.1) | Value-function heatmap from `map_desc` with the paper's value-iteration code (gamma 0.85). The released route overlay is the `"released"` option. |
| Scaffolding encoder psi (§3.2, App. D.1) | A trainable copy of the VLM vision tower (668.7M parameters) plus one attention-pooling block: K learnable queries read the helper tokens and each other, followed by an MLP (about 25M parameters at d=2048). |
| Scaffolding Phase, eq. 1 | The base VLM is frozen; only psi trains, on answer cross-entropy. |
| Latent-Prediction Phase, eq. 2 | psi is frozen and z* is precomputed for all rows. The predictor pass runs `[x, BOT, e_1..e_K]`: the queries see the prefix and each other, and z is the final-norm hidden state at those positions, through an optional linear projection (variant B). The answer pass reads the teacher-forced z* and supervises BOT and the answer. Loss = 1.0·latent + 0.5·CE, where the latent term is eq. 2's squared L2 reduced by `latent_reduction` (see [variants](#latent-prediction-variants)). |
| Inference (§3.2) | Decode until BOT is sampled, run the query embeddings against the KV cache, crop them out, append z (+Δ) and EOT, and resume. |
| Scaffolding RL, eq. 3 and App. C | Two-layer mean/scale MLP heads on z. The mean starts at zero and σ at 0.05. Δ ~ N(μ, σ²) is sampled during rollouts. The block-level Gaussian log-ratio (eq. 5, in log space) is added to the per-token text ratios in the clipped GRPO objective, with KL against the frozen Stage 1 model (eq. 6). Gradients flow through z^θ. |
| Reward | Reference-evaluator route success (1/0). |

The RL objective lives in `policy.py` as tensor functions that do not depend on
where the latent prior comes from. The planned SPARA phase can reuse it with
LIVR's latents.

## Hyperparameters

From the paper (App. D.1): K=4; AdamW, weight decay 0.01, cosine schedule with
5% warmup, BF16; 3 epochs at batch 32 per Stage 1 phase, with learning rates
5e-5 (Scaffolding Phase) and 2e-5 (Latent-Prediction Phase); λ_latent 1.0 and
λ_task 0.5; GRPO with G=8, ε=0.2, learning rate 1e-6, β_KL 0.001.
Engineering settings follow the LIVR baseline: FP32 parameters with BF16
autocast, microbatch 1 with gradient accumulation, gradient clipping at 1.0,
gradient checkpointing, SDPA and seed 42. All values are in
[configs/scaffolding.json](../configs/scaffolding.json).

On 1,000 rows, 3 epochs at batch 32 is 96 optimizer steps per Stage 1 phase.
The paper's 4,550 rows gave 429. The paper's explicit epoch count is kept;
raise `epochs` to match the step count instead.

## Recorded assumptions

The paper does not specify these. They are also recorded in each run manifest.

| Topic | Choice |
| --- | --- |
| BOT/EOT embeddings | Copied from `<|vision_start|>`/`<|vision_end|>`, so the frozen VLM reads z* between familiar delimiters. Slot and query rows use N(0, 0.02), as LIVR's latents do. |
| Definition of z | The final-norm hidden state at the query positions; in variant B, passed through a zero-initialized d→d linear projection that is kept for inference. `latent_loss_normalized` (‖z−z*‖² / ‖z*‖², always on the sum scale) and `latent_cosine` are logged. Values of the first well below 1 mean the predictor is tracking z*. |
| Attention pooling | One pre-norm cross-attention block with a d→d MLP and 16 heads, sized to match the paper's ~80M at 7B. |
| VLM vision tower | Frozen in every phase, as in the LIVR baseline. |
| Heatmap rendering | One flat colour per cell on an inferno-like ramp, at the released 320×320 map size. |
| RL scale | 500 steps (the range plotted in the paper's Figure 2) × 8 prompts × G=8 rollouts on the same 1,000 training maps. Cosine schedule. One update per rollout batch (the DeepSeekMath GRPO default), so the clip only acts if `ppo_epochs` > 1. |
| RL guards | RL stops after `stop_after_zero_reward_steps` (5) consecutive steps in which every rollout fails, since zero advantages give zero gradient. `run_scaffolding.sh` starts RL only if Stage 1 scored at least `--rl-floor` (5%). |
| Rollouts | Temperature 1.0 with no top-k, top-p or repetition penalty, so sampling equals the scored policy. The recipe check rejects other settings. At most 256 sampled tokens. |
| Advantage, KL | Group std uses the unbiased estimate. KL is computed exactly over the vocabulary at each sampled-text position. |
| Never-sampled tokens | Slot, query and EOT tokens, plus image/video/vision markers, are never sampled. BOT is blocked once the single block is used. The same mask applies when scoring, so importance ratios stay exact. A trained model essentially never emits these tokens. |
| Evaluation | The LIVR protocol: seed 42, file order, temperature 0.7, top-p 0.9, inherited top-k 50 and repetition penalty 1.05. The 1,024-token limit counts sampled tokens only, so the five forced block positions are extra (HF's `generate` would count them). Sampling uses HF's own logits-processor stack. Stage 2 samples Δ (paper §4); Stage 1 uses Δ=0. |

The latent log-probability sums over K×d = 8,192 dimensions with σ≈0.05, as eq. 5
specifies. On the tiny test model, this latent term's gradient norm is about three
orders of magnitude above the text term's. The pilot measures both on the real
model. If the latent term dominates there too, review it before spending RL hours.
`latent_logprob_reduction: "mean"` is a recorded alternative, not a paper setting.

## First run (job 6257067)

One H100 ran the paper-literal recipe (sum reduction, no projection) on
2026-10-14; logs and Stage 1 outputs are in `spara_results/`.

- The Scaffolding Phase trained normally: answer loss fell from 3.09 to about
  0.5, and z* settled at a norm of about 48.
- The Latent-Prediction Phase collapsed the language model. Qwen's final RMSNorm
  weight has norm 161.6, so final-norm hidden states start with a norm of about
  160-190, while z* lives in the frozen model's input-embedding space at about 48.
  Summed over 2,048 dimensions, the latent term started at 38,437 with gradient
  norms near 4 million, against cross-entropy near 2. Every clipped update served
  the latent term: z matched z* (cosine 0.998), but answer loss rose from 2.1 to
  8.0 and ended at 1.5, against 0.31 for LIVR at the same step.
- Stage 1 scored 0/400. Every answer was unrelated text that never opened the
  latent block and ran to the 1,024-token cap (2.9 hours).
- RL then logged 106 of 500 steps, about 12.2 hours, with every reward zero and
  therefore no gradient, before the job ended. The guards above now prevent this.

The Scaffolding Phase checkpoint is unaffected and is reused by both variants.

## Latent-prediction variants

The failure follows from regressing an output-space vector onto an input-space
target, so loss weighting alone may not be enough. Both corrections depart from a
literal reading of eq. 2 and are recorded as adaptations:

| Variant | Recipe | `latent_reduction` | `latent_projection` |
| --- | --- | --- | --- |
| A: MSE | `configs/scaffolding-mse.json` | `mean` (over K and d) | none |
| B: projection + MSE (default) | `configs/scaffolding.json` | `mean` | `linear` |
| Paper-literal (failed) | set `sum` and `null` | `sum` (over d, mean over K) | none |

Variant A divides the latent term by 2,048, but the shared model must still
bend its final layer into input space. Variant B adds a 4.2M-parameter linear
map from the hidden state into z-space, trained in this phase and kept for
inference; the paper's "latent generation" does not rule it out. Its zero
initialization means z starts at 0 (`latent_loss_normalized` = 1). The language
model then receives no latent gradient until the projection has started to fit.
The Gaussian RL heads read the projected z, which is the latent prior of eq. 3.

To compare them, train each from the existing Scaffolding Phase checkpoint and
read the logs before any RL. Answer loss (`ce_loss`) should fall below about 0.5
by step 96, as LIVR's does, while `latent_loss_normalized` falls well below 1.

## Commands

The environment, data and model come from the LIVR setup (`scripts/pace/setup.sh`).

```bash
python -m spara.scaffolding.cli pilot --base-model M --output P
python -m spara.scaffolding.cli train --phase scaffold --base-model M --output S1A
python -m spara.scaffolding.cli train --phase predict --base-model M --encoder S1A/final --output S1B
python -m spara.scaffolding.cli evaluate --checkpoint S1B/final --output E1   # Stage 1 row
python -m spara.scaffolding.cli train --phase rl --checkpoint S1B/final --output S2
python -m spara.scaffolding.cli evaluate --checkpoint S2/final --output E2    # Stage 1 + RL row
```

`scripts/run_scaffolding.sh` runs these steps in order, skips completed phases
on rerun and logs each phase. `--scaffold-from DIR` reuses a completed
Scaffolding Phase, and `--phases predict,eval1` stops before RL:

```bash
S1A=~/scratch/spara/results/scaffolding-6257067/stage1-scaffold
bash scripts/run_scaffolding.sh --model M --run RA --config configs/scaffolding-mse.json \
  --scaffold-from "$S1A" --phases predict,eval1
bash scripts/run_scaffolding.sh --model M --run RB --scaffold-from "$S1A" --phases predict,eval1
bash scripts/run_scaffolding.sh --model M --run RB --scaffold-from "$S1A" --phases rl,eval2  # better variant
```

On PACE, `bash scripts/pace/submit-scaffolding.sh pilot` (or `baseline`) takes the
same account and GPU variables as the LIVR pilot, plus `SPARA_MINUTES`. Within a
phase, `--resume` continues from a periodic checkpoint. The text-only GRPO control
from the paper's Table 3 is `"variant": "text_only"` from the same Stage 1 checkpoint.

## Checks

The CPU tests (`tests/test_scaffolding_*.py`) cover:

- query-mask structure and bidirectionality, and SDPA/eager agreement;
- cached decoding through a latent block versus uncached scoring;
- grouped rollouts rescoring to their sampling log-probabilities;
- zero adjustment reproducing the Stage 1 logits exactly, while nonzero
  adjustments change only later tokens;
- a latent ratio of exactly 1 at θ_old, and finite KL gradients through suppressed tokens;
- Stage 1a updating only psi, and Stage 1b updating queries and language weights
  while the vision tower stays frozen;
- checkpoint and encoder roundtrips, and mid-epoch resume equivalence;
- the real tokenizer layout, and the reference evaluator scoring answers next to
  the structural tokens;
- the zero-initialized projection starting at z = 0, passing no latent gradient
  into the language model at first, agreeing between cached and uncached passes,
  and surviving save/load;
- RL stopping after consecutive all-zero-reward steps;
- heatmap values encoding every released shortest route exactly (0.85^(moves−1) at the start).

The pilot repeats the correctness checks on the real model and adds learning
checks on one row, including a separate check that answer loss falls (the first
run's combined loss fell while answer loss rose). It measures time and peak memory per phase, splits the RL
gradient norm into latent and text terms, and probes evaluation decoding. It then
writes `summary.json`, with GPU-hour estimates for a full run.

## Cost and storage

Job 6257067 measured the Stage 1 phases on one H100: 16 minutes for the
Scaffolding Phase and 30 minutes for the Latent-Prediction Phase, including model
loading. Its evaluation (2.9 hours) and RL (about 430 seconds per step) timings
come from a broken model that always generated to the token cap, so they overstate
a healthy run; the pilot gives better estimates. Expect three full
checkpoints per run: about 8 GB for the Scaffolding Phase (encoder and AdamW
state) and about 40 GB each for the Latent-Prediction and RL phases. Check the
scratch quota before submitting, since the LIVR run already uses part of the
300 GB allowance. The pilot writes one encoder checkpoint and one weights-only
model checkpoint (about 24 GB in total); remove them after review.

## Open items

- Run variants A and B from the job 6257067 Scaffolding Phase checkpoint (about
  30 minutes of training each, plus evaluation), then continue the better one to RL.
- Decide whether to keep the paper's 3 epochs or match its step count.
- Review the latent/text gradient balance in the RL logs.
