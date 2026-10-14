# Scaffolding Minds baseline: implementation specification

Status, 2026-10-13: implemented in `src/spara/scaffolding/` and checked on CPU with
tiny random Qwen models plus the real tokenizer and VSP records. **It has not run on
PACE.** No GPU budget has been agreed for it, and no job has been submitted.

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
| Latent-Prediction Phase, eq. 2 | psi is frozen and z* is precomputed for all rows. The predictor pass runs `[x, BOT, e_1..e_K]`: the queries see the prefix and each other, and z is the final-norm hidden state at those positions. The answer pass reads the teacher-forced z* and supervises BOT and the answer. Loss = 1.0·(1/K)Σ‖z−z*‖² + 0.5·CE. |
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
| Definition of z | The final-norm hidden state. Its scale is constrained by RMSNorm, so `latent_loss_normalized` (loss / mean‖z*‖²) is logged. Values well below 1 mean the predictor is tracking z*. |
| Attention pooling | One pre-norm cross-attention block with a d→d MLP and 16 heads, sized to match the paper's ~80M at 7B. |
| VLM vision tower | Frozen in every phase, as in the LIVR baseline. |
| Heatmap rendering | One flat colour per cell on an inferno-like ramp, at the released 320×320 map size. |
| RL scale | 500 steps (the range plotted in the paper's Figure 2) × 8 prompts × G=8 rollouts on the same 1,000 training maps. Cosine schedule. One update per rollout batch (the DeepSeekMath GRPO default), so the clip only acts if `ppo_epochs` > 1. |
| Rollouts | Temperature 1.0 with no top-k, top-p or repetition penalty, so sampling equals the scored policy. The recipe check rejects other settings. At most 256 sampled tokens. |
| Advantage, KL | Group std uses the unbiased estimate. KL is computed exactly over the vocabulary at each sampled-text position. |
| Never-sampled tokens | Slot, query and EOT tokens, plus image/video/vision markers, are never sampled. BOT is blocked once the single block is used. The same mask applies when scoring, so importance ratios stay exact. A trained model essentially never emits these tokens. |
| Evaluation | The LIVR protocol: seed 42, file order, temperature 0.7, top-p 0.9, inherited top-k 50 and repetition penalty 1.05. The 1,024-token limit counts sampled tokens only, so the five forced block positions are extra (HF's `generate` would count them). Sampling uses HF's own logits-processor stack. Stage 2 samples Δ (paper §4); Stage 1 uses Δ=0. |

The latent log-probability sums over K×d = 8,192 dimensions with σ≈0.05, as eq. 5
specifies. On the tiny test model, this latent term's gradient norm is about three
orders of magnitude above the text term's. The pilot measures both on the real
model. If the latent term dominates there too, review it before spending RL hours.
`latent_logprob_reduction: "mean"` is a recorded alternative, not a paper setting.

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
on rerun and logs each phase. On PACE, `bash scripts/pace/submit-scaffolding.sh pilot` (or `baseline`) takes the
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
- heatmap values encoding every released shortest route exactly (0.85^(moves−1) at the start).

The pilot repeats the correctness checks on the real model and adds learning
checks on one row. It measures time and peak memory per phase, splits the RL
gradient norm into latent and text terms, and probes evaluation decoding. It then
writes `summary.json`, with GPU-hour estimates for a full run.

## Cost and storage

Not yet measured; the pilot produces the estimates. Expect three full
checkpoints per run: about 8 GB for the Scaffolding Phase (encoder and AdamW
state) and about 40 GB each for the Latent-Prediction and RL phases. Check the
scratch quota before submitting, since the LIVR run already uses part of the
300 GB allowance. The pilot writes one encoder checkpoint and one weights-only
model checkpoint (about 24 GB in total); remove them after review.

## Open items

- Agree a pilot budget, then a full-run budget from the pilot's estimates.
- Decide whether to keep the paper's 3 epochs or match its step count.
- Review the latent/text gradient balance and `latent_loss_normalized` in the pilot.
