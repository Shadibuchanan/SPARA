# Scaffolding Minds baseline: implementation specification

Status, 2026-10-27: implemented in `src/spara/scaffolding/` and checked on CPU with
tiny random Qwen models plus the real tokenizer and VSP records. The first PACE run
(job 6257067) failed in the Latent-Prediction Phase; see [First run](#first-run-job-6257067).
Of the two corrected variants, B scored 223/400 (55.75%) at Stage 1. Scaffolding RL at
the paper's settings raised it to 242/400 (60.5%), or 237/400 with Δ = 0, but its latent
term carried almost all of the gradient and its σ = 0.05 adjustment rarely changed a
route. A rerun with mean reduction and σ_init 1.0 reached **270/400 (67.5%)**, or
264/400 (66.0%) with Δ = 0. Its Gaussian heads barely moved, so the gain is in the
language-model weights, not in a learned latent adjustment. A 20x head learning rate
moved the heads but scored the same (267/400 with Δ = 0). See [Results](#results-2026-10-27).

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

## Results, 2026-10-27

### Stage 1 variants (job 6273886)

Both variants trained from the job 6257067 Scaffolding Phase checkpoint on one H100
(1 h 16 min for both, including evaluation).

| Variant | Overall | 3 x 3 | 4 x 4 | 5 x 5 | 6 x 6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A: MSE | 191/400 (47.75%) | 60 | 57 | 45 | 29 |
| B: projection + MSE | 223/400 (55.75%) | 78 | 68 | 51 | 26 |
| LIVR baseline (6189457), for reference | 323/400 (80.75%) | 92 | 91 | 81 | 59 |

Variant B fixed the collapse: answer loss fell from 2.11 to 0.17 and the predicted
latent reached cosine 0.9999 with z*. It is the Stage 1 checkpoint for RL. The gap to
LIVR is not a verdict on helper supervision: the language model trained for 3 epochs
(96 updates at batch 32) against LIVR's 20 epochs (2,500 updates at batch 8). Its 177
failures are 105 holes and 72 routes that stop short, against LIVR's 71 and 6.

### Scaffolding RL at the paper's settings (jobs 6523751 and 6623076)

RL ran 500 steps from variant B with `configs/scaffolding.json` (sum reduction,
σ_init 0.05) in 3 h 51 min on one H100; evaluation followed as job 6623076.

| | Overall | 3 x 3 | 4 x 4 | 5 x 5 | 6 x 6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 1 (Δ = 0) | 223/400 (55.75%) | 78 | 68 | 51 | 26 |
| Stage 1 + RL (Δ sampled, paper protocol) | 242/400 (60.5%) | 80 | 70 | 59 | 33 |
| Stage 1 + RL (Δ = 0, job 6623192) | 237/400 (59.25%) | 78 | 73 | 56 | 30 |

Most of the gain comes from the RL weights, not from sampling Δ at evaluation. With
Δ = 0 the RL checkpoint gains 26 maps and loses 12 against Stage 1 (exact McNemar
p ≈ 0.03). Its two evaluations disagree on 47 maps, 26 to 21, which is within sampling
variance; two identical LIVR runs disagreed on 31. The training logs do not show the gain:

| Steps | Reward | Pass@8 | Block opened | KL | \|μ\| | σ | Latent-term gradient norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-50 | 0.516 | 0.80 | 0.956 | 0.001 | 0.0013 | 0.0500 | ~202,000 |
| 201-250 | 0.498 | 0.80 | 0.925 | 0.172 | 0.0079 | 0.0500 | ~26,000 |
| 451-500 | 0.476 | 0.76 | 0.909 | 0.225 | 0.0091 | 0.0500 | ~26,000 |

- The total gradient norm equals the latent term's alone to within about 0.01%, as
  the [assumptions](#recorded-assumptions) warned. Text-only RL on LIVR, with the
  same text term, had a median total norm of about 1.
- With `max_grad_norm` 1.0, clipping scaled each update by about 1/26,000 (about
  1/200,000 in the first 50 steps). The text term reached the weights at that fraction
  of its size, so every update followed the latent term.
- KL to the reference rose to about 0.25 (text-only RL: 0.004-0.04) while training
  reward did not improve, and fewer rollouts opened the latent block.
- σ never left 0.05 and \|μ\| reached 0.009.
- With one update per rollout batch, every ratio is 1, so clipping never acted.

Reward was measured on training maps at temperature 1.0 and the test score at 0.7, so
the KL drift may have sharpened the policy without raising temperature-1.0 reward; the
logs cannot say which weights moved for which term.

### Latent-sensitivity probe (job 6623151)

`probe` decodes 64 training maps greedily from the variant B checkpoint, once with
Δ = 0 and four times per σ with isotropic Δ, and counts route changes. Greedy decoding
means every change comes from Δ. It took 85 seconds.

| σ | Routes changed | Maps with any change | Success | Gained / lost | Median first-token KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | | | 36/64 (56.3%) | | |
| 0.05 (paper) | 2.7% | 2 | 57.8% | 4 / 0 | 6e-10 |
| 0.1 | 2.7% | 4 | 57.8% | 4 / 0 | 7e-10 |
| 0.2 | 4.3% | 5 | 59.4% | 8 / 0 | 1e-9 |
| 0.5 | 6.3% | 7 | 59.4% | 8 / 0 | 2e-8 |
| 1.0 | 11.7% | 12 | 60.9% | 13 / 1 | 4e-7 |
| 2.0 | 21.1% | 15 | 62.1% | 15 / 0 | 0.04 |

The latent's per-dimension RMS is 1.07, so the paper's σ is about 5% of its scale and
almost never changes what the model writes. Exploration that cannot change a route
cannot earn reward. Isotropic noise also raised greedy success at every σ here, so
any Stage 2 score with sampled Δ includes a noise effect as well as learning.

### RL rerun (job 6623191, `configs/scaffolding-rlfix.json`)

Three settings change; everything else matches the first run. Each departs from the
paper and is recorded as an adaptation, like variants A and B.

| Setting | Paper | Rerun | Reason |
| --- | --- | --- | --- |
| `latent_logprob_reduction` | `sum` | `mean` | Divides the latent term by K x d = 8,192, from about 25,000 to about 3 at σ = 0.05: the same order as the text term |
| `sigma_init` | 0.05 | 1.0 | The smallest probed σ that changed at least 10% of greedy routes without lowering success by more than 5 points; the rule was fixed before the probe ran. At 10%, about 57% of groups of 8 contain a route changed by Δ |
| `log_sigma_bounds` | [-9.21, 0] | [-9.21, 1.3863] | σ_init sat on the old ceiling of 1; the new ceiling is 4 |

The learning rate of the Gaussian heads was left unchanged, on the reasoning that their
small movement in the first run fit a gradient with no consistent direction, since Δ
rarely changed a route, rather than a learning-rate limit. The rerun did not bear this
out (below). It shares variant B's Stage 1 checkpoint and evaluation through links, so
the first run's directory is untouched. It ran from the PACE checkout at 4b41fc0 with
this rerun's files copied in, next to uncommitted edits of unknown authorship already
there, including step-checkpoint pruning in `training.py`.

RL took 3 h 42 min for 500 steps (3 h 55 min with the sampled-Δ evaluation); the Δ = 0
evaluation was job 6625890.

| | Overall | 3 x 3 | 4 x 4 | 5 x 5 | 6 x 6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 1 (Δ = 0) | 223/400 (55.75%) | 78 | 68 | 51 | 26 |
| Paper-settings RL (Δ = 0) | 237/400 (59.25%) | 78 | 73 | 56 | 30 |
| Rerun (Δ sampled, paper protocol) | 270/400 (67.5%) | 83 | 77 | 63 | 47 |
| Rerun (Δ = 0) | 264/400 (66.0%) | 84 | 77 | 59 | 44 |

With Δ = 0 the rerun gains 55 maps and loses 14 against Stage 1 (exact McNemar
p ≈ 7e-7), and gains 46 and loses 19 against the paper-settings RL at Δ = 0
(p ≈ 0.001). Its two evaluations disagree on 22 maps, 14 to 8 (p ≈ 0.29). Unlike the
first run's, the training logs show the gain (gradient norms are window medians):

| Steps | Reward | Pass@8 | Block opened | KL | \|μ\| | σ range | Latent-term gradient norm | Total gradient norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-50 | 0.527 | 0.77 | 0.954 | 0.032 | 0.0015 | 0.98-1.01 | 0.16 | 3.3 |
| 201-250 | 0.577 | 0.69 | 0.939 | 0.128 | 0.0064 | 0.97-1.04 | 0.11 | 4.8 |
| 451-500 | 0.668 | 0.75 | 0.942 | 0.103 | 0.0068 | 0.97-1.04 | 0.10 | 3.4 |

- The latent term carries about 3% of the total norm (median ratio 0.026), and
  clipping scales updates by about 1/4, so the text term reaches the weights.
- Reward peaked at 0.692 in steps 351-400. KL settled near 0.1, below the first run's
  0.25, and about 94% of rollouts still opened the latent block.
- The Gaussian heads still barely moved: \|μ\| reached 0.007 against a latent RMS of
  1.07, and σ stayed within 0.97-1.04. Adam moves each parameter by at most about the
  learning rate per step, so 500 steps at 1e-6 cap each head weight's change near
  5e-4. This run cannot tell that cap from a gradient with no consistent direction.
- Sampled Δ at evaluation is therefore close to isotropic noise at σ = 1, which the
  probe found raises greedy success by itself. The Δ = 0 row excludes that effect.

So the rerun's gain is in the language-model weights, trained with the latent block
under σ = 1 noise; it is not a learned latent adjustment. Telling whether the noise
helped, or only the unswamped text term, needs text-only GRPO from the same Stage 1
checkpoint (`"variant": "text_only"`).

### Head learning rate (job 6627337, `configs/scaffolding-rlfix-headlr.json`)

The rerun's recipe plus `rl.head_learning_rate` 2e-5, which gives the Gaussian heads
their own AdamW groups; the language model stays at 1e-6. Head movement scales about
linearly with an Adam learning rate, so the rerun's movement predicted per-dimension
σ of about 0.74-1.46 at 1e-5, 0.55-2.15 at 2e-5 and 0.22-6.7 at 5e-5. Among the paper's
own rates, 2e-5 is the largest whose prediction stays inside `log_sigma_bounds`, and it
matches the σ movement the paper reports. RL took 3 h 50 min with the sampled-Δ
evaluation; the Δ = 0 and Δ = μ evaluations were jobs 6636987 and 6636988.

| Steps | Reward | Pass@8 | KL | \|μ\| | Mean σ | σ range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-50 | 0.516 | 0.75 | 0.052 | 0.033 | 1.000 | 0.69-1.46 |
| 201-250 | 0.581 | 0.68 | 0.153 | 0.253 | 1.004 | 0.65-1.48 |
| 451-500 | 0.676 | 0.75 | 0.095 | 0.308 | 1.006 | 0.64-1.51 |

| | Overall | 3 x 3 | 4 x 4 | 5 x 5 | 6 x 6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Head rate 2e-5 (Δ sampled) | 259/400 (64.75%) | 82 | 75 | 60 | 42 |
| Head rate 2e-5 (Δ = μ) | 266/400 (66.5%) | 84 | 74 | 63 | 45 |
| Head rate 2e-5 (Δ = 0) | 267/400 (66.75%) | 85 | 75 | 63 | 44 |
| Rerun, for reference (Δ = 0) | 264/400 (66.0%) | 84 | 77 | 59 | 44 |

- The heads moved: \|μ\| reached 0.31 against the rerun's 0.007, and per-dimension σ
  spread to 0.55-1.77. Mean σ stayed at 1.00-1.01, so σ spread in both directions
  without a net change in exploration width, and \|μ\| stopped growing once the
  cosine schedule decayed after step 300.
- The learned mean does almost nothing: Δ = μ and Δ = 0 differ on 7 maps, 3 to 4.
- Against the rerun at Δ = 0 it gains 24 maps and loses 21 (p ≈ 0.77). Two RL runs
  that differ only in head dynamics therefore land within noise of each other, at
  264-267, and the heads' extra movement bought no test accuracy.

A higher head rate makes the heads move but not learn a useful adjustment, so the
evidence now favours an incoherent latent gradient over a learning-rate limit.

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

The probe, an RL rerun with another recipe, and evaluation with a chosen latent mode
start from a completed Stage 1 run (`SPARA_SOURCE_RUN`) through
`scripts/pace/scaffolding-rl-fix.sbatch`; its header lists the steps and outputs:

```bash
python -m spara.scaffolding.cli probe --checkpoint S1B/final --output P   # --sigmas, --rows, --draws
python -m spara.scaffolding.cli evaluate --checkpoint S2/final --evaluation-latent none --output E
sbatch [resources] --export=ALL,SPARA_STEP=rl,SPARA_CONFIG=configs/scaffolding-rlfix.json,SPARA_SOURCE_RUN=RB \
  scripts/pace/scaffolding-rl-fix.sbatch
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
a healthy run. A healthy RL run (job 6523751) took about 26 seconds per step, 3 h 51 min
for 500 steps, and its evaluation 5.5 minutes (job 6623076). The rerun (job 6623191)
took about 27 seconds per step, and its Δ = 0 evaluation 3 minutes. Its run directory
holds 38 GB, which left scratch at 259.6 of 300 GB (86.5%). Of each RL checkpoint's
40 GB, 25 GB is optimizer state that only resuming reads; it was removed from the
three finished RL runs, and variant A's Stage 1 checkpoint was deleted, leaving 139.4
GB in use on 2026-10-28. On 2026-11-08 the final weights of the paper-settings RL
run (`scaffolding-linear-6273886/stage2-rl`) and the head-rate run
(`scaffolding-headlr-2e5`) were deleted, along with the optimizer state of the two
finished Stage 1 checkpoints. Their evaluations and logs remain. The rerun's weights
(job 6623191) and the Stage 1 weights of the Scaffolding Phase and variant B are
kept. Expect three full
checkpoints per run: about 8 GB for the Scaffolding Phase (encoder and AdamW
state) and about 40 GB each for the Latent-Prediction and RL phases. Check the
scratch quota before submitting, since the LIVR run already uses part of the
300 GB allowance. The pilot writes one encoder checkpoint and one weights-only
model checkpoint (about 24 GB in total); remove them after review.

## Open items

- Choose the Scaffolding RL row to report. The rerun (264/400 with Δ = 0) departs from
  the paper in three recorded settings; the paper-settings run (237/400) is the literal
  reproduction.
- The Gaussian heads did not learn a useful adjustment in any run, including at a head
  learning rate of 2e-5. Text-only GRPO from variant B would show how much of the
  rerun's gain needs the latent block at all.
- Decide whether to keep the paper's 3 epochs or match its step count.
- Condition C reuses this RL objective with LIVR's latents. Probe LIVR's latent scale
  before choosing its σ, and use the mean reduction; the paper's σ = 0.05 and sum
  reduction are not safe defaults.
