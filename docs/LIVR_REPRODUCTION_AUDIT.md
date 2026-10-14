# Audit of the LIVR score discrepancy

Date: 2026-10-11. Scope: CPU-only audit of saved run `6189457`; no new training,
generation, model changes or production scoring changes.

## Finding

The recorded **323/400 (80.75%)** result survives the checks below. No scoring or
answer-leakage bug was found in the inspected paths. The reason for the
**14.75-percentage-point difference** from LIVR's reported 3B result of 66%
remains unknown. This is a reproduction attempt with a score discrepancy, not a
verified one-for-one reproduction or demonstrated improvement over LIVR.

A higher score is not itself a software failure. Without the authors' code,
checkpoint and exact evaluation manifest, this audit cannot reproduce a causal
failure behind the difference. It uses falsifiable checks for specific
inflation/leakage mechanisms instead, and makes no red-to-green bug-fix claim.

## Checks and outcomes

| Concern | Test | Result |
| --- | --- | --- |
| Wrong split or altered data/images | Recheck archive, JSONL and input-image hashes; compare against both training manifests and evaluation manifest | Original 1,000/400 split and images match |
| Counting or cached result error | Rescore all saved responses; verify row order, map IDs and summary | 323/400, all outcomes match |
| Simulator false positives | Independently walk each extracted route on its map | All 400 outcomes agree, including 323 valid successes |
| Parser turns malformed text into a lucky route | Inspect all extracted routes and compare whole direction words to reference character extraction | All 400 use ordinary direction words/letters and separators; action sequences match |
| Prompt text or a supplied answer is being scored | Reconstruct every real processor input and compare it to the saved raw-response prefix | All 400 prefixes match; scored answer is exactly the generated continuation |
| Helper images, answer fields or simulator metadata enter inference | Replace those fields with poison values/nonexistent helper paths and re-encode each record | All input tensors remain identical for all 400 examples; no labels supplied; released test records have no answer field |
| Multiple attempts per example | Inspect saved records, generation manifest and evaluation loop | One returned sequence per test example; all 400 rows appear once |
| Exact train/test overlap inflates the result | Remove the eight overlapping rows for a diagnostic only | 315/392 = 80.36%; the overlap does not account for the gap |
| Stage 1 mask leaks the image through prompt positions | Rerun existing image-intervention and first-answer mask test | Passed on the tiny model through the real implementation |
| Training serialization includes helper metadata or mismatched answer boundary | Rerun existing real-processor serialization test | Passed |

The source evaluator accepts off-grid no-ops and does not require shortest
paths. Only two successful saved routes are nonshortest; both contain an off-grid
move. Even counting those two as failures would give 321/400 (80.25%). This is
only a sensitivity diagnostic, not a proposed replacement metric or evidence
that LIVR used a stricter criterion. The official result remains 323/400.

The independent route walker was also checked against the reference on 1,000
seeded random routes. A deliberately flipped outcome in a temporary copy caused
the audit to fail, without writing a success report; original outputs were not
modified. Two targeted architectural/serialization tests passed. The initial
audit-script comparison needed JSON key normalization for per-grid integer keys;
this was a diagnostic-harness issue, not a changed dataset or baseline bug.

## What remains unresolved

The [LIVR paper, Section 4.2 and Appendix B](https://arxiv.org/html/2512.21218v2)
explicitly supports the 3B comparison, four latents, full language fine-tuning,
frozen vision/projector, 10+10 epochs, AdamW recipe and effective batch size eight.
Those reported settings match this run. The paper does not provide enough
information to establish identical implementation and execution.

Potential sources of a difference, **not established explanations**, include:

- Exact model/tokenizer/software revisions, image preprocessing and numerical
  precision. We pin stock Transformers and use FP32 trainable parameters with
  BF16 autocast; the authors' exact corresponding implementation is unavailable.
- Exact latent placement, assistant-header formatting, answer tokenization,
  embedding initialization and treatment of existing embedding rows.
- Decoding and checkpoint-selection details. We use the documented Mirage
  fallback for omissions, final stage checkpoints and a fresh Stage 2 optimizer.
- Stochastic training/generation and hardware differences. These were not
  isolated or measured here, and are not sufficient grounds to dismiss the gap.
- The exact original test-file manifest, which LIVR does not identify by hash.

The [pinned Mirage evaluation entry point](https://github.com/UMass-Embodied-AGI/Mirage/blob/53f26de2682025e146781c5b198ec93bdfe4c4d6/src/test.py)
uses sampling with temperature 0.7, top-p 0.9 and a 1,024-token limit. Our saved
manifest also records inherited top-k 50 and repetition penalty 1.05. These are
known settings of our run; they are not verified LIVR author settings.

Do not change the evaluator or tune against the test set to force a 66% result.
Keep the existing checkpoint and outputs fixed. Further causal diagnosis would
require resolving these paper-to-code assumptions, obtaining author artifacts,
or separately designed controlled experiments. This audit did not run those
experiments and does not establish across-seed variance or whether trained-model
performance depends on the latent tokens.

## Paper review addendum, 2026-10-14

LIVR v2 (§3, §4.2 and Appendix B) was reread against the code. Every setting the
paper states for the VSP comparison is implemented as stated:

| Paper | Implementation |
| --- | --- |
| K=4 distinct latent tokens, randomly initialized, with trainable embedding rows (§3.2, §4.2) | `LATENT_TOKENS` and `initialize_latents` in `src/spara/model.py` |
| Latents appended after the prompt; the model never generates them (§3.2, §4.3.3) | After the user turn, before the assistant header; supplied at inference |
| Stage 1: answer and prompt tokens cannot attend to image tokens; latents read the image and the prompt (§3.2, Table 6a default) | `bottleneck_mask` |
| Stage 2: standard mask, same loss (§3.2) | `livr_stage == 2` uses stock attention |
| Per-sequence mean NLL over answer tokens only (§3.2) | Prefix labels −100; per-example mean, averaged over the accumulated examples |
| Full language-backbone fine-tuning; vision encoder and projector frozen (App. B, Mirage comparison) | `freeze_vision` freezes `visual.*`, which includes the merger/projector |
| AdamW 1e-5, weight decay 0.01, betas (0.9, 0.999), ε 1e-8; batch 1 × 8 accumulation; 5% warmup and cosine per stage; 10+10 epochs; seed 42; single GPU (App. B) | `configs/livr.json` and `src/spara/training.py` |

The LoRA recipe in §3.3 and Appendix B (rank 16, LR 1e-4, only latent rows
unfrozen) belongs to the BLINK data-matched experiments, not to VSP. On
2026-10-14 the project page still linked only the paper. Chat-template
boundaries, end-token supervision, decoding and checkpoint selection therefore
remain recorded assumptions. Their Mirage-derived values were rechecked against
the pinned Mirage code, and they match: prompts end with `<|im_start|>assistant`;
temperature 0.7, top-p 0.9, 1,024 new tokens and seed 42; and per-example loss
averaging (the Qwen forward in Mirage's bundled Transformers fork accepts no loss
kwargs).

### Numerical precision is the most concrete candidate for the gap

The paper does not state its precision, but three pieces of evidence point away
from this implementation's choice:

- **The authors' memory configuration differed from ours.** Appendix B says
  every experiment ran single-GPU on RTX 6000 Ada (48 GB) cards. Before any
  activations, this implementation already needs about 48.5 GiB: FP32 weights,
  gradients and AdamW states for the 3.09B trainable parameters (46.0 GiB), plus
  the frozen FP32 vision tower (2.5 GiB). It peaked at 55.2 GiB allocated
  (pilot report).
- **Mirage's released code is pure BF16.** The authors reproduced Mirage on
  that hardware (§4.2). Mirage's released code, pinned here at `53f26de`, loads
  Qwen in `torch.bfloat16` and trains with `bf16=True` and `adamw_torch_fused`
  ([`src/main.py`](https://github.com/UMass-Embodied-AGI/Mirage/blob/53f26de2682025e146781c5b198ec93bdfe4c4d6/src/main.py),
  lines 55, 57 and 188–189). Weights and Adam states therefore stay in BF16, with
  no FP32 master copy. ADR-0002's fallback policy points to that setting; the
  implementation instead recorded FP32 parameters with BF16 autocast as an
  engineering choice.
- **Pure BF16 rounds most of these updates away.** A BF16 value cannot move by
  less than half its spacing. `scripts/probe_bf16_updates.py` applied one
  Adam-sized step (w − lr·sign(g)) to sampled real weights. It weighted the result
  by parameter count over all 3.09B language parameters. At the peak learning rate
  of 1e-5, **86.9%** of weights were unchanged; at 1e-6 (early warmup and the end
  of the cosine decay), **98.3%**. Adam steps can exceed the learning rate (up to
  about 3× with these betas), but even a 3e-5 step leaves **74.5%** unchanged.
  The sample was the leading 4 MB of every projection, norm and q-bias tensor in
  layers 0, 9, 18, 27 and 35, plus the leading rows of the tied embedding. FP32
  master weights accumulate these updates.

If the authors trained this way, most fine-tuning updates were lost. That would
lower their score relative to ours, which is the direction of the gap. **This
is not established.** The authors could have kept FP32 weights with an 8-bit or
offloaded optimizer instead. Their 7B LIVR run would not fit in 48 GB even in
pure BF16, so the hardware statement may be approximate. A controlled
diagnostic could test the hypothesis directly: rerun the unchanged recipe with
BF16 parameters and fused AdamW. Keep run `6189457` as the baseline either way;
a BF16 rerun would be an additional, separately budgeted diagnostic.

## Reproduce the audit

```powershell
.venv/Scripts/python scripts/audit_livr_result.py --run results/pace-baseline-6189457/results/baseline-6189457 --processor results/pace-baseline-6189457/results/baseline-6189457/stage1/final --output results/pace-baseline-6189457/discrepancy-audit.json
```

The processor needs only local tokenizer/preprocessing files, not model loading.
The structured output is `results/pace-baseline-6189457/discrepancy-audit.json`.
The script is forensic tooling; the single production reference evaluator and
reported benchmark remain unchanged.

The precision probe in the addendum downloads only byte ranges of the pinned
Qwen weights and loads no model:

```bash
python scripts/probe_bf16_updates.py
```
