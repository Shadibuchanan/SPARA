# LIVR dataset availability and comparison protocol

Research date: 2026-10-11. This note records verified sources and archive inspection, not an implemented experiment. Recommendations were updated after the user's scope clarification; [ADR-0002](../adr/0002-reproduce-livr-reference-protocol.md) records the active decision.

## Recommendation

Use Mirage's released VSP planning data and evaluator for the first LIVR reproduction attempt and retain that evaluator for later SPARA comparisons. Preserve the source split and disclose its overlap. The user withdrew the separate SPARA scorer and additional direct-SFT experiment; filtered-subset results and generated maps are not requirements of the initial reproduction. Dataset availability does not depend on LIVR releasing its implementation.

Accepted source policy: LIVR's explicit settings take precedence. Fill relevant
unspecified experimental details from the pinned Mirage implementation, recording
each as an assumption about LIVR rather than a verified original setting. The
fallback does not replace LIVR's method with Mirage's architecture or objectives.

## Paper evidence

LIVR explicitly uses Mirage's released VSP dataset, removes helper-image supervision, and uses four latents. Its reported Qwen2.5-VL-3B results are zero-shot **6.00%**, Mirage **46.00%**, and LIVR **66.00%**. The 7B results are LIVR **77.50%** and Mirage **76.00%**; the Mirage 7B score is taken from prior reporting, whereas the 3B comparison is reproduced. Thus 77.50% is not the 3B reproduction target. [LIVR v2, §4.2](https://arxiv.org/html/2512.21218v2#S4.SS2)

For this comparison, LIVR trains the language backbone fully, freezes vision encoder/projector, uses 10 epochs per stage, AdamW at 1e-5, effective batch size 8, and separate 5% warmup/cosine schedules. The stated seed is 42. A 3B LoRA run remains an adaptation even on identical data. The paper does not identify archive hashes, an exact VSP test manifest, or detailed VSP decoding/parser settings; current released files are therefore the best verified starting point, not proof of byte-identical authors' inputs. [LIVR v2, Appendix B](https://arxiv.org/html/2512.21218v2#A2)

## Available release, pinned for reproducibility

Inspected official Mirage commit: **`53f26de2682025e146781c5b198ec93bdfe4c4d6`**. The README announces VSP training data on July 9, 2025 and test data/code on July 23, 2025, and names `train_direct.jsonl` and `test_direct.jsonl` in usage instructions. It separately mentions a 100-example spatial-reasoning sample; that is not the planning training set audited here. [Pinned README](https://github.com/UMass-Embodied-AGI/Mirage/blob/53f26de2682025e146781c5b198ec93bdfe4c4d6/README.md)

| Archive | Compressed bytes | JSONL member | Records |
|---|---:|---|---:|
| [Training archive](https://github.com/UMass-Embodied-AGI/Mirage/blob/53f26de2682025e146781c5b198ec93bdfe4c4d6/data/vsp_spatial_planning/vsp_spatial_planning.tar.gz) | 43,988,567 | `./train_direct.jsonl` | 1,000 |
| [Test archive](https://github.com/UMass-Embodied-AGI/Mirage/blob/53f26de2682025e146781c5b198ec93bdfe4c4d6/data/vsp_spatial_planning/vsp_spatial_planning_test.tar.gz) | 2,932,686 | `test_direct.jsonl` | 400 |

Archive SHA-256 digests calculated locally:

```text
training df07b7f215d1a461dc1efc27c626caff11e22d5ea8ac694e2ba7be6a0487193c
test     0272164f29dbed8b191977f7fa71cff8964b701bb84af293bf7525ada8394956
```

Direct JSONL inspection gives:

| Grid | Train | Test |
|---|---:|---:|
| 3×3 | 100 | 100 |
| 4×4 | 200 | 100 |
| 5×5 | 300 | 100 |
| 6×6 | 400 | 100 |

Neither archive contains a validation JSONL or 8×8 examples. Start/goal positions vary, so the earlier fixed-corner proposal would change the data distribution. Training records include prompt, input image, helper image, answer, map ID, and numeric map description. Test records include prompt, input image, map ID, and map description, but no answer string. All 2,000 referenced training images and 400 referenced test images exist in their respective archives. These are observations of the two pinned archives above, not inferred dataset sizes from rounded paper scores.

## Split integrity audit

Exact normalized `map_desc` comparisons, including start and goal cells, found 1,000 unique training layouts and 395 unique test layouts. Seven distinct layouts appear in both splits, affecting **8 of 400 test records**. SHA-256 comparisons of the referenced input-image file bytes independently found the same counts: seven shared hashes affecting eight test records. This establishes exact overlap in the public release; it does not establish which records LIVR's authors evaluated.

Affected test JSONL rows (one-based): **38, 62, 109, 152, 269, 278, 344, 366**. Repeated test-layout row groups: `(32,80)`, `(33,119)`, `(61,299)`, `(92,161)`, `(152,366)`. Source: the pinned training/test archives above; computed during this audit using Python `tarfile`, `json`, and `hashlib`.

The earlier research recommendation was to supplement the official 400-record result with a 392-record overlap-excluded result (or 388 unique layouts after deduplication). These are optional future diagnostics, not part of the clarified initial scope. Retain the official split and disclose overlap; silently filtering it would change the benchmark. Do not choose hyperparameters or checkpoints using test results.

## Evaluator differences that affect comparability

The released [action parser and simulator](https://github.com/UMass-Embodied-AGI/Mirage/blob/53f26de2682025e146781c5b198ec93bdfe4c4d6/src/task.py#L115) behave as follows:

The SPARA proposal in the right column is superseded. The active plan uses the reference evaluator for all compared methods.

| Behavior | Released Mirage evaluator | Superseded SPARA proposal |
|---|---|---|
| Action parsing | Uppercase the string and retain individual `U,D,R,L` characters | Accept only `UP,DOWN,LEFT,RIGHT` |
| Off-grid move | Stay in place | Stay in place |
| Hole | Immediate failure | Immediate failure |
| Goal | Check final position after the entire sequence | Succeed immediately on reaching goal |
| Simulator route cap | None | 2 × grid area |
| Optimality | No shortest-path comparison | Any successful route |

Local probes of the isolated inspected functions confirmed that on `[[1,2],[0,0]]`, `R` succeeds, `R,L` fails, and `WRONG` parses to `R` and succeeds. The nominal invalid-action branch cannot catch discarded non-action characters. Preserve these semantics in the compatibility scorer rather than silently replacing them.

The released [test entry point](https://github.com/UMass-Embodied-AGI/Mirage/blob/53f26de2682025e146781c5b198ec93bdfe4c4d6/src/test.py) extracts boxed content through `mathruler.grader.extract_boxed_content`, then calls that simulator. It seeds with 42 and generates with `do_sample=True`, temperature 0.7, top-p 0.9, and `max_new_tokens=1024`. A generation-token limit is not an action-count limit. The exact boxed-content dependency version and malformed-output behavior still need pinning; the LIVR paper does not confirm it used this exact public decoder. Treat this as the verified Mirage reference protocol, with LIVR correspondence still to verify.

## Next concrete steps

1. Pin archive hashes, input manifests, preprocessing, scorer, and generation parameters. Use only input images and final answers for LIVR training; exclude helper images and map metadata from model inputs.
2. Profile full 3B language-backbone fine-tuning on PACE-ICE, then target the published recipe. Any necessary LoRA fallback is an explicitly documented deviation.
3. Apply the approved source precedence to checkpoint selection: subsequent [reference-source research](https://github.com/Shadibuchanan/SPARA/blob/9386caaa181b0905476a630b4fa434ae8309df5f/docs/research/LIVR_REFERENCE_SETTINGS.md) establishes final checkpoints and fresh optimizer state between stages as the relevant Mirage defaults. Use these documented assumptions and keep the training set intact; LIVR's original VSP checkpoint practice remains unverified. Never tune on test results.
4. Reimplement and evaluate LIVR with the reference protocol. Use the same reference evaluator for later RL comparisons; do not introduce the withdrawn custom scorer or additional direct-SFT experiment.
5. Qualify the conclusion: matching a score on the same benchmark supports behavioral reproduction, but does not alone prove equivalent implementation. Check masks, gradients, save/reload behavior, and decoding consistency too.

The delegated audit created this note; the parent subsequently updated its recommendations to reflect the user's scope correction. Archives and inspected source copies are in the local temporary directory `spara-vsp-research`; no model weights were downloaded and no project implementation was changed.
