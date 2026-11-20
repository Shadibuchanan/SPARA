# V* results: three arms trained from base on Visual-CoT (2026-11-18)

**Result:** no trained arm beats the untuned Qwen2.5-VL-3B on V*. Direct SFT is level with it; the two
latent arms (LIVR, and global + patch latents) are 12 to 16 questions lower, and do not differ from each
other. Later ablations (below) trace the drop to the Stage 1 bottleneck: LIVR's latents trained without
it are level with the base model. This is one seed per run on 191 questions.

Setup is in [PERCEPTION_JEPA_SPEC.md](PERCEPTION_JEPA_SPEC.md), section "V* and the arms from base".

## V* accuracy

Correct answers out of 191 (115 attribute questions, 76 relative-position questions). Greedy decoding,
16 new tokens. "Native" is the released image at up to 16,384 image tokens (mean 4,472); "1,280" caps
each image at 1,280 image tokens (mean 1,244), the cap the training images had.

| Model | Prompt format | Native: all | Native: attribute | Native: position | 1,280: all | 1,280: attribute | 1,280: position |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Base, untuned | chat template | 143 (74.9%) | 94 | 49 | 128 (67.0%) | 81 | 47 |
| Base, untuned | project format, no latents | 133 (69.6%) | 91 | 42 | 126 (66.0%) | 80 | 46 |
| `sft`: no latents | project format | 131 (68.6%) | 87 | 44 | 121 (63.4%) | 75 | 46 |
| `livr`: 4 latents | project format | 121 (63.4%) | 76 | 45 | 110 (57.6%) | 67 | 43 |
| `patch`: 4 globals + 16 patch latents, globals hidden from the answer | project format | 119 (62.3%) | 75 | 44 | 110 (57.6%) | 68 | 42 |

- The fair baseline for the trained arms is the second row: same prompt format, no training.
- Every trained arm answered every question with a bare letter: no unparsed replies, none in words.
- The base rows in the project format were rescored from the saved replies after the scorer was changed
  to drop a leading full stop or colon (the job itself reported 121 and 122).

## Question-by-question comparison

Questions one model got right and the other got wrong, with a two-sided exact sign test. No correction
for the twelve comparisons.

| Comparison | Native: gained / lost | p | 1,280: gained / lost | p |
| --- | --- | --- | --- | --- |
| `sft` against base | 10 / 12 | 0.83 | 10 / 15 | 0.42 |
| `livr` against base | 16 / 28 | 0.10 | 14 / 30 | 0.02 |
| `patch` against base | 14 / 28 | 0.04 | 13 / 29 | 0.02 |
| `livr` against `sft` | 13 / 23 | 0.13 | 11 / 22 | 0.08 |
| `patch` against `sft` | 10 / 22 | 0.05 | 5 / 16 | 0.03 |
| `patch` against `livr` | 17 / 19 | 0.87 | 13 / 13 | 1.00 |

- **SFT against base:** no detectable change.
- **Latent arms against base and SFT:** a consistent loss of about 10 to 16 questions net, almost all of
  it on attribute questions. Individually the tests are borderline; the direction is the same in all eight.
- **Patch against LIVR:** no difference. The patch latents neither helped nor hurt relative to four
  plain latents.

## Held-out Visual-CoT rows

500 held-out rows, greedy, exact match after normalising case and punctuation.

| Arm | All 500 | Without flickr30k (354) | flickr30k (146) |
| --- | --- | --- | --- |
| `sft` | 227 (45.4%) | 224 (63.3%) | 3 |
| `livr` | 210 (42.0%) | 207 (58.5%) | 3 |
| `patch` | 213 (42.6%) | 212 (59.9%) | 1 |

- flickr30k answers are free-form sentences, so exact match is close to zero for every arm and says
  nothing there. It is 29% of validation and 34% of training rows.
- The base model was not scored on these rows, so these numbers rank the arms but do not show how much
  training helped on the training distribution.
- The ranking matches V*: SFT ahead, the two latent arms together.

## Training

| Arm | Job | Segment 1 (2 epochs) | Segment 2 (3 epochs) | Peak GPU memory | Job time | Mean loss: first epoch → last epoch |
| --- | --- | --- | --- | --- | --- | --- |
| `sft` | 7156097 | 2 h 00 min | 2 h 52 min | 56.3 GiB | 5 h 07 min | 0.87 → 0.019 |
| `livr` | 7156098 | 1 h 59 min | 2 h 56 min | 56.3 GiB | 5 h 11 min | 1.31 → 0.022 |
| `patch` | 7156099 | 1 h 58 min | 2 h 58 min | 56.3 GiB | 5 h 12 min | 1.41 → 0.028 |

- 10,000 rows, 5 epochs, 6,250 optimizer steps per arm, one H200 each, 15.5 GPU-hours in total.
- Segment 1 is Stage 1 (bottleneck mask) for the latent arms and plain training for SFT; segment 2 is
  unmasked for all three.
- Final-epoch loss near 0.02 on rows seen five times suggests the arms largely memorised the 10,000
  rows. No held-out loss was logged per epoch, so this is an inference from the training loss.

## Audit and no-latent runs (2026-11-18, later the same day)

**Result:** the scores above are real and reproduce from the saved weights, and leaving the latent
tokens out of the prompt does not bring the latent arms back. The loss is in the trained weights, not
in the answer leaning on the latent tokens.

### Audit

- **Latents were in the prompt.** Each V* manifest lists the tokens used: none for `sft`, the four LIVR
  tokens for `livr`, four globals plus sixteen patch tokens (globals hidden) for `patch`.
- **Saved weights reproduce the scores.** `livr` and `patch` were reloaded from `segment2/final` and
  re-scored (jobs 7164179, 7164180): the same counts, and all 191 replies identical at both image sizes.
- **`sft` reload:** the first attempt (job 7164178) failed on a loader check that assumed four latent
  tokens for any saved weights. After the fix, job 7164754 gave the same counts (131 and 121) and 191
  identical replies at both sizes.
- **Segment 1 weights were not saved**, so nothing here reads the model after the bottleneck stage alone.

### V* with latent tokens left out

Correct out of 191 (attribute / position in brackets).

| Weights | Prompt | Native | 1,280 tokens | Answers that changed against "as trained" |
| --- | --- | --- | --- | --- |
| `livr` | as trained (4 latents) | 121 (76 / 45) | 110 (67 / 43) | |
| `livr` | no latent tokens | 121 (77 / 44) | 112 (66 / 46) | 13 native, 11 at 1,280 |
| `patch` | as trained (globals hidden) | 119 (75 / 44) | 110 (68 / 42) | |
| `patch` | no latent tokens | 119 (76 / 43) | 108 (66 / 42) | 8 native, 17 at 1,280 |
| `patch` | globals visible to the answer | 120 (76 / 44) | 112 (69 / 43) | 3 native |

- Removing the latents changes about 5 to 9% of answers and leaves accuracy where it was, so the
  earlier guess (the answer over-relies on a few latent vectors) is not supported.
- The latent arms are below `sft` and the base model with or without their latents. Whatever lowered
  V*, training with the latents put it in the language model's weights.

### Held-out Visual-CoT rows, with the base model

354 held-out rows (flickr30k left out). "Contains" counts a reply that contains the reference answer;
it is lenient towards long replies, which is how the base model answers.

| Model | Exact match | Contains the answer |
| --- | --- | --- |
| Base, untuned | 0 | 230 (65.0%) |
| `sft` | 224 (63.3%) | 241 (68.1%) |
| `livr` | 207 (58.5%) | 223 (63.0%) |
| `livr`, no latent tokens | 205 (57.9%) | 225 (63.6%) |
| `patch` | 212 (59.9%) | 231 (65.3%) |
| `patch`, no latent tokens | 216 (61.0%) | 232 (65.5%) |

- The base model scores zero on exact match because it answers in full sentences ("The total ... are
  $796.65." for "796.65"), so exact match measures answer style, not perception.
- On the lenient measure, five epochs of training moved held-out accuracy by at most 11 rows in either
  direction. Together with the training loss near 0.02, the arms learned the answer style and the
  10,000 training rows, and little that transfers.

### Training ablations

**Result:** the bottleneck stage is what lowers V*. LIVR's four latents trained with both segments
unmasked score 134 at native size, level with the base model and SFT, against 121 with the bottleneck
stage. One seed each.

Correct out of 191 (attribute / position in brackets). "Held-out" is the 354 non-flickr30k rows, counted
as "reply contains the answer". Jobs 7164756 to 7164758; none saved weights.

| Run | Read after | Epochs so far | V* native | V* 1,280 | Held-out |
| --- | --- | --- | --- | --- | --- |
| Base, untuned | | 0 | 133 (91 / 42) | 126 (80 / 46) | 230 |
| `sft` (original) | end | 5 | 131 (87 / 44) | 121 (75 / 46) | 241 |
| `sft`, 1 + 1 epochs | segment 1 | 1 | 138 (93 / 45) | 126 (79 / 47) | 246 |
| `sft`, 1 + 1 epochs | end | 2 | 133 (89 / 44) | 121 (74 / 47) | 250 |
| `livr` (original: bottleneck, then unmasked) | end | 2 + 3 | 121 (76 / 45) | 110 (67 / 43) | 223 |
| `livr`, 1 + 1 epochs | segment 1, bottleneck still on | 1 | 68 (30 / 38) | 65 (28 / 37) | 187 |
| `livr`, 1 + 1 epochs | end | 1 + 1 | 132 (87 / 45) | 118 (71 / 47) | 237 |
| `livr`, no bottleneck stage | segment 1 | 2 | 140 (92 / 48) | 129 (79 / 50) | 248 |
| `livr`, no bottleneck stage | end | 5 | 134 (88 / 46) | 130 (83 / 47) | 246 |

Paired comparisons (gained / lost, two-sided exact sign test, uncorrected):

| Comparison | Native | p | 1,280 | p |
| --- | --- | --- | --- | --- |
| no bottleneck against original `livr` | 29 / 16 | 0.07 | 28 / 8 | 0.001 |
| no bottleneck against `sft` | 7 / 4 | 0.55 | 14 / 5 | 0.06 |
| no bottleneck against base | 12 / 11 | 1.00 | 18 / 14 | 0.60 |
| no bottleneck after 2 epochs against base | 16 / 9 | 0.23 | 17 / 14 | 0.72 |
| `livr` 1 + 1 against original `livr` | 23 / 12 | 0.09 | 19 / 11 | 0.20 |

- **The bottleneck stage is the cause.** Same latents, same rows, same number of steps: without the
  Stage 1 mask the model is 13 questions higher at native size and 20 higher at 1,280 tokens, and no
  longer below the base model or SFT.
- **Under the bottleneck the model cannot answer V\*.** After one masked epoch it scores 68 of 191,
  which is chance (attribute questions have four options: 30 of 115; position questions have two: 38 of
  76). Four latents carry no usable V* detail at that point. This run was not scored after two masked epochs.
- **One unmasked epoch undoes most of it.** The 1 + 1 run ends at 132, level with the base model at
  native size. The original run had twice the masked training and ended 11 lower, so the damage grows
  with time under the mask or with the later epochs; these runs do not separate the two.
- **Nothing beats the base model.** The best reading, 140 after two unmasked epochs with latents, is 7
  above the base model and within noise (p = 0.23). SFT after one epoch (138) is the same.
- **Longer training does not help.** In all three families the score after fewer epochs is at or above
  the score after more (138 → 133 → 131 for SFT; 140 → 134 without the bottleneck). The differences are
  within noise; none points the other way.

Attention maps for the four saved models are in [PERCEPTION_VSTAR_ATTENTION.md](PERCEPTION_VSTAR_ATTENTION.md).

## What this does and does not show

- **Shown:** at this scale (10,000 answer-only rows, 3B model, one seed), fine-tuning on Visual-CoT does
  not improve V*, and adding latents costs accuracy on attribute questions compared with SFT on the same
  rows and steps.
- **Not shown:** that latents hurt in general. LIVR's paper trains on its full task set with a 7B model
  and reports V* rising from 78.5 to 80.1; this run used a 2.5% sample of Visual-CoT, capped training
  images at 1,280 tokens, and repeated the sample five times.
- **Size of the benchmark:** 191 questions gives a 95% interval of roughly ±6 to 7 points on any single
  score, so only the paired comparisons above carry weight.
- **Format cost:** the project prompt format alone costs the base model 10 questions at native size
  (143 → 133), mostly on relative position.

## Not done

- A second seed for any arm.
- The patch arm without its bottleneck stage, and any run with a longer or shorter masked stage alone.
- A larger sample or native-size training images.

## Where the results are

On PACE-ICE under `$HOME/scratch/SPARA/results/`:

- `vstar-base-7155372/{chat,livr}-{16384,1280}/`: base-model replies and summaries.
- `perception-sft-7156097/`, `perception-livr-7156098/`, `perception-patch-7156099/`: `summary.json`,
  `validation.jsonl`, `vstar-{16384,1280}/responses.jsonl`, per-step losses, and the final weights at
  `segment2/final` (15 GB per arm).
