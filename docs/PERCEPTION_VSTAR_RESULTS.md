# V* results: three arms trained from base on Visual-CoT (2026-11-18)

**Result:** no trained arm beats the untuned Qwen2.5-VL-3B on V*. Direct SFT is level with it; the two
latent arms (LIVR, and global + patch latents) are 12 to 16 questions lower, and do not differ from each
other. This is one seed per arm on 191 questions.

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
- A base-model score on the held-out Visual-CoT rows.
- Earlier checkpoints (after segment 1, or after fewer epochs), which would test the memorisation reading.
- A larger sample or native-size training images.

## Where the results are

On PACE-ICE under `$HOME/scratch/SPARA/results/`:

- `vstar-base-7155372/{chat,livr}-{16384,1280}/`: base-model replies and summaries.
- `perception-sft-7156097/`, `perception-livr-7156098/`, `perception-patch-7156099/`: `summary.json`,
  `validation.jsonl`, `vstar-{16384,1280}/responses.jsonl`, per-step losses, and the final weights at
  `segment2/final` (15 GB per arm).
