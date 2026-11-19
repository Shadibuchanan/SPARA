# Perception benchmarks: LIVR, patch latents and Scaffolding Minds on MMVP, V* and BLINK

Status, 2026-11-19: pilots done; base (7161795) and livr (7165735) scored. The first patch and
scaffold jobs (7165736, 7165737) failed on the scratch quota while saving weights and were rerun
(7178426, 7178427); scaffold now saves its final checkpoints without optimizer state. The readings
below were fixed before any GPU run.

## Question

Do region latents help perception questions, where no planning step sits between seeing and answering?

On FrozenLake, EGD's block latents held the map and the answer used them, but routes did not
improve. The limit was how moves are chosen (see
[the latent JEPA spec](PERCEPTION_JEPA_SPEC.md#result-egd-jobs-7124258-and-7125522)). Perception
benchmarks remove that step.

Requested by the user on 2026-11-18 ("lets run it on the perception benchmarks and then see what if
anything we should try to fix").
- **Run size:** the user chose 8k training examples.
- **Training data:** Zebra-CoT, after the user asked for LIVR and Scaffolding Minds baselines on the
  same data. SAT, the first choice, has no helper images, which Scaffolding Minds needs.

## Data

[scripts/prepare_perception.py](../scripts/prepare_perception.py), on a CPU node.

**Training: 8,000 Zebra-CoT rows** (Scaffolding Minds' own training set). An equal third comes from
each of three categories that match the benchmarks:
- visual search (as V*);
- visual jigsaw (BLINK's Jigsaw);
- multi-hop object counting (BLINK's Counting).

Rows are drawn with seed 42 from the first shards. How they are prepared:
- The prompt is the question, with each problem image in place.
- The answer is the final answer.
- Answers longer than four words are dropped, to match the benchmarks' short answers.
- A letter answer gets "Answer with the option's letter from the given choices directly."; any
  other answer gets "Answer the question using a single word or phrase."
- The reasoning traces are not used. Scaffolding Minds alone also reads the first reasoning image,
  as its helper.

**Evaluation: 2,392 questions, every one multiple choice, scored by the option letter.**

| Benchmark | Questions | Notes |
| --- | --- | --- |
| MMVP | 300 | Options rewritten as (A)/(B); also scored by pair (both questions of a pair right) |
| V* | 191 | 115 attribute and 76 relative-position questions |
| BLINK (val) | 1,901 | 14 subtasks, 1 to 4 images, with the released prompts |

- **V\* keeps its released option order.** Its correct letters are already spread: 71 A, 70 B, 27 C
  and 23 D.
- **Image size:** every image is resized, keeping its aspect ratio, to at most 802,816 pixels (1,024
  merged image tokens). This caps V*'s fine detail; every arm sees the same images.

**Halved to 4,000 rows on 2026-11-18, before any full run.** The pilots put the 8k runs at about 10.5
H200-hours, twice the estimate, and the user chose to halve them. Every arm trains on the first 4,000
rows of the shuffled set. Step counts below are for 8,000 rows; at 4,000 they halve.

## Arms

Every arm starts from base Qwen2.5-VL-3B (the pinned revision) and trains on the same 8,000 rows.
Training follows LIVR's recipe ([configs/livr.json](../configs/livr.json)) except for the epochs.
- **base:** no training and no latents. The reference.
- **livr:** LIVR's four latents. One epoch of Stage 1 (bottleneck), then one of Stage 2: 1,000
  optimiser steps each.
- **patch:** the four global latents plus
  [global + patch latents](PERCEPTION_JEPA_SPEC.md#global--patch-latents-for-images-without-a-grid-built-not-run).
  - 4 x 4 tiles per image, each patch latent masked to its tile, seeing the prompt text.
  - Steps as livr.
  - **The globals are hidden from the answer** on every Stage 1 row and every other Stage 2 row,
    as EGD, so the answer must read the patch latents.
- **scaffold:** Scaffolding Minds Stage 1, from
  [configs/scaffolding_perception.json](../configs/scaffolding_perception.json).
  - One epoch of the scaffold phase (the encoder learns from the helper image), then one of the
    latent-prediction phase, at batch 32 (250 steps each).
  - No RL. The paper's RL stage is out of budget.

**Evaluation:** greedy decoding, at most 16 new tokens, on every question.
- **The answer's letter:** a leading letter, else the first "(X)", else the first standalone option
  letter.
- **patch** is scored with the globals shown, and again, descriptively, with them hidden.
- **Amended 2026-11-18, after the pilots and before any full result:** an answer with no letter is
  matched to the one option whose text it equals or contains (case-insensitive). The patch pilot
  often answered with the option's words ("closed"). Saved answers are rescored on CPU
  (`perception.rescore`), so any run can be rescored the same way.

## Readings (fixed before the runs)

`perception.compare`, exact McNemar tests on paired questions.
- **Validity:**
  - every arm is scored on the same 2,392 questions;
  - at most 5% of each arm's answers on each benchmark have no letter. Above that, that benchmark's
    reading for that arm is "invalid: answers not parsed".
- **Primary:** patch against livr on all 2,392 questions, p < 0.05.
- **Secondary,** each on all questions and on each benchmark:
  - livr against base;
  - patch against base;
  - scaffold against livr;
  - scaffold against base.
- **Descriptive:**
  - accuracy by benchmark and BLINK subtask;
  - MMVP pair accuracy;
  - patch with the globals hidden.

| Primary: patch against livr | Reading |
| --- | --- |
| Better, p < 0.05 | Patch latents help perception. Next, locate where (subtasks). |
| Worse, p < 0.05 | Patch latents hurt perception at this scale |
| Neither | No detectable difference between patch and LIVR latents at this scale |

- **If livr does not beat base,** training on 8k Zebra-CoT rows may itself cost general perception.
  The arms are then compared with each other, not read as gains.
- **"What to fix" is decided afterwards,** from the subtask results. It will be labelled post hoc.

## Budget

No per-run allowance (the user's standing preference); each job gets a time limit with headroom.
1. **Pilot first:** 16 rows of every set for patch and scaffold, the two arms with new code paths, to
   measure seconds per row.
2. **Estimate,** to be replaced by the pilot's: about 4 to 5 H200-hours for the three trained arms
   and four evaluations.
3. **Weights:** each trained arm saves its final weights, about 15 GB each; scaffold also saves its
   encoder, about 2.6 GB. Scratch had about 80 GB free after the 2026-11-18 deletions.

## Running

```bash
python scripts/prepare_perception.py data/perception        # pyarrow, pillow, huggingface_hub
python -m spara.perception --arm patch --data-root data/perception --base-model M --output RUN [--limit 16]
python -m spara.perception --compare base=B livr=L patch=P scaffold=S --output comparison.json
```

On PACE: [scripts/pace/perception.sbatch](../scripts/pace/perception.sbatch) with `SPARA_ARM` and,
for a pilot, `SPARA_LIMIT`.
