# Latent-exploration diagnostics after C′ (jobs 7148362 and others)

The diagnostics in [LATENT_DIAGNOSTICS_SPEC.md](../../LATENT_DIAGNOSTICS_SPEC.md), and the confirmation
in its amendment of 2026-11-19. Nothing trains. The full readings are in `summary.json` (all parts), and
the full table of perturbations is in `q4-table.md`. The confirmation is in `confirm.json`. Per-map
outputs, states and bases stay on PACE scratch (`sft-run/results/latent-diag-*`, `latent-confirm`).

## What ran

| Job | Part | GPU time (one 48 GB Blackwell slice) |
| --- | --- | --- |
| 7148362 | Sweep at layers 16 and 24, with a text baseline; stopped at its limit after 96 of 200 maps | 3 h 00 |
| 7148373 | Sweep at layer 32, plus answer-token noise at 16 and 24; stopped after 156 maps | 2 h 30 |
| 7148708 | Text baseline only (64 rollouts per map), all 200 maps | 0 h 39 |
| 7148374 | Gradient split (Q1, Q2), then final-verdict and latent search (Q6) | 1 h 47 |
| 7148638 | Five checkpoints (Q7); a first attempt, 7148375, failed on a missing field | 0 h 41 |
| 7151591 | Confirmation of `pca/24/0.25` on maps 97–200 | 0 h 14 |
| 7148376 | Weights (Q8), on the CPU | — |

About 8.9 slice-hours in all, against the spec's estimate of 3.5. The sweeps were slower per map than
estimated, and the user agreed to stop them at their limits rather than extend them.

## Result

- **C′ as built was confounded (Q1, Q2, Q8).**
  - Its latent term made the gradient about 200 times larger (norm 515–1,237 against B′'s 4.7–6.4).
  - Under one clip, 71–96% of the update's entries in layers 17–36 fell below 1e-7, where Adam's ε
    damps them.
  - C′'s weights in those layers moved 20–35% less than B′'s.
  - The pre-registered B′ > C′ result stands, but it does not show that latent exploration cannot work.
- **Most latent noise adds nothing over text sampling (Q4).** At layers 16 and 32, the label-free
  families change little at any scale. At layer 24, noise from c = 0.5 up breaks about as many maps
  as it rescues.
- **Noise in label-chosen directions does add exploration** (`decision` at layer 32: mixed-share gain
  up to +0.094, 7–8 rescues, 0–1 breaks). But RL has no labels, so this is an oracle.
- **One usable setting passed: C′'s basis at layer 24, c = 0.25.** On fresh maps it was confirmed by
  the rule fixed beforehand:

| Maps | Mixed-share gain (95% CI) | Rescues | Breaks | Success: noise against text | Passes |
| --- | --- | --- | --- | --- | --- |
| 1–96 (selection) | +0.085 (+0.008, +0.165) | 6 | 1 | 0.158 against 0.164 | yes |
| 97–200 (confirmation; two groups of 8 per map) | **+0.121 (+0.065, +0.180)** | 3 | 3 | 0.150 against 0.167 | **yes** |
| 97–200 at c = 0.1 (secondary) | +0.044 (+0.003, +0.084) | 3 | 0 | 0.167 against 0.167 | yes |

  - The confirmation's checks passed:
    - all 104 maps were in the expected order;
    - LIVR's greedy answers matched the text run's on every map;
    - the noise's measured size was 0.98–1.00 of its expected size per slot;
    - the basis's sha256 matched sweep 7148362's.
  - **Where the gain comes from** (maps 97–200, c = 0.25):

    | Maps, by text success | Maps | Share of the +0.121 | Noise success against text |
    | --- | --- | --- | --- |
    | text never solves (0 of 64) | 60 | +0.048 | 0.018 against 0 |
    | text solves under half | 29 | +0.003 | 0.125 against 0.120 |
    | text solves at least half | 15 | +0.070 | 0.729 against 0.925 |

  - More than half of the confirmed gain comes from **lowering success on maps the policy already
    solves**: the pattern of D1, smaller. The rest is real signal on maps text never solves (about 1
    success in 55 rollouts). The rule's guards held, but narrowly: 3 rescues against 3 breaks, and a
    success drop of 0.017 against a limit of 0.02.
  - At c = 0.1 the gain is smaller, about a third, with no breaks and no success drop.
- **Latent search is not worth adding to expert iteration (Q6).** On LIVR's 170 greedy failures,
  final-verdict search solved 110. Latent search at layer 24 solved 78 (text pass@64: 49), and
  adding it to final-verdict search gave 117.
- **RL did not change how much the model relies on its latents (Q7).** The answer follows the donor's
  latents on 84–90 of 120 swaps for LIVR, B′ and C′.
- **The heads' learnability (Q3) is unanswered.** The offline fits diverged, with too few
  mixed-reward groups to train on.

## What it means

- Latent noise at layer 24, at a small scale, adds exploration that text sampling does not.
  - That is confirmed on fresh maps.
  - About half of it is "noise breaking easy maps", not "noise finding new routes".
- By the amendment, a C″ is now specified (layer 24, C′'s basis, small σ, offset replay, latent term
  confined to the heads), and run only with the user's agreement.
- The expected effect is modest. In a training step of 24 maps:
  - text sampling gives about 4.7 mixed groups;
  - this noise would add about 3, about 1 of them on a map text never solves.

## Limits

- **One base model (LIVR) and one pool.** The confirmation used the pilot maps, the first maps RL
  trains on.
- **The sweeps stopped early** (96 and 156 of 200 maps).
- **About 80 configurations were tested before the confirmation,** which is why it was needed.
- **The rule's rescue and break counts are small** (3 to 9), so they carry little weight on their own.
- **The `decision` family uses the label route.** This was not flagged in the spec before the run.
