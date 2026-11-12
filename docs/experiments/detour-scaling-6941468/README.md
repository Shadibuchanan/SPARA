# Evidence for detour scaling check 6941468–6941471

The [detour scaling check](../../DETOUR_SCALING_SPEC.md): does fine-tuning on more
distinct detour layouts generalize to unseen detour maps?

- **Starting point:** every arm starts from the LIVR rerun's final Stage 2 weights
  (`baseline-6253300`).
- **Jobs:** one H200 per arm, 2026-11-10. All four completed on the first submission.

| Arm | Job | Extra detour layouts | Wall time | Training | Last-quarter loss |
| --- | --- | --- | --- | --- | --- |
| L | 6941468 | none | 3 min 59 s | — | — |
| V98 | 6941469 | 98, about 20 repeats each | 23 min 15 s | 375 steps, 808 s | 0.023 |
| V500 | 6941470 | 500, 4 each | 23 min 9 s | 375 steps, 906 s | 0.074 |
| V2000 | 6941471 | 2,000, once each | 19 min 47 s | 375 steps, 866 s | 0.113 |

- **GPU-seconds:** 4,210 (about 70 GPU-minutes) by Slurm's elapsed time, against an
  estimate of about 80. Peak memory was 55 GiB for the V arms and 20 GiB for L.
- **Source:** the hashes in each manifest match the local code for every file this
  experiment uses. The only mismatches are files it does not import:
  - `src/spara/scaffolding/training.py`, `configs/scaffolding.json`,
    `scripts/pace/pace.conf` and `scripts/pace/scaffolding-variants.sbatch`, which
    differ on PACE and are left as they are;
  - the package's `SOURCES.txt`.
- **Weights:** the three fine-tuned checkpoints (15 GB each) were deleted from PACE
  scratch at the user's request on 2026-11-10. The run folders keep their logs, routes,
  benchmark responses and summaries. Rerunning an arm recreates its weights.

## Validity: all four checks passed

| Check | Threshold | Result |
| --- | --- | --- |
| Pool and held-out labels accepted by the reference evaluator | 100% | 2,200 of 2,200 |
| Held-out set and pool share no layout with each other, `fresh/probe`, train or test | 0 shared | 0 in all 7 pairs |
| L's greedy routes on the 40 test detour maps match planning probe 6887458 | ≥ 38 | 40 of 40 |
| V98 solves its seen maps (the first 98 pool maps) | ≥ 90% | 98 of 98 |

## Primary: variety helps

V2000 against V98 on the 240 held-out maps, greedy decoding, exact McNemar.

| | Maps | V2000 | V98 | Only V2000 | Only V98 | p |
| --- | --- | --- | --- | --- | --- | --- |
| **All held-out** | 240 | **107** | **64** | 51 | 8 | 9 × 10⁻⁹ |
| Generated (`detour-heldout`) | 200 | 87 | 53 | 39 | 5 | 1.4 × 10⁻⁷ |
| Test detour maps | 40 | 20 | 11 | 12 | 3 | 0.035 |
| 3 x 3 | 11 | 9 | 8 | 1 | 0 | 1.0 |
| 4 x 4 | 36 | 24 | 20 | 7 | 3 | 0.34 |
| 5 x 5 | 73 | 36 | 18 | 20 | 2 | 1.2 × 10⁻⁴ |
| 6 x 6 | 120 | 38 | 18 | 23 | 3 | 8.8 × 10⁻⁵ |

- **The gain holds on the released test detour maps on their own.** These 40 maps
  were not made by the generator, so the gain is not just a match to the generator's
  distribution.
- **Most of the gain is on 5 x 5 and 6 x 6 maps.** Small maps were already mostly
  solved.

## Secondary: detour fine-tuning helps, but only on the generated maps

V98 against L on the same 240 maps.

| | Maps | V98 | L | Only V98 | Only L | p |
| --- | --- | --- | --- | --- | --- | --- |
| **All held-out** | 240 | **64** | **42** | 34 | 12 | 0.0016 |
| Generated | 200 | 53 | 34 | 29 | 10 | 0.0034 |
| Test detour maps | 40 | 11 | 8 | 5 | 2 | 0.45 |

The reading is *detour fine-tuning helps*. Its significance comes from the generated
maps; on the 40 test detour maps alone the difference is not detectable.

## Descriptive (no verdicts)

**Trend across arms.** Held-out success rises with variety. The step from 500 to 2,000
layouts is not distinguishable in this run (V2000 21 maps only, V500 11 only, p = 0.11).

| | L | V98 | V500 | V2000 |
| --- | --- | --- | --- | --- |
| Held-out, 240 | 42 | 64 | 97 | 107 |
| Generated held-out, 200 | 34 | 53 | 78 | 87 |
| Test detour, 40 | 8 | 11 | 19 | 20 |
| Seen (first 98 pool maps) | 22 | 98 | 62 | 50 |

**Memorisation is visible on the seen maps.**
- V98 saw each seen map about 20 times. It solves all of them, but only 27% of the
  held-out maps.
- V2000 saw each one once. It solves 51% of them, close to its 45% on held-out maps.

**Benchmark (reference protocol, one sampled answer per test map).**
- The V arms were evaluated in this run. L is the recorded LIVR rerun evaluation.
- Decoding is the same in both: temperature 0.7, top-p 0.9, 1,024 new tokens,
  repetition penalty 1.05.

| | L | V98 | V500 | V2000 |
| --- | --- | --- | --- | --- |
| All 400 | 328 | 316 | 328 | 319 |
| 40 detour maps | 9 | 10 | 18 | 16 |
| 360 no-detour maps | 319 | 306 | 310 | 303 |

- **Detour maps:** V500 and V2000 roughly double LIVR on the benchmark's detour maps
  (V500: 10 maps only, 1 only LIVR, p = 0.012; V2000: 8 and 1, p = 0.039).
- **No-detour maps lose more.** All three V arms lose 9–16 no-detour maps
  (V98 p = 0.047, V500 p = 0.12, V2000 p = 0.014). No arm improves the overall
  benchmark.

**How big run-to-run noise is on the no-detour maps** (computed after the run):
- **Two LIVR runs:** 316 (6189457) and 319 (6253300), with 12 and 15 maps solved
  by only one of them (p = 0.70).
- **Repeated sampling:** 16 samples per test map from the LIVR rerun
  ([multisample 6267411](../multisample-6267411/)) predict one sampled pass at
  318 ± 2.2.
- **Reading:** V98 (306) and V2000 (303) are more than 5 standard deviations below. So
  the no-detour loss is real for these runs, not sampling noise. V500 (310) is 3.6
  standard deviations below, and its paired test is not significant.

The likely cause is in the spec's limits: the fine-tuning mix is 70% detour examples.
Variation between training runs is not measured.

**How the arms fail** (computed after the run with the reference rules):

| | L | V98 | V500 | V2000 |
| --- | --- | --- | --- | --- |
| Held-out greedy: goal by a shortest route | 40 | 62 | 95 | 106 |
| Held-out greedy: goal by a longer route | 2 | 2 | 2 | 1 |
| Held-out greedy: into a hole | 192 | 170 | 139 | 130 |
| Held-out greedy: stops short of the goal | 6 | 6 | 4 | 3 |
| Benchmark no-detour: shortest / longer | 317 / 2 | 303 / 3 | 305 / 5 | 296 / 7 |
| Benchmark no-detour: hole / stops short | 38 / 3 | 46 / 8 | 45 / 5 | 45 / 12 |

- **Holes dominate the failures in every arm.** Even V2000 walks into a hole on 54%
  of the held-out maps.
- **V2000 overshoots on no-detour maps.** It loses 21 shortest-route successes
  against L: 7 more into holes, 9 more stopped short and 5 more by longer routes. The
  last two are unneeded detours.

## Files

| File | Contents |
| --- | --- |
| comparison.json | Output of `scripts/analyze_detour_scaling.py`: validity, primary, secondary, every descriptive comparison, in-job seconds |
| harm-and-failures.json | The noise floor (two LIVR runs; single-pass s.d. from 16 samples) and the outcome breakdown above |
| `<arm>/summary.json` | Set checks, training summary, greedy counts by population, run time, peak memory |
| `<arm>/greedy.jsonl` | One row per greedy evaluation: population, split, row, grid size, route, success |
| `<arm>/manifest.json` | Run manifest: config, recipe, package versions, source hashes, data and fresh-set manifests. Paths under the PACE results root are written as `$SPARA_REMOTE_ROOT/` |
| `<arm>/steps.jsonl` | V arms: per optimizer step loss, gradient norm, learning rate |
| `<arm>/benchmark-outcomes.jsonl` | V arms: per test map row, grid size, success, extracted answer (prompts and raw text dropped) |
| `<arm>/benchmark-summary.json` | V arms: benchmark accuracy overall and by size |
