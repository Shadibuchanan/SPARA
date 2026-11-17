# Evidence for the latent JEPA gate job 7106850

The gate job in the [latent JEPA spec](../../PERCEPTION_JEPA_SPEC.md#gate-job-one-gpu-job-before-the-arm),
from E (round 1's fine-tune, `expert-train-search-6961875`), unmasked.

- **Reading (rule): gates passed, run EJ2.**
- **Compute:** one H200, 1,148 GPU-seconds by Slurm's elapsed time (19 min 8 s; 893 s inside the
  job; peak 32.9 GiB), against an estimate of 20–25 GPU-minutes.
- **A first submission, job 7105700,** failed after 2 seconds on a shell quoting error in the job
  script, before any Python. `tests/test_pace_scripts.py` now parses every job script.
- **Source:** the hashes in manifest.json match the local code for every file this run imports.
  The mismatches are files it does not import: the protected PACE files
  (`src/spara/scaffolding/training.py`, `configs/scaffolding.json`, `scripts/pace/pace.conf`,
  `scripts/pace/scaffolding-variants.sbatch`) and the package's `SOURCES.txt`.

## The gates

**2a, the image tokens show the holes at 7 x 7 and 8 x 8: passed.**
- **Probe:** fitted on the 500 pool maps' cells and scored on the 400 held-out maps' cells.
- **Result:** balanced accuracy 1.00 at both sizes and at every candidate layer (4, 8, 12, 16).
- **Target layer (rule: the earliest passing):** 4.

**2b, the loss weight: within bounds.**
- **Rule:** the median answer-gradient norm into the language model over the median JEPA-term norm,
  on the first 16 rows of phase 1 under the Stage 1 mask.
- **Result:** β = 2.47, inside 0.01 to 100.
- **Answer gradients vary widely:** 0.001 to 37.5. Rows whose answers E already fits give almost
  none.
- **JEPA term:** gradient into the language model 0.31 to 9.7; into the heads 0.65 to 1.08.

**2c, E's route follows its latents: passed, near the line.**
- **Maps:** 200 held-out maps (the first 100 of each size), greedy, with every layer's latent states
  swapped from another map of the same size.

| | Exact donor route (rule: at least 50%) | First two moves the donor's | Keeps its own route | Own answers solved |
| --- | --- | --- | --- | --- |
| E | 102 of 200 (51%: 50 at 7 x 7, 52 at 8 x 8) | 141 | 1 | 30 |
| E2 (descriptive) | 108 of 200 (54%) | 151 | 0 | 40 |

- **The exact-route share passed by two maps.**
- **The other counts point the same way.** Swapped answers almost never keep their own route, and
  most begin as the donor's. On these long, mostly failed routes, the latents set the route.
- **Identity:** re-inserting each of the first four maps' own states reproduced its answer, for both
  models.

## Other outputs

**Features:** 2,185 maps in 253 s:
- the 1,486 distinct training maps, built as round 2 builds its rows;
- the pool and the held-out maps;
- 128 validation maps: pool maps the search did not solve, beyond the 98 seen maps.

**Warm start of the heads, on E's frozen features.** Validation error is the share of target variance
unexplained.
- **Untrained heads:** 1.25.
- **After 30 epochs:** 0.65 on validation maps, 0.44 on training maps.
- **The best epoch was the last,** so the warm start had not converged. The arm's training continues
  from there.
- **Reading, descriptive:** from E's frozen latents, the heads predict about a third of the target
  variance on maps never trained on.

## Files

| File | Contents |
| --- | --- |
| summary.json | The gates and their readings; the probe by layer and size; the warm start's history; per-row gradient norms and β; both swap checks; timing |
| manifest.json | Run manifest: config, settings, package versions, source hashes, data and fresh-set manifests. The PACE root is written as `$SPARA_REMOTE_ROOT/` |
| swap-e.jsonl, swap-e2.jsonl | One row per swapped map: row, donor, size, whether the answer is the donor's route, whether its first two moves are, whether it kept its own route, success |

`jepa_heads.pt` stays on PACE scratch at `$SPARA_REMOTE_ROOT/results/jepa-gate-7106850/`. It holds the
heads, the centring means, the settings with the target layer and β, and the SIGReg queues; no model
weights.
