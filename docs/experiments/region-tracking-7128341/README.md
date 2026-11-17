# Evidence for position tracking on EGD (job 7128341)

The probe in [the latent JEPA spec](../../PERCEPTION_JEPA_SPEC.md#position-tracking-on-egd-the-state-probe):
- **What it is:** the [planning probe](../../PLANNING_PROBE_SPEC.md#2-state-tracking)'s state probe, on
  EGD's final weights (job 7124258). There is no training.
- **The question:** when EGD steps into a hole, does the state that writes the move hold the
  agent's cell?
- **Two answer modes:** unmasked, and masked with the global latents hidden, so that the block
  latents are the answer's only map.
- **Maps:** round 2's 400 held-out maps (greedy routes) and 500 pool maps (released answers
  teacher-forced, to fit the probes). All are 7 x 7 or 8 x 8.

## The run

- **Compute:** one H200, 1,019 GPU-seconds by Slurm's elapsed time (16 min 59 s; 739 s in the
  run). The time limit was 30 minutes.
  - About 4.5 minutes were outside the run, mostly the model load.
  - Each mode took about 6 minutes: 400 greedy answers and 900 state passes.
- **Source:** the hashes in manifest.json match the local code for every file the run imports,
  except four files changed after the run.
  - **Those files:** `src/spara/model.py`, `serialization.py`, `region_latents.py` and `data.py`.
  - **The change:** the global + patch layout. It leaves FrozenLake prompts and the block mask as
    they were: 60 real prompts encode identically, and the block-mask tests pass unchanged.
  - **Other mismatches** are scaffolding files the run does not import.
- **The features** (hidden states, 212 MB per mode) are on PACE at
  `$SPARA_REMOTE_ROOT/results/region-tracking-7128341/` and in the git-ignored local `results/`.

## Pre-declared readings

| | Unmasked | Masked, globals hidden |
| --- | --- | --- |
| Solved | 78 of 400 | 63 of 400 |
| Routes equal to EGD's recorded routes | 400 of 400 | 400 of 400 |
| Position probe's layer | 20 | 32 |
| Position accuracy, held-out pool maps | 19% | 16% |
| Position accuracy, solved test steps (floor 50%) | 23% | 16% |
| **Reading** | **Inconclusive: the probe is too weak** | **Inconclusive: the probe is too weak** |

- **Validity: passed.** Both modes reproduce EGD's routes on every map.
- **The probe is far below the floor.** It is also below LIVR's 39–49% on 3 x 3 to 6 x 6 maps, which
  the planning probe already called too weak for a verdict.
- **Chance level:** a uniform guess over cells is right 1 time in 49 at 7 x 7 and 1 in 64 at 8 x 8.

**S1 as computed (descriptive only, because the probe is too weak):**

| | Fatal steps, k ≥ 2 | Decoded correctly | Expected from solved steps | Gap | p |
| --- | --- | --- | --- | --- | --- |
| Unmasked | 240 | 41 (17%) | 19% | 1.5 points | 0.31 |
| Masked, globals hidden | 244 | 41 (17%) | 14% | −2.4 points | 0.88 |

**S2:**
- **The question:** for fatal steps whose decoded cell is wrong, would the fatal move be safe from
  the decoded cell?
- **Unmasked:** 82%, against 82% from other cells.
- **Masked, globals hidden:** 85%, against 82%.
- **No sign** that the model "thought it was somewhere safe".

**S3:**
- **The bias check fired again.** The hole-ahead probe scores the direction taken higher than other
  safe directions: 0.46 against 0.21 unmasked, 0.61 against 0.19 masked. Its fatal-against-avoided
  comparison is therefore confounded.

## Exploratory, post hoc (CPU)

[scripts/analyze_planning_probe.py](../../../scripts/analyze_planning_probe.py) on each mode's
features. Output: `<mode>/exploratory.json`.

| Check | Unmasked | Masked, globals hidden | LIVR (job 6887458) |
| --- | --- | --- | --- |
| The move about to be written, read from its state, layer 36 | 77% | 75% | 98% |
| Start cell at the first move (linear), best layer | 42% | 33% | 45–68% |
| Absolute cell, small MLP, best of layers 16, 20 and 28 | 21% | 19% | 33–55% |
| Offset from the start on the model's own routes, layer 20: solved steps | 97% | 98% | 99% |
| Offset from the start, layer 20: fatal steps | 223 of 240 | 231 of 244 | 49 of 51 |
| Position at earlier steps of failed routes, against expected | 18% against 20% | 17% against 14% | 28% against 25% |
| Position at fatal steps, against solved steps next to a hole | 17% against 18% | 17% against 15% | 16% against 28% |
| Hole read ahead in the direction taken: fatal holes against safe cells | 36% against 39% | 55% against 63% | 41% against 21% |

**What these suggest (interpretation, with the weak-probe caveat):**
- **The model keeps count of its moves.** The offset from the start is present at almost every step,
  fatal ones included, as for LIVR. That count can be read off the moves already written.
- **The absolute cell is barely resident** where moves are written on 7 x 7 and 8 x 8 maps, at any
  step. There is no drop specific to fatal steps, unlike LIVR's.
- **The move-alignment check is low (77% against LIVR's 98%).** The probes are fitted on the pool
  maps' released routes, which EGD often would not write itself. Those states are partly off its
  own policy.
- **The state that writes a fatal move does not mark the hole ahead** any more than it marks a safe
  cell.

## Beside step grounding (job 6926184)

[Step grounding](../../STEP_GROUNDING_SPEC.md#result) already tested the tracking hypothesis on
LIVR at 3 x 3 to 6 x 6.
- **What it did:** trained the model's states to hold the agent's cell.
- **The cell before a fatal move** was then read correctly 76–78% of the time, against 16%.
- **Walks into holes did not fall:** 67 against 68 on greedy test routes.
- **That result weighs against tracking as the limit.** This probe's design should have cited it.

Together:
- **EGD:** the model holds the map in its latents and uses them, and still steps into holes they
  flag.
- **Step grounding:** a model that holds its cell, and often the hole ahead, still steps into holes.
- **This probe** adds no verdict on tracking at 7 x 7 and 8 x 8.
- **The remaining explanation:** how a move is chosen from a state that has the information. Expert
  iteration, the only lever that has moved routes, trains exactly that choice.

## Files

| File | Contents |
| --- | --- |
| summary.json | Per mode: solved maps, the reproduction check and the reading |
| manifest.json | Run manifest; the PACE root is written as `$SPARA_REMOTE_ROOT/` |
| greedy/, masked_hidden/ | Per mode: summary.json (the state probe: layer selection, S1 to S3), exploratory.json (the post hoc checks), routes.jsonl (greedy test routes, then the teacher-forced pool maps), steps.jsonl (each test step: cell, decoded cell, move, fatal, solved, hole-ahead scores) |
