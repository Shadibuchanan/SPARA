# Evidence for global dropout from EGR, EGD (job 7124258)

This is the arm in [the latent JEPA spec](../../PERCEPTION_JEPA_SPEC.md#global-dropout-from-egr-egd).
- **The arm:** from EGR's weights, round 2's two phases with the global latents hidden from the
  answer: on every row in phase 1, on every other row in phase 2.
- **Then:** evaluation on the 400 held-out maps.

## The run

- **Compute:** one H200, 2,063 GPU-seconds by Slurm's elapsed time (34 min 23 s). The allowance was
  2,100.
- **Slurm reports TIMEOUT.** The job hit its 34-minute limit at the start of the swap test, its last
  step.
  - **Why it ran long:** the model load took about 3.8 minutes, against about 1.3 for EGR. The rest
    ran as expected (phase 1 7.8 minutes, phase 2 about 7.5, each evaluation 3.5 to 3.8, the states 2).
- **Complete:** training, the three evaluations (400 maps each), the latent states and the final
  weights (15 GB, on PACE at `$SPARA_REMOTE_ROOT/results/region-egd-7124258/phase2/final`).
- **Not run in this job:** the swap test. It ran afterwards as job 7125522 (below).
- **Source:** the hashes in manifest.json match the local code for every file the run imports. The
  mismatches are files it does not import.

## Readings available without the swap test

Exact McNemar on the 400 held-out maps (partial-readings.json):

| | EGD | Against | Only EGD | Only the other | p |
| --- | --- | --- | --- | --- | --- |
| Masked, globals shown | 78 | EGR 68 | 21 | 11 | 0.11 |
| Masked, globals shown | 78 | EB2 73 | 19 | 14 | 0.49 |
| Unmasked | 78 | EGR 75 | 21 | 18 | 0.75 |
| Unmasked | 78 | EB2 78 | 17 | 17 | 1.0 |
| Unmasked | 78 | E2 84 | 18 | 24 | 0.44 |

- **Neither improvement on EGR reaches p < 0.05.** The masked difference is the largest so far (+10).
- **Masked with the globals hidden: 63 of 400.**
  - In this mode the answer sees no image tokens and no global latents, so its only access to the
    map is the block latents.
  - EGR solves 68 masked with its globals.
- **Phase 1** (globals always hidden, image cut): the answer loss fell from 0.69 over the first 10
  steps to 0.08 over the last 10.
  - With no other access to the map, it could only fall that far by reading the block latents. This
    is an interpretation, not the pre-declared test.
- **The block latents still hold their blocks:** count-matched probe 0.946, separation −0.03.

## The swap test (separate job 7125522) and the reading

The swap test ran as its own job on these weights: masked, globals hidden, 100 maps. See
[region-swap-7125522](../region-swap-7125522/README.md). The readings are in comparison.json.
- **The answer uses the block latents.**
  - With a donor's block latents, 98 of 100 routes become the donor's route.
  - With them averaged, no route is unchanged and none is solved.
  - The pre-declared bar was a 30% change.
- **No improvement on EGR:** masked 78 against 68 (p = 0.11), unmasked 78 against 75 (p = 0.75).
- **The pre-declared reading:** "uses the block latents, no improvement".
  - Reading the map from the latents does not fix the policy.
  - **The limit is the decision process.**

**Does EGD step into holes its latents flag?** (post hoc, mistakes-vs-latents.json, as
[EGR's check](../region-egr-7122822/README.md))

| Hole cells read as holes by EGD's block latents | Share |
| --- | --- |
| Every hole on the held-out maps | 92% |
| First mistakes, masked with globals hidden, a safe toward-goal move existed | 68 of 74 (92%) |
| First mistakes, masked with globals hidden, at a forced point | 118 of 141 (84%) |
| First mistakes, unmasked, a safe toward-goal move existed | 80 of 89 (90%) |

- **Even with routes built entirely from the block latents,** the model steps into the holes those
  latents flag, at the latents' general rate.
- **Reading (interpretation):**
  - the answer reads the block latents for where to go: averaged, they give one fixed route, and
    swapped, they give the donor's;
  - it does not read them for what to avoid. Most failures still head toward the goal and fall into
    a hole.

## Files

| File | Contents |
| --- | --- |
| greedy.jsonl, masked.jsonl, masked_hidden.jsonl | One row per answer: unmasked, masked with globals shown, masked with globals hidden |
| partial-readings.json | The McNemar tests above, masked-hidden successes, the block probe and separation |
| comparison.json | `region_latents.compare_dropout` with the separate swap test: validity, use of the block latents, the McNemar tests, the reading |
| mistakes-vs-latents.json | Post hoc: whether the block latents read as holes the cells that first mistakes stepped into |
| phase1/, phase2/ | Each phase's summary and per-step log |
| manifest.json | Run manifest; the PACE root is written as `$SPARA_REMOTE_ROOT/` |
