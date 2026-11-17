# Evidence for the swap test on EGD's weights (job 7125522)

This is [the swap test](../../PERCEPTION_JEPA_SPEC.md#egr-does-the-answer-read-the-block-latents-the-swap-test)
on [EGD](../region-egd-7124258/README.md)'s weights. EGD's own job timed out before it ran the test.
- **Mode:** masked greedy with the global latents hidden, so the block latents are the answer's only
  map.
- **Maps:** the first 50 held-out maps at 7 x 7 and 8 x 8.
- **Conditions:** own, block swap, block mean.

- **Compute:** one H200, 436 GPU-seconds (7 min 16 s; 234 s in the run). The time limit was 15 minutes.
- **Source:** the hashes in manifest.json match the local code for every file the run imports.
- **Validity: passed.** Own routes reproduced EGD's recorded masked, globals-hidden routes on 100 of
  100 maps. Re-inserting a map's own block latents reproduced 4 of 4.

## Result: the answer reads the block latents

| Condition | Solved | Route unchanged from own | Route equals the donor's |
| --- | --- | --- | --- |
| Own | 15 of 100 | | |
| Block latents from a same-size donor, every layer | 0 | 0 | 98 |
| Block latents replaced by their mean | 0 | 0 | |

- **By size:** the donor's route on 48 of 50 at 7 x 7 and 50 of 50 at 8 x 8.
- **With the block latents averaged,** the answer writes the same route, RUULLDDR, on every map.
- **Compared with EGR** (job 7123991, globals visible): there, 93 of 100 routes were unchanged by
  averaging, and none followed the donor.

**A wording error in the spec's table.** The script prints the swap test's pre-declared label for
this outcome, "the answer reads the block latents, but their content does not shape its moves". The
second half is wrong here: the block latents' content shapes almost every route. The label should
have read only "the answer reads the block latents". EGD's reading comes from its own table, in
[its README](../region-egd-7124258/README.md).

## Files

| File | Contents |
| --- | --- |
| summary.json | Validity checks, per-condition counts, time |
| swaps.jsonl | Per map: row, size, donor, and the route and success under each condition |
| manifest.json | Run manifest; the PACE root is written as `$SPARA_REMOTE_ROOT/` |
