# Evidence for the swap test on EGR's block latents (job 7123991)

Specified in [the latent JEPA spec](../../PERCEPTION_JEPA_SPEC.md#egr-does-the-answer-read-the-block-latents-the-swap-test):
masked greedy (the answer sees the image only through the latents) on EGR's saved weights
([EGR, job 7122822](../region-egr-7122822/README.md)), on the first 50 held-out maps at 7 x 7 and 8 x 8.

## The run

- **Compute:** one H200, 391 GPU-seconds by Slurm's elapsed time (6 min 31 s; 281 s in the run). The
  allowance was 480.
- **Source:** the hashes in manifest.json match the local code for every file the run imports. The
  mismatches are files it does not import.

**Validity: passed.**
- Own routes reproduced EGR's recorded masked routes on 100 of 100 maps.
- Re-inserting a map's own block latents reproduced its route on 4 of 4.

## Result: the answer ignores the block latents

| Condition | Solved | Route unchanged from own | Route equals the donor's |
| --- | --- | --- | --- |
| Own | 17 of 100 | | |
| Block latents from a same-size donor, every layer | 18 | 92 | 0 |
| Block latents replaced by their mean | 17 | 93 | |
| Global latents from the donor | 0 | 0 | 92 |

- **By size:** block swap leaves 47 of 50 routes unchanged at 7 x 7 and 45 at 8 x 8; block mean, 47
  and 46. Global swap gives the donor's route on 45 and 47.
- **The pre-declared reading:** block mean leaves 93% of routes unchanged (threshold 90%), and block
  swap gives the donor's route on 0% (threshold 10%). **The answer ignores the block latents.**
- **The route is set by the four global latents.** With another map's globals, the answer writes
  that map's route on 92 of 100 maps.
- **Where the changes occur:** the block latents change a route on about 7 maps in 100, mostly in
  later moves (for example RDDDRDDD becomes RDDDLDDR).

**Reading (interpretation):**
- **The globals cannot see the block latents.** The global latents come before the block latents in
  the prompt, so under causal attention they never see them. The route is already formed in the
  globals.
- **The answer copies that route.** It could attend to the block latents while writing moves, but
  does not.
- **Why the globals plan badly anyway:** they see the image tokens directly, and those show every
  hole. So the globals' poor plans are not for lack of hole information.
- **The pre-declared next step:** train under the bottleneck with the global latents dropped out, so
  that the block latents are the only map the answer has.

## Files

| File | Contents |
| --- | --- |
| summary.json | Validity checks, per-condition counts, the reading, time, peak memory |
| swaps.jsonl | Per map: row, size, donor, and the route and success under each condition (and the identity check on the first four) |
| manifest.json | Run manifest; the PACE root is written as `$SPARA_REMOTE_ROOT/` |
