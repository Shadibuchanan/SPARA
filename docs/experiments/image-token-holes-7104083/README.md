# Evidence for image-token hole probe 7104083

Gate 1 of the [latent JEPA spec](../../PERCEPTION_JEPA_SPEC.md#gate-1-result-cpu-job-7104083): do
LIVR's image tokens show the holes?

- **Features:** the [latent battery](../latent-battery-6848878/README.md)'s saved `cells` (LIVR 6189457,
  unmasked).
  - Each cell's feature is the overlap-weighted mean of the image-token states at the probe layers 4,
    8, ... 36.
  - The battery fitted on 1,000 training maps and scored on 400 test maps, sizes 3 x 3 to 6 x 6.
- **Script:** [scripts/probe_image_token_holes.py](../../../scripts/probe_image_token_holes.py). The
  0.95 threshold was in the script before the run.
- **Compute:** Slurm job 7104083 on one `ice-cpu` node (8 cores, 48 GB requested), 3 minutes 56
  seconds. No GPU.

## Results

**Holes:** one ridge probe per size, from a cell's feature to hole or not, strength by leave-one-out
error. Balanced accuracy on the test maps' cells:

| Layer | 3 x 3 | 4 x 4 | 5 x 5 | 6 x 6 | All sizes, one probe |
| --- | --- | --- | --- | --- | --- |
| 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 8 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 12 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 16 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 20 | 1.00 | 1.00 | 0.999 | 1.00 | 1.00 |
| 24 | 1.00 | 1.00 | 0.999 | 1.00 | 1.000 |
| 28 | 1.00 | 1.00 | 0.999 | 1.00 | 1.000 |
| 32 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 36 | 1.00 | 1.00 | 0.999 | 1.00 | 1.000 |

- **Test holes:** 1,463, of which 122, 253, 445 and 643 are at 3 x 3 to 6 x 6.
- **Misses and false alarms:** none at layers 4 and 12. The 0.999 entries are one missed hole.
- **Reading (rule): the image tokens separate holes.** Every layer passes at every size.

**Position after centring (descriptive):**
- **Method:** subtract the training mean feature of each (size, row, column), then decode the cell's
  row and column with ridge probes (row accuracy times column accuracy, all cells).
- **Result:** chance at every layer: 0.10 to 0.12 at 3 x 3, 0.06 to 0.07 at 4 x 4, 0.04 to 0.05 at
  5 x 5, and 0.03 to 0.04 at 6 x 6. Chance is 1/9, 1/16, 1/25 and 1/36.
- **For comparison:** before centring, the battery decoded safe cells' positions at 0.96 to 1.00.

## What it shows

- **The frozen vision side and the image-token states carry every hole.** The latents of the same
  maps and checkpoint read holes at only 0.75 to 0.88 at layer 12. The information is lost between
  the image tokens and the latents, not in the vision encoder.
- **Centring removes linearly decodable position.** A learned target projection over centred cells
  has no linear position shortcut.

**Limits:**
- One checkpoint (LIVR's), not E.
- Sizes 3 x 3 to 6 x 6 only. The gate job repeats the probe on E at 7 x 7 and 8 x 8.
- The probe is linear, so this is a lower bound.

## Files

| File | Contents |
| --- | --- |
| image-token-holes.json | Per layer and size: balanced accuracy, miss and false-alarm rates, test cells and holes, ridge strength; position after centring; the reading |
