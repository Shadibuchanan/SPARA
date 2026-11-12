# Evidence for structured-exploration probe 6884608

The probe in [the structured-exploration spec](../../STRUCTURED_EXPLORATION_SPEC.md),
run on [LIVR rerun 6253300](../livr-6253300/)'s final Stage 2 checkpoint. It used
400 fresh maps (`fresh/probe`: seed 101, 100 each of 5 x 5 to 8 x 8, maps.jsonl
sha256 `229b7372…`) and the 1,000 released training maps for the basis.

- **Compute:** one H200, 1,354 GPU-seconds (0.38 GPU-hours, above the 0.3 estimate).
- **Phases:** basis about 2 minutes, sweep 8, noise probe 9.
- **Decoding:** at most 128 new tokens, against B's 1,024; completions here are 10 to 20 tokens.

## Results

The rules below were fixed in the spec before the run.

**Basis.** 16 principal directions per latent slot capture 78%, 94%, 88% and 85%
of the variance in layer-16 latent states across the training maps (after
centring within grid size).

**Size sweep.** Greedy decoding plus eight samples per map (temperature 0.7, top-p 0.9).

| Size | Greedy success | Mixed (1–7 of 8) | All fail | All succeed | Greedy failures rescued by a sample | Peak GPU |
| --- | --- | --- | --- | --- | --- | --- |
| 5 | 77% | 11% | 17% | 72% | 6 of 23 | 21.1 GiB |
| 6 | 67% | 12% | 29% | 59% | 5 of 33 | 21.4 GiB |
| 7 | 43% | 8% | 53% | 39% | 4 of 57 | 21.6 GiB |
| 8 | 30% | 10% | 65% | 25% | 5 of 70 | 21.7 GiB |

- **Training sizes (rule): all four.** Every size is within 5 points of the best mixed share.
- **Mixed share is flat with size.** 41 of 400 maps (10.25%) are mixed. Harder maps turn successes into total failures; they do not produce more maps with mixed outcomes.
- **Same signal density as B's training.** During B's training on the released maps, 99 of 1,000 groups (9.9%) had mixed rewards. C had 168 of 1,000, with its latent noise.
- **Difficulty matches the benchmark.** LIVR's greedy success on the released test maps is 86% at 5 x 5 and 62% at 6 x 6, against 77% and 67% here.

**Noise probe.** This ran on the 200 fresh maps of sizes 5 and 6.
- **Draws:** greedy decoding, one unadjusted route per map, then four draws per scale.
- **Baseline:** greedy success is 71.5%, with 57 failures and 143 solved maps.
- **Rescues and breaks:** both are counted over draws (out of 228 failing draws and 572 solved draws).

| Family | Scale | Routes changed | Success | Rescues | Breaks | Rescue share | Failing maps rescued |
| --- | --- | --- | --- | --- | --- | --- | --- |
| isotropic | σ = 1.2 | 11.4% | 70.4% | 13 | 22 | 0.14 | 7 |
| isotropic | σ = 1.6 | 25.5% | 62.1% | 15 | 90 | 0.07 | 8 |
| isotropic | σ = 2.0 | 49.5% | 46.5% | 7 | 207 | 0.02 | 5 |
| structured | c = 2 | 5.9% | 72.6% | 12 | 3 | 0.26 | 6 |
| structured | c = 4 | 10.0% | 71.5% | 17 | 17 | 0.21 | 11 |
| structured | c = 8 | 35.0% | 55.4% | 12 | 141 | 0.04 | 9 |

- **Decision (rule): structured noise at c = 4.** Its rescue share is 0.21, against 0.14 for the isotropic scale with the closest route-change rate (σ = 1.2).
- **The decision is fragile.** c = 4 changed exactly 80 of 800 routes, which is the 10% threshold. One fewer change and no structured scale would qualify, so the rule would fall back to isotropic σ = 1.2. The scale grid is coarse.
- **The advantage is not significant.** In a paired bootstrap over maps, the rescue-share difference (structured minus isotropic) has a 95% interval of −0.02 to 0.16. Rescues minus breaks differs by −6 to +24.
- **Structured noise is far smaller.** The expected size of a c = 4 offset is 4 to 31 per slot, against 54 for isotropic σ = 1.2. Yet it changes routes about as often.
- **Text sampling already finds most of these rescues.** Of the 57 failing maps:
  - eight text samples rescue 12;
  - four structured draws rescue 11, of which 7 overlap with text sampling;
  - together they rescue 16.
- **Few new signal groups.** Of the 45 failing maps where all eight text samples also fail, structured noise rescues 4 and isotropic noise 1.
- **Greedy routes reproduce.** The noise probe's unadjusted routes agree with the sweep's greedy routes on 197 of 200 maps (199 of 200 by success).
- **Untested at larger sizes.** The probe tests noise only at sizes 5 and 6, so it says nothing about whether these directions help at 7 and 8.

## Files

| File | Contents |
| --- | --- |
| summary.json | Basis variance captured, per-size sweep statistics with the size rule, and noise statistics per family and scale with the chosen scales and decision |
| sweep.jsonl | For each of the 400 fresh maps (row_index into fresh/probe): grid size, prompt tokens, greedy route and success, the eight sample successes, and peak GPU memory |
| noise.jsonl | For each of the 200 maps of sizes 5 and 6: the unadjusted greedy route and success, and per family and scale, whether each draw changed the route and whether it succeeded |
| manifest.json | Recipe, package versions, source hashes, data and fresh-set manifests, generation settings, probe settings and seed |

This copy of `manifest.json` gives the checkpoint path relative to the PACE results
root, so its hash differs from the PACE original. The other three files match the
originals byte for byte.

The basis (`basis.pt`: directions, scales and captured variance; no model weights)
stays in the PACE run directory, `results/explore-probe-6884608/probe`. C′ would
use it.
