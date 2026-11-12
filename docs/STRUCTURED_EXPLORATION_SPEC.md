# Structured exploration on harder maps (B′ and C′)

Status, 2026-11-08: the probe ran (job 6884608; see [Probe result](#probe-result)).
No B′ or C′ run has been approved. The training code for structured noise (Gaussian
heads in the structured basis) is not built.

## Why

[Condition C](LATENT_RL_RESULTS.md) added isotropic noise to LIVR's latent states and
scored the same as text-only RL (B). Two limits showed up in its training data:

- **Per-map success is bimodal.** On most maps the model succeeds in all eight
  rollouts or in none, and GRPO learns nothing from either. On the test maps, only
  28 of 400 had a success rate strictly between 0 and 1 over 16 samples, so new maps
  of the same sizes would not by themselves give more signal than the training maps.
- **Isotropic noise does not rescue failing maps on balance.** C succeeded at least
  once on 19 training maps where all of B's rollouts failed, and B on 18 where all
  of C's failed. Most of C's extra signal came from noise breaking maps the model
  already solved.

This plan changes both: noise only in the directions in which latents of different
maps actually differ, and training maps on which the model succeeds about half the
time.

## New maps

Every released image is a FrozenLake render: gymnasium 1.0.0's renderer, given the
row's `map_desc`, reproduces all 1,000 training and 400 test images pixel for pixel.
The released layouts fit a simple generator: start and goal are distinct cells drawn
uniformly, every other cell is a hole with probability about 0.2, and every map is
solvable. For example, a 3 x 3 start sits next to its goal in 34% of training maps;
uniform placement gives 33%.

[src/spara/fresh_maps.py](../src/spara/fresh_maps.py) draws layouts this way, rejects
unsolvable ones and any layout in the train or test split or already drawn, renders
them, and writes rows with the released prompt text. Each set (probe, training,
held-out) has its own seed and a manifest with the row file's SHA-256 and the
renderer versions ([requirements/fresh-maps.txt](../requirements/fresh-maps.txt)).
The training loop and probes read a set as split `fresh/<name>` under the data root.

## Structured noise

- **Basis:** the latent states at the output of decoder layer 16 for the 1,000
  training maps, from the LIVR rerun `6253300`, centred within each grid size. For
  each of the four latent slots, the top 16 principal directions and the standard
  deviation along each. The number 16 is an assumption, from the battery's
  standardised participation ratio of 10 to 18.
- **Adjustment:** δ = c Σⱼ uⱼ sⱼ εⱼ with εⱼ ~ N(0, 1): noise along each direction in
  proportion to how much maps differ along it, with c the scale in those units.
- If the training maps are 7 x 7 or 8 x 8, recompute the basis on those sizes before
  any training run and report the variance it captures.

## Probe (one GPU job, about 0.3 GPU-hours)

All on the LIVR rerun `6253300`, with a fresh probe set of 100 maps each of 5 x 5,
6 x 6, 7 x 7 and 8 x 8.

1. **Basis**, as above.
2. **Size sweep.** For every probe map, the greedy answer and eight samples with the
   RL generation settings (temperature 0.7, top-p 0.9). Per size: the share of maps
   with mixed sample outcomes (1 to 7 of 8 succeed), all failing and all succeeding;
   greedy success; the share of greedy failures where at least one sample succeeds
   (text sampling's own rescue rate, B's exploration); prompt length and peak memory.
3. **Noise probe.** On the 200 probe maps of 5 x 5 and 6 x 6: the greedy answer with
   no adjustment, and with 4 draws each of isotropic σ ∈ {1.2, 1.6, 2.0, 2.4} and
   structured c ∈ {0.5, 1, 2, 4, 8}. Per scale: route-change rate; success rate;
   rescue rate (draws that succeed on maps whose unadjusted answer fails); break rate
   (draws that fail on maps whose unadjusted answer succeeds); rescue share (rescued
   draws over route-changed draws); and maps with any rescue or any break, since
   draws on one map are not independent.

## Rules fixed before the probe

- **Scale:** for each noise family, the smallest scale that changes at least 10% of
  greedy routes and lowers success by at most 5 points (C's rule).
- **Noise for C′:** structured noise at its chosen scale, if one qualifies and its
  rescue share is at least that of the isotropic scale with the closest route-change
  rate. Otherwise isotropic noise at its chosen scale.
- **Training sizes:** every size whose mixed share is within 5 points of the best
  size's, with equal numbers of maps per size.

## Probe result

The full results and files are in
[experiments/explore-probe-6884608](experiments/explore-probe-6884608/README.md).

**Harder maps did not add signal.**
- **Mixed share is flat with size.** It is 11%, 12%, 8% and 10% at 5 x 5 to 8 x 8. On the larger grids, all-fail maps replace all-success ones: 53% and 65% all fail at 7 x 7 and 8 x 8.
- **Same as B's training.** That is the same signal density B had on the released training maps (9.9% of groups mixed).
- **Rule outcome.** The size rule therefore selects all four sizes.

**Structured noise passed its rule, but only just.**
- **What the rule chose:** c = 4. It changes 10.0% of greedy routes, exactly the threshold, and leaves success unchanged. Its rescue share is 0.21, against 0.14 for the isotropic scale with the same route-change rate.
- **Uncertainty:** the bootstrap interval on that difference includes zero.
- **Overlap with text sampling:** of 57 failing maps, structured noise rescues 11 and text sampling 12; the two together rescue 16.
- **Coverage:** the noise probe covered sizes 5 and 6 only.

**Open decision.** B′ as specified trains only on fresh maps of sizes 5 to 8. The
benchmark is sizes 3 to 6, so B′ against B would also change the training sizes,
not only the exploration. B′ against C′ is unaffected.

## How B′ and C′ would be read

B′ and C′ would each train one epoch on the same 1,000 fresh maps, with B's and C's
recipes otherwise unchanged.

- **Primary:** C′ with δ = μ against B′ on the 400-map reference benchmark, exact
  McNemar test, different only if p < 0.05. The benchmark stays primary so results
  remain comparable with A, B and C ([ADR-0002](adr/0002-reproduce-livr-reference-protocol.md)).
- **Secondary:** both on 400 held-out fresh maps of the training sizes; each against
  B and the LIVR rerun on the benchmark; C′ with δ = μ against δ = 0.
- **Diagnostic:** the share of training groups with mixed rewards.

## Budget

The probe is about 0.3 GPU-hours. B′ and C′ would take about 1 GPU-hour each, and
need separate approval after the probe.

## Running

```bash
python -m spara.fresh_maps --data-root data/vsp --name probe --sizes 5,6,7,8 --per-size 100 --seed 101
python -m spara.cli explore-probe --data-root data/vsp --checkpoint /path/to/stage2/final \
  --split fresh/probe --output /path/to/probe
```

On PACE, submit [scripts/pace/explore_probe.sbatch](../scripts/pace/explore_probe.sbatch)
with `SPARA_LIVR_CHECKPOINT`.
