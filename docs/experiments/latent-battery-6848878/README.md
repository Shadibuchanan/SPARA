# Evidence for latent battery run 6848878

Latent-token measurements on [LIVR run 6189457](../livr-6189457/README.md)'s final
Stage 2 checkpoint (Shadi Buchanan's original run, not the rerun `6253300`).
See [the results](../../LATENT_BATTERY_RESULTS.md). One H100, 2,021
GPU-seconds; greedy decoding with repetition penalty 1.05 and at most 128 new
tokens.

| File | Contents |
| --- | --- |
| summary.json | Per mode and condition: successes overall and by grid size, routes changed from own, mean reference NLL, truncations, McNemar comparison with own, and for swaps how often the answer solves or copies the donor map |
| rows.jsonl | For each of the 400 test maps in original order (row_index zero-based): the answer, extracted route, success, reference NLL and token count under every mode and condition, and the swap donor's row index |
| identity.json | Check that re-inserting a map's own recorded latent states leaves its route unchanged (first four maps, both modes) |
| probes.json | Output of `scripts/analyze_latent_battery.py` (Slurm job `6850902`): chance levels, probe scores for every mode, feature, layer and grid size, collapse statistics, and image-cell distinguishability |
| failures.json | Output of `scripts/analyze_battery_failures.py` (Slurm job `6878532`): for each layer, how often the hole probe reads the holes the model entered as safe against position- and size-matched base rates, per-map misread counts for failed and solved maps, and outcomes by grid size |
| manifest.json | Recipe, package versions, source hashes, evaluator provenance, probe layers, effective generation settings and the swap-donor assignment; the checkpoint path is given relative to the PACE results root |

The probe features (`features-train.pt`, 1.3 GB; `features-test.pt`, 0.46 GB)
stay in the PACE run directory, `results/latent-battery-6848878`. These files
contain no model weights, images or credentials.

To rerun the analysis from the features:

```bash
python scripts/analyze_latent_battery.py <run directory>
```
