# Evidence for LIVR run 6189457

This compact evidence bundle accompanies [the full results](../../LIVR_BASELINE_RESULTS.md).
The original run completed 10+10 epochs and scored 323/400 (80.75%) on the
released Mirage VSP test set. Files are committed so charts can be regenerated
without PACE access or model weights.

| File | Contents |
| --- | --- |
| evaluation-summary.json | Overall and per-grid reference success counts |
| evaluation-outcomes.jsonl | All 400 generated answers, extracted routes and outcomes in original test order; row_index is zero-based |
| stage1-steps.jsonl / stage2-steps.jsonl | Original optimizer-step logs: stage, zero-based epoch, one-based update, answer loss, pre-clipping gradient norm, post-update learning rate and update seconds |
| provenance.json | Recipe, data/evaluator/model hashes, hardware, completed checks, original runtime source hashes and evidence checksums |
| discrepancy-audit.json | Completed scoring/input/overlap audit; diagnostic alternate counts do not replace the official metric |

The per-example export omits the repeated prompt and image-token prefix from
raw responses; generated answers and reference outcomes are preserved. These
files contain no model weights, optimizer states, dataset images or credentials.
Full raw outputs, source snapshot, manifests and checkpoints remain in the
ignored local results directory and the original PACE run directory.

The original runtime source manifest includes generated editable-install
metadata; those hashes document the environment used and are not an assertion
that generated metadata files are committed. Additional reporting/collection
tools were added after training and do not change the recorded run.

From the repository root:

```bash
python -m pip install -e '.[report]'
python scripts/plot_livr_results.py
```

The script checks evidence hashes and counts, then writes PNG and SVG files to
`docs/figures/`. Epoch means average the 125 equal-size optimizer-step losses
within each epoch; the thin curves show individual update means. Loss curves
are training diagnostics, not held-out performance. The published 66% reference
in the evaluation figure is contextual: LIVR's exact original protocol remains
partly unrecovered, so the chart is not a controlled superiority claim.
