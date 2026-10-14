# Run Scaffolding Minds as a helper-supervised reference

On 2026-10-13 the user asked for the Scaffolding Minds paper
([v2](../related_literature/Scaffolding_Minds_2608.19669v2.pdf)) to be implemented
as a baseline. The method cannot run without helper images: its Stage 1 trains a
scaffolding encoder that turns a training-time helper image into the latent
target. The glossary's helper-image entry and
[ADR-0002](0002-reproduce-livr-reference-protocol.md) exclude helper images from
SPARA's own training and evaluation, and the project plan names a "matched
helper-supervised reference" as a stretch goal. This baseline fills that role. It
is an exception for this reference only, not a change to SPARA's method.

The baseline reads helper images only during Stage 1 training, as the paper
specifies. Evaluation inputs stay the unsolved map and released prompt. It uses
the same Mirage VSP split, Qwen2.5-VL-3B backbone, reference evaluator and
decoding settings as the LIVR baseline, so its score is directly comparable to
LIVR, but it has strictly more supervision. Report it separately and label it
as helper-supervised wherever it appears beside LIVR, text-only RL or SPARA.

The default helper is the paper's value-function heatmap, computed from each
training map's layout with the paper's Appendix E.1 code. The released
solved-route overlay (the paper's red-arrow alternative) is a recipe option.
The paper's own FrozenLake benchmark (8x8 to 32x32 grids) and 7B backbone are
not reproduced; ADR-0002's single-benchmark protocol takes precedence.
Implementation choices the paper leaves open are recorded in
[the baseline specification](../SCAFFOLDING_BASELINE_SPEC.md).
