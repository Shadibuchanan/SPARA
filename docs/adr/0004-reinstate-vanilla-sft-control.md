# Reinstate the vanilla SFT control beside the LIVR baseline

[ADR-0001](0001-include-matched-sft-control.md) accepted a matched direct-SFT
control on 2026-10-11; the user withdrew it the same day and
[ADR-0002](0002-reproduce-livr-reference-protocol.md) superseded it to prioritise
the LIVR reproduction. The project proposal's Oct 5–14 block still commits to
"vanilla SFT & pause tokens w/ images", so the repository and the proposal
disagreed. That reproduction is now done: job `6189457` scored 323/400 (80.75%),
14.75 points above LIVR's published 66%. Without a run lacking the latent
machinery, that number cannot be attributed to the latents, to the split, or to
the recipe.

On 2026-10-14 the user approved reinstating the control as specified in
[the plan](../plans/VANILLA_SFT_CONTROL_PLAN.md): Qwen2.5-VL-3B on the same split,
recipe, precision, seed, data order and reference evaluator as `6189457`, with
no latent tokens and no Stage-1 bottleneck, at LIVR's 2,500 optimizer updates.
Two arms run: one 20-epoch stage, and two 10-epoch segments with a fresh
optimizer and schedule at the midpoint, which separates the schedule restart from
the latents. This resolves the disagreement in the proposal's favour for vanilla
SFT only. Pause/filler tokens stay out of scope.

This decision supersedes ADR-0002 on the direct-SFT control only. The rest of
ADR-0002 — the reference protocol, evaluator, split and source policy — remains
in force, and the control is held to it.
