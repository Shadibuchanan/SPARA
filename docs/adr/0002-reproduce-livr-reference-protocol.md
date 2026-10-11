# Reproduce LIVR's planning baseline with one reference protocol

The user clarified on 2026-10-11 that the first phase should reproduce LIVR's
published planning baseline as closely as available evidence permits. Use the
released Mirage VSP data, the documented 3B full-fine-tuning recipe, and the
released reference evaluator; use that same evaluator for later SPARA comparisons.
This replaces the previously accepted custom scorer and additional direct-SFT
experiment, prioritizing comparability over new experimental variants.

Preserve the benchmark split and evaluator semantics, including known quirks,
and disclose the observed train/test overlap. Extra filtered-subset metrics and
custom-map experiments are not required for this initial reproduction. Profile
full fine-tuning on PACE-ICE first; any resource-driven LoRA fallback must be
identified as an adaptation rather than silently substituted.

The Mirage release is the available reference, not proof of LIVR's exact evaluator
version, test manifest, decoding settings, or checkpoint-selection policy. Record
unverified details and implementation assumptions explicitly. The user accepted
the following source policy on 2026-10-11: follow LIVR's published settings first;
where a necessary detail is unspecified, use the pinned Mirage implementation
where applicable and record that choice as an assumption about LIVR. Mirage's
different latent architecture or supervision objectives are not defaults to copy
into LIVR. If neither source supplies a relevant setting, keep it open rather
than presenting an invented value as part of the paper. This decision
supersedes [the direct-SFT control decision](0001-include-matched-sft-control.md)
and the earlier SPARA-specific scoring proposal; it retains the later project's
LIVR, LIVR plus text-only RL, and SPARA comparisons.
