# Can a learned latent policy learn here? Offline checks on recorded data (2026-11-19)

CPU-only analyses run before any C‴ (learned latent heads on C″'s settings) was trained. They use
data already on PACE scratch: C″'s training rollouts, each with its noise draw ε and reward
(`fair-n-seed{42,43}-7154355/rl/rollouts.jsonl`, 2,145 groups with mixed rewards); LIVR's layer-24
latent states for all 2,000 pool maps (`latent-diag-sweep-L16-24-7148362/states.pt`); and the
diagnostics' latent search and text baseline. Nothing here trains the model. Scripts are in this
folder; each reads the scratch paths directly. These analyses led to
[LATENT_DISTILLATION_SPEC.md](../../LATENT_DISTILLATION_SPEC.md) instead of a C‴.

**AUC** below: within a group of 8 rollouts on one map, the chance that a successful rollout
scores higher than a failed one on the quantity tested. 0.5 means no relation. Chance ranges come
from shuffling rewards within groups.

## 1. Is there a latent direction worth learning (μ)? (`eps_signal.py`, `per_map.py`)

| Direction, fitted on one part and tested on another | Held-out AUC | Chance range (97.5%) |
| --- | --- | --- |
| One for all maps (seed 42 → 43, 43 → 42, early → late steps, map halves) | 0.493–0.499 | up to 0.515–0.519 |
| One per grid size | 0.48–0.55, all within chance | — |
| One per map, fitted on one seed's groups, tested on the other seed's groups on the same map (the best case for a learned mean) | 0.518 and 0.521 (p ≈ 0.035, 0.015) | up to 0.519–0.520 |

μ has essentially nothing to learn: no shared direction, and a per-map direction barely carries
over between rollouts on the same map.

## 2. Is there a noise size worth learning (σ)? (`sigma_sig.py`)

Within groups, does a rollout with more noise energy (Σε²) succeed more often?

| Groups, by successes out of 8 | Groups | AUC (more noise → success) | Chance range |
| --- | --- | --- | --- |
| 2 or fewer | 781 | 0.510 | 0.481–0.520 |
| 3–5 | 741 | 0.491 | 0.483–0.514 |
| 6 or more | 623 | **0.446** | 0.472–0.521 |

More noise hurts on maps the policy mostly solves. A σ that shrinks on easy maps would cut C″'s
breaks. This is a real target.

## 3. Do LIVR's latents show which maps are easy? (`ease_probe.py`, part a → `ease_probe_a.out`)

Ridge regression from LIVR's layer-24 latents (top-r principal components) to each map's success
rate in C″'s training, on 400 held-out maps: Spearman 0.40–0.56, R² 0.11–0.34. Yes; the input a
σ(z) head would need is there.

## 4. Can policy-gradient heads learn it? (`head_sim.py` → `head_sim.out`; `ease_probe.py`, part b → `ease_probe_b.out`)

Heads trained by the latent policy-gradient term alone, step by step in C″'s order on one seed's
mixed groups, read on held-out maps. Inputs are LIVR's latents (a proxy: online, z drifts with the
policy; part 3 shows the input is not the bottleneck).

| Heads | Result |
| --- | --- |
| Scaffolding-style (2,048 hidden), learning rate 2e-5 (C′'s) | Barely move: σ 0.245, \|μ\| 0.10 |
| Scaffolding-style, 64 hidden or linear, learning rate 1e-3 or 1e-2, μ bounded to ±0.25, σ in [0.025, 0.5] | μ goes to its bound on almost every coordinate; σ goes to its floor or ceiling per coordinate (99.9% at a bound), the same on every map (spread across maps ≤ 0.008); held-out μ AUC 0.48–0.54 |
| σ only, 32 inputs, smooth bounds, pull toward σ₀ (β ∈ {0, 1, 10}), per slot or per coordinate, learning rate 1e-3 or 1e-2 (24 settings) | σ never depends on ease: Spearman(σ, ease) −0.16 to +0.02; easy versus hard maps within 0.011 |

**Why:** a rollout's only clue for σ is how large its noise happened to be, which varies by about
±18% between rollouts (64 coordinates). Over about 1,000 mixed groups in 125 steps, that is too
faint for the policy gradient.

## 5. How big is the gap the latents could fill? (`gap.py`)

LIVR's 170 greedy failures among the 200 pilot maps:

| | Maps |
| --- | --- |
| Never solved by 64 text samples | 121 |
| … solved by random latent offsets at layer 24 (64 tries at scales 1, 2, 4, 8) | 46 |
| … solved by final-verdict search (text) | 65 |
| Solved by latent search but not by final-verdict search (all 170) | 7 |
| Success per latent try on the 121 maps | 47/1,936 at scale 1; 23, 39, 33 at 2, 4, 8 |

The gap is real: latents unlock routes the text policy essentially never samples. But the
successes are rare (about 1 in 40 tries) and map-specific, which is why RL with sampled latent
noise and learned heads cannot use them, and why search followed by distillation is the
mechanism to try.

## Also here

`nodetour_signatures.json`: `scripts/analyze_nodetour_failures.py` on the recorded benchmark
answers of LIVR (same stack), B′, C′, C″, E, E2 and EB2. 54–60% of B′'s and C″'s failed no-detour
routes contain a move away from the goal, against 12% for LIVR and 23–31% for expert iteration.
