# Search against independent sampling at equal budget

Rebuttal evidence for research finding 3. The readings below were fixed before the sampling run. They were first committed in `docs/DETOUR_SCALING_SPEC.md`, "Search against independent sampling at equal budget" (PR #13, commit 45e8c16).

**Budget accounting.** A search policy call reads the policy's distribution over the next move once. A sampled route of n moves makes n + 1 such decisions. Samples are counted in the order drawn.

**Primary: maps solved within 64 policy decisions,** search against sampling, paired by map, exact McNemar, two-sided, at 0.05.

| Outcome | Reading |
| --- | --- |
| Search solves more, p < 0.05 | Structured search adds value over rejection sampling at equal policy budget |
| No detectable difference | At this budget, search is not detectably better than rejection sampling |
| Sampling solves more, p < 0.05 | Rejection sampling is better at equal policy budget |

**Second: equal verifier budget.** Per map, sampling gets as many verifier calls as the search used there, on its first distinct routes. The same test.

**Limits.** One policy, one seed, one budget.

- `search-routes.jsonl`: e3's final-verdict best-first search on maps 0 to 999 of `fresh/detour-pool-78j`, at most 64 policy calls a map (job 7285148). 736 of 1,000 solved; 4.07 verifier calls a map on average.
- `samples.jsonl`: 64 independent answers a map from e3 at temperature 1 on the same maps (job 7305126). Added when the job finishes.

Run `python scripts/compare_search_sampling.py` once both files are present.
