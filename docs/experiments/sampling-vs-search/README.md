# Search against independent sampling at equal budget

Rebuttal evidence for research finding 3. Readings are fixed in `docs/DETOUR_SCALING_SPEC.md`, "Search against independent sampling at equal budget".

- `search-routes.jsonl`: e3's final-verdict best-first search on maps 0 to 999 of `fresh/detour-pool-78j`, at most 64 policy calls a map (job 7285148). 736 of 1,000 solved; 4.07 verifier calls a map on average.
- `samples.jsonl`: 64 independent answers a map from e3 at temperature 1 on the same maps (job 7305126). Added when the job finishes.

Run `python scripts/compare_search_sampling.py` once both files are present.
