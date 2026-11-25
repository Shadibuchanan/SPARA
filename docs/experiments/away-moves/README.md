# Away-move accuracy on the held-out detour maps (post hoc)

Every map of `fresh/detour-heldout-78` (400 generated 7 x 7 and 8 x 8 maps, each needing at
least one move away from the goal) with four arms' greedy routes, to test whether failures
are lookahead failures. A post hoc analysis, not a reading fixed before a run.

- `outcomes.jsonl`: `arm`, `row_index`, `map_desc`, `route`, `success`. Arms: EB3 (expert
  iteration round 3), e3 (three label-free rounds on found routes), n24000 (route labels,
  32,000 maps), p8000 (writes holes, distance levels and path before the route; a supervised
  reference). Runs: `docs/DETOUR_SCALING_SPEC.md`.
- **Reproduce:** `python scripts/analyze_away_moves.py`.

| Arm | Solved | Routes making the needed away moves | Failures that never move away |
| --- | --- | --- | --- |
| EB3 | 83 | 189 | 158 of 317 |
| e3 | 197 | 274 | 61 of 203 |
| n24000 | 273 | 322 | 24 of 127 |
| p8000 | 395 | 398 | 1 of 5 |
