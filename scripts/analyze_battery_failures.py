"""Do LIVR's route failures follow missing hole information in its latents? CPU only.

Reads a latent battery run directory (features-train.pt, features-test.pt and
rows.jsonl from `spara battery`) and writes failures.json there. It uses the
unmasked own answers and the unmasked latent states.

A ridge probe per grid size and layer, as in analyze_latent_battery.py, maps the
four latents to a hole score for every cell (fitted on training maps, applied to
test maps); a cell is read as a hole when its score exceeds 0.5. Routes are walked
with the reference evaluator's rules: off-grid moves are ignored, entering a hole
fails at once, and the route must end on the goal.

Predictions, fixed on 2026-11-07 before the analysis was run. If missing hole
information in the latents causes failures:
1. A hole the model's route enters (the fatal hole) is read as safe more often
   than (a) holes at the same cell position in test maps the model solved, and
   (b) holes beside the route in solved maps, which the model avoided.
2. Within a grid size and hole count, failed maps have more misread cells than
   solved maps.
The primary layer is 12 (one-based): the highest hole accuracy averaged over grid
sizes in probes.json, chosen from probe accuracy rather than from failures. Other
layers are secondary. A linear probe bounds what is linearly decodable; the model
may read information that the probe misses.

Usage: python scripts/analyze_battery_failures.py RUN_DIR
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_latent_battery import MOVES, ridge  # noqa: E402

PRIMARY_LAYER = 12
PERMUTATIONS = 20000
SEED = 42


def walk(grid, route):
    """The reference evaluator's walk: outcome, visited cells and the hole entered, if any."""
    n = len(grid)
    r, c = next((r, c) for r in range(n) for c in range(n) if grid[r][c] == 1)
    visited = [(r, c)]
    for move in route:
        dr, dc = MOVES[move]
        if not (0 <= r + dr < n and 0 <= c + dc < n):
            continue
        r, c = r + dr, c + dc
        visited.append((r, c))
        if grid[r][c] == -1:
            return "hole", visited, (r, c)
    return ("goal" if grid[r][c] == 2 else "short"), visited, None


def poisson_binomial_upper(probs, observed):
    """P(at least `observed` successes) for independent Bernoulli trials."""
    dist = [1.0]
    for p in probs:
        dist = [a * (1 - p) + b * p for a, b in zip(dist + [0.0], [0.0] + dist)]
    return sum(dist[observed:])


def stratified_permutation(values, failed, strata, generator):
    """One-sided p that failed maps' mean exceeds solved maps', permuting labels within strata."""
    values, failed = torch.tensor(values, dtype=torch.double), torch.tensor(failed)
    def difference(labels):
        labels = labels.double()
        return (labels * values).sum(-1) / labels.sum(-1) - ((1 - labels) * values).sum(-1) / (1 - labels).sum(-1)
    labels = failed.expand(PERMUTATIONS, -1).clone()
    groups = {}
    for i, key in enumerate(strata):
        groups.setdefault(key, []).append(i)
    for members in groups.values():
        members = torch.tensor(members)
        order = torch.rand(PERMUTATIONS, len(members), generator=generator).argsort(-1)
        labels[:, members] = failed[members][order]
    observed = float(difference(failed))
    hits = int((difference(labels) >= observed - 1e-12).sum())
    return observed, (hits + 1) / (PERMUTATIONS + 1)


def auc(values, failed):
    """Probability that a failed map has more misreads than a solved one (ties count half)."""
    f = [v for v, x in zip(values, failed) if x]
    s = [v for v, x in zip(values, failed) if not x]
    if not f or not s:
        return None
    return sum((a > b) + 0.5 * (a == b) for a in f for b in s) / (len(f) * len(s))


def matched_test(fatal, key):
    """Fatal holes read as safe against the base rates stored under `key` (where defined)."""
    matched = [f for f in fatal if f[key] is not None]
    observed = sum(f["read_safe"] for f in matched)
    return {"fatal_holes": len(matched), "read_safe": observed,
            "expected_read_safe": sum(f[key] for f in matched),
            "p_one_sided": poisson_binomial_upper([f[key] for f in matched], observed)}


def layer_records(train, test, rows, position):
    """Fatal holes with position- and size-matched base rates, avoided holes and per-map misreads.

    A fatal hole's position base rate is how often holes at its cell are read as safe
    in solved maps of its size; its avoided base rate is how often holes beside the
    route in solved maps of its size are read as safe.
    """
    fatal, avoided, maps = [], [], []
    for size in sorted({len(r["map_desc"]) for r in test["rows"]}):
        start = len(fatal)
        beside_holes = []
        tr = [i for i, r in enumerate(train["rows"]) if len(r["map_desc"]) == size]
        te = [i for i, r in enumerate(test["rows"]) if len(r["map_desc"]) == size]
        x = train["latent"]["unmasked"][tr][:, position].reshape(len(tr), -1)
        z = test["latent"]["unmasked"][te][:, position].reshape(len(te), -1)
        holes = torch.tensor([[float(v == -1) for line in train["rows"][i]["map_desc"] for v in line] for i in tr])
        scores, _ = ridge(x, holes, z)
        grids = [test["rows"][i]["map_desc"] for i in te]
        truth = [[v == -1 for line in g for v in line] for g in grids]
        read = (scores > 0.5).tolist()
        outcomes = [walk(g, rows[i]["route"]) for g, i in zip(grids, te)]
        for o, i in zip(outcomes, te):
            if (o[0] == "goal") != rows[i]["success"]:
                raise ValueError(f"Walk disagrees with the evaluator on test row {i}")
        solved = [o[0] == "goal" for o in outcomes]
        base = {}
        for k in range(size * size):
            cells = [not read[m][k] for m in range(len(te)) if solved[m] and truth[m][k]]
            if cells:
                base[k] = sum(cells) / len(cells)
        for m, (status, visited, hole) in enumerate(outcomes):
            maps.append({"size": size, "holes": sum(truth[m]), "failed": not solved[m], "status": status,
                         "misread": sum(t != p for t, p in zip(truth[m], read[m]))})
            if status == "hole":
                k = hole[0] * size + hole[1]
                fatal.append({"size": size, "read_safe": not read[m][k], "score": float(scores[m, k]),
                              "position_base_rate": base.get(k)})
            elif solved[m]:
                path = set(visited)
                beside = {(r + dr, c + dc) for r, c in path for dr, dc in MOVES.values()} - path
                beside_holes += [not read[m][r * size + c] for r, c in beside
                                 if 0 <= r < size and 0 <= c < size and grids[m][r][c] == -1]
        rate = sum(beside_holes) / len(beside_holes) if beside_holes else None
        for f in fatal[start:]:
            f["avoided_base_rate"] = rate
        avoided.append({"size": size, "holes": len(beside_holes), "read_safe": sum(beside_holes)})
    return fatal, avoided, maps


def summarise(fatal, avoided, maps, generator):
    missed = sum(f["read_safe"] for f in fatal)
    values, failed = [m["misread"] for m in maps], [m["failed"] for m in maps]
    if any(failed) and not all(failed):
        difference, p_maps = stratified_permutation(values, failed, [(m["size"], m["holes"]) for m in maps], generator)
    else:
        difference = p_maps = None
    return {
        "fatal_holes": len(fatal), "fatal_read_safe": missed,
        "fatal_read_safe_rate": missed / len(fatal) if fatal else None,
        "fatal_by_size": {str(s): sum(f["size"] == s for f in fatal) for s in sorted({m["size"] for m in maps})},
        "position_matched": matched_test(fatal, "position_base_rate"),
        "avoided_size_matched": matched_test(fatal, "avoided_base_rate") | {"avoided_by_size": avoided},
        "maps": {
            "failed": sum(failed), "solved": len(maps) - sum(failed),
            "mean_misread_failed": sum(v for v, x in zip(values, failed) if x) / max(sum(failed), 1),
            "mean_misread_solved": sum(v for v, x in zip(values, failed) if not x) / max(len(maps) - sum(failed), 1),
            "difference": difference, "p_one_sided_within_size_and_holes": p_maps, "auc": auc(values, failed)},
    }


def main(run):
    run = Path(run)
    train, test = torch.load(run / "features-train.pt"), torch.load(run / "features-test.pt")
    rows = [r["modes"]["unmasked"]["own"] for r in map(json.loads, (run / "rows.jsonl").read_text().splitlines())]
    if len(rows) != len(test["rows"]):
        raise ValueError("rows.jsonl and features-test.pt cover different maps")
    report = {"primary_layer": PRIMARY_LAYER, "permutations": PERMUTATIONS, "layers": {}}
    statuses = None
    for position, layer in enumerate(train["layers"]):
        fatal, avoided, maps = layer_records(train, test, rows, position)
        report["layers"][str(layer + 1)] = summarise(fatal, avoided, maps, torch.Generator().manual_seed(SEED))
        if statuses is None:
            statuses = {}
            for m in maps:
                statuses.setdefault(str(m["size"]), {}).setdefault(m["status"], 0)
                statuses[str(m["size"])][m["status"]] += 1
        r = report["layers"][str(layer + 1)]
        brief = {"fatal": f"{r['fatal_read_safe']}/{r['fatal_holes']}",
                 "position_expected": r["position_matched"]["expected_read_safe"],
                 "position_p": r["position_matched"]["p_one_sided"],
                 "avoided_expected": r["avoided_size_matched"]["expected_read_safe"],
                 "avoided_p": r["avoided_size_matched"]["p_one_sided"],
                 "misread_failed": r["maps"]["mean_misread_failed"], "misread_solved": r["maps"]["mean_misread_solved"],
                 "maps_p": r["maps"]["p_one_sided_within_size_and_holes"], "auc": r["maps"]["auc"]}
        print(f"layer {layer + 1}", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in brief.items()}),
              flush=True)
    report["outcomes_by_size"] = statuses
    (run / "failures.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
