import json
import math
import random

import pytest
import torch

from spara import jepa, route_search
from spara.detour_probe import move_probabilities
from spara.fresh_maps import draw_layouts, label_route
from spara.jepa import (ACTIONS, OUTCOMES, Decisions, Jepa, JepaFrozenLake, LivrPolicy, collect, decide,
                        imagination, pairs, rates, route_cells, route_states, run_jepa, train, walks)
from spara.planning_probe import walk
from spara.route_search import RAW, FrozenLakeEnvironment
from spara.search import ALIVE, DEAD, DONE, STOP, guided_search
from test_detour_probe import WIDE
from test_planning_probe import GRID, Processor, Tokenizer, prompt_of
from test_route_search import StatusEvaluator

SETTINGS = {"seed": 0, "epochs": 3, "patience": 2, "batch": 16, "learning_rate": 1e-3, "weight_decay": 0.0}


def test_walks_start_with_the_label_and_stay_on_the_grid():
    rng = random.Random(0)
    grid = WIDE
    routes = walks(grid, rng, count=6)
    assert routes[0] == label_route(grid) and len(routes) == 6
    for route in routes[1:]:
        cells = route_cells(grid, route)
        assert all(0 <= r < 4 and 0 <= c < 4 for r, c in cells) and len(route) <= 8
        assert all(cells[i] != cells[i + 1] for i in range(len(route)))  # no move off the grid
        status = walk(grid, route)[1]
        assert status in ("hole", "goal") or len(route) == 8
    assert walks(grid, random.Random(0), count=6) == routes


def tiny(model):
    model = model.eval()
    model.config.livr_stage = 2
    return model


def test_batched_states_match_single_passes_and_the_probe(tiny_model, tiny_batch):
    model, prompt = tiny(tiny_model), prompt_of(tiny_batch)
    routes = ["RRD", "DL", ""]
    states = route_states(model, Tokenizer(), prompt, routes, [0, 2])
    assert [s.shape for s in states] == [(4, 2, 32), (3, 2, 32), (1, 2, 32)]
    for route, tensor in zip(routes, states):
        for k in range(len(route) + 1):
            probs, z = decide(model, Tokenizer(), prompt, route[:k], 2)
            assert torch.allclose(z, tensor[k, 1], atol=1e-4)
    expected = move_probabilities(model, Tokenizer(), prompt, "R", RAW)["raw"]
    probs, _ = decide(model, Tokenizer(), prompt, "R", 0)
    assert all(abs(probs[m] - expected["moves"][m]) < 1e-6 for m in "UDLR") and abs(probs["end"] - expected["end"]) < 1e-6


def test_collect_labels_every_move_on_the_grid(tiny_model, tiny_batch, monkeypatch):
    monkeypatch.setattr(jepa, "encode", lambda *args, training=True: prompt_of(tiny_batch))
    x, items = collect(tiny(tiny_model), Processor(), "unused", "test", [(0, {"map_desc": WIDE})], [1], seed=0)
    assert x.shape == (len(items), 1, 32) and x.dtype == torch.float16
    start = items[0]  # WIDE: start (1, 0); up and down safe, right safe, left off the grid
    assert start["k"] == 0 and start["cell"] == [1, 0] and start["outcomes"] == {"U": ALIVE, "D": ALIVE, "R": ALIVE}
    assert all(it["outcomes"] == {} for it in items if not it["alive"])
    judged, taken = pairs(items)
    assert len(taken) == sum(len(it["route"]) for it in items if it["k"] == 0)
    for i, a, j in taken.tolist():
        assert items[j]["k"] == items[i]["k"] + 1 and ACTIONS[a] == items[i]["next"]
    assert (judged[:, 2] == OUTCOMES.index(DEAD)).any()


def separable(n, d, seed):
    """States whose first coordinates spell out what each move leads to."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    labels = torch.randint(0, 3, (n, 4), generator=g)
    x[:, :12] = torch.nn.functional.one_hot(labels, 3).reshape(n, 12).float() * 4
    judged = torch.tensor([(i, a, int(labels[i, a])) for i in range(n) for a in range(4)])
    taken = torch.tensor([(i, i % 4, (i + 1) % n) for i in range(n)])
    return x, judged, taken, labels[taken[:, 2], 0]


def test_training_learns_a_separable_judge_and_stops_early():
    x, judged, taken, nxt = separable(400, 32, 0)
    vx, vj, vt, vn = separable(100, 32, 1)
    settings = SETTINGS | {"epochs": 30, "patience": 3}
    for kind in ("jepa", "probe"):
        model, history = train(kind, x, judged, taken, nxt, (vx, vj, vt, vn), settings, "cpu", 0)
        predicted = model.judge(vx[vj[:, 0]], vj[:, 1]).argmax(-1)
        assert rates(predicted, vj[:, 2])["accuracy"] > 0.9 and len(history) <= 30
        assert ("prediction" in history[0]) == (kind == "jepa")
    oracle, _ = train("oracle", x, judged, taken, nxt, (vx, vj, vt, vn), settings, "cpu", 0)
    assert oracle(vx[:1]).shape == (1, 3)


def test_rates_count_misses_and_false_alarms():
    truth = torch.tensor([1, 1, 1, 0, 0, 2, 2, 0])
    predicted = torch.tensor([1, 0, 2, 1, 0, 2, 0, 0])
    r = rates(predicted, truth)
    assert r["dead"] == 3 and abs(r["miss_rate"] - 2 / 3) < 1e-9 and abs(r["false_alarm_rate"] - 1 / 5) < 1e-9
    assert r["done_missed"] == 1 and r["done_false"] == 1


class Truthful:
    """Decisions and a JEPA stand-in that encode the map and cell, so the judge is exact."""
    def __init__(self, grids):
        self.grids, self.forwards = grids, 0

    def get(self, problem, prefix):
        self.forwards += 1
        grid = problem["row"]["map_desc"]
        r, c = route_cells(grid, prefix)[-1]
        z = torch.tensor([float(problem["index"]), float(r), float(c)])
        return {m: 0.25 for m in "UDLR"} | {"end": 0.1}, z

    def judge(self, z, a):
        z = z[0]
        grid = self.grids[int(z[0])]
        move = ACTIONS[int(a[0])]
        cell = jepa.in_grid(grid, (int(z[1]), int(z[2])), move)
        return torch.nn.functional.one_hot(torch.tensor([OUTCOMES.index(jepa.cell_outcome(grid, cell))]), 3).float()


def test_a_perfect_learned_judge_searches_like_the_simulator():
    rng = random.Random(1)
    grids = [WIDE, GRID] + draw_layouts(5, 6, rng, set())
    problems = [{"id": f"m{i}", "index": i, "row": {"map_desc": g}} for i, g in enumerate(grids)]
    truthful = Truthful(grids)
    policy = LivrPolicy(truthful)
    env = JepaFrozenLake(StatusEvaluator(), truthful, truthful, "cpu")
    exact = guided_search(FrozenLakeEnvironment(StatusEvaluator()), policy, problems, 64, 30)
    learned = guided_search(env, policy, problems, 64, 30)
    assert [(r["actions"], r["policy_calls"]) for r in exact] == [(r["actions"], r["policy_calls"]) for r in learned]
    assert all(r["solved"] for r in learned)
    rates_in_search = jepa.in_search_rates(env, problems, learned)
    assert rates_in_search["miss_rate"] == 0 and rates_in_search["false_alarm_rate"] == 0
    assert rates_in_search["done"] > 0 and rates_in_search["done_missed"] == rates_in_search["done_false"] == 0
    [reply] = policy([(problems[0], ["R"], ["U", STOP])])
    assert abs(reply[STOP] - math.log(0.1)) < 1e-9


def test_imagination_compares_rolled_and_true_states():
    model = Jepa(8, width=16, action_dim=4).eval()
    items = [{"split": "s", "row_index": 0, "walk": 0, "k": k, "alive": True, "next": "R",
              "outcomes": {"R": ALIVE}} for k in range(4)]
    items[-1] |= {"next": None, "outcomes": {}}
    out = imagination(model, torch.randn(4, 8), items, "cpu", horizons=(1, 2))
    assert out["1"]["imagined"]["pairs"] == 2 and out["2"]["imagined"]["pairs"] == 1


def test_run_jepa_end_to_end(tiny_model, tiny_batch, tmp_path, monkeypatch):
    model = tiny(tiny_model)
    rng = random.Random(4)
    layouts = draw_layouts(4, 16, rng, set())
    detours = [WIDE]
    data = {"fresh/jt": [{"map_id": i, "map_desc": g} for i, g in enumerate(layouts[:10])],
            "test": [{"map_id": 0, "map_desc": GRID}, {"map_id": 1, "map_desc": WIDE}],
            "fresh/pr": [{"map_id": 0, "map_desc": layouts[10]}],
            "fresh/h": [{"map_id": 0, "map_desc": layouts[11]}, {"map_id": 1, "map_desc": layouts[12]}],
            "train": [{"map_id": 0, "map_desc": layouts[13]}]}
    for module in (jepa, route_search):
        monkeypatch.setattr(module, "records", lambda root, split: data[split])
    monkeypatch.setattr(route_search, "is_detour", lambda grid: grid in detours)
    monkeypatch.setattr(jepa, "encode", lambda *args, training=True: prompt_of(tiny_batch))
    probs, _ = decide(model, Tokenizer(), prompt_of(tiny_batch), "R", 1)
    point = {"split": "test", "row_index": 1, "kind": "forced", "taken_outcome": "hole", "route": "RRR", "k": 2,
             "p_moves": {m: probs[m] for m in "UDLR"}}
    points = tmp_path / "points.jsonl"
    points.write_text(json.dumps(point) + "\n")
    settings = SETTINGS | {"train_split": "fresh/jt", "heldout_split": "fresh/h", "fresh_split": "fresh/pr",
                           "layers": [0, 1], "validation_every": 5, "budget": 16, "max_depth": 16,
                           "veto": 10.0, "reopen": True, "probe_tolerance": 0.01, "max_expansions": 50_000}
    summary = run_jepa(model, Processor(), StatusEvaluator(), "unused", tmp_path / "run", settings, {}, points)
    assert summary["checks"]["disjoint"]["valid"] and summary["checks"]["probe_points"] == 1
    assert summary["checks"]["state_alignment_min_cosine"] > 0.999
    assert summary["layer"] in (0, 1) and set(summary["layer_selection"]) == {"0", "1"}
    assert summary["primary"]["maps"] == 1 and summary["search"]["exact"]["solved"] == 1
    assert summary["data"]["fit_maps"] == 8 and summary["data"]["validation_maps"] == 2
    assert set(summary["judge_accuracy"]["jepa"]["by_population"]) == {"search", "heldout"}
    for name in ("summary.json", "summary-before-search.json", "manifest.json", "jepa.pt", "states.pt",
                 "searches.jsonl", "items-eval.jsonl", "search-exact.json", "search-jepa.json",
                 "searches-verdict_only.jsonl"):
        assert (tmp_path / "run" / name).exists()
    rows = [json.loads(line) for line in (tmp_path / "run/searches.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["test/1"] and set(rows[0]) >= {"exact", "jepa", "verdict_only"}
