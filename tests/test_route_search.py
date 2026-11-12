import json
import math
import os
import random
from pathlib import Path

import pytest
from mathruler.grader import extract_boxed_content

from spara import route_search
from spara.detour_probe import move_probabilities
from spara.evaluation import ReferenceEvaluator
from spara.fresh_maps import draw_layout
from spara.planning_probe import walk
from spara.route_search import (RAW, FrozenLakeEnvironment, LivrMovePolicy, MisperceivedFrozenLake, search_pilot,
                                toward_goal_policy)
from spara.search import ALIVE, DEAD, DONE, STOP
from test_detour_probe import WIDE
from test_planning_probe import GRID, Processor, Tokenizer, prompt_of

STATUS = {"hole": "Fell in hole", "goal": "Reached goal", "short": "Did not reach goal"}


class StatusEvaluator:
    """The reference rules with the reference's status strings, without the released data."""
    def score(self, row, text):
        route = "".join(ch for ch in extract_boxed_content(text).upper() if ch in "UDRL")
        status = walk(row["map_desc"], route)[1]
        return {"success": status == "goal", "status": STATUS[status]}


@pytest.fixture
def data_root():
    root = Path(os.environ.get("SPARA_TEST_DATA", "data/vsp"))
    if not (root / "manifest.json").is_file():
        pytest.skip("Run spara prepare-data first")
    return root


def test_steps_agree_with_the_reference_evaluator(data_root):
    """On random maps and routes, every prefix up to the first hole or goal gets the
    reference evaluator's verdict."""
    evaluator = ReferenceEvaluator(data_root)
    env, rng = FrozenLakeEnvironment(evaluator), random.Random(0)
    expected = {ALIVE: "Did not reach goal", DEAD: "Fell in hole", DONE: "Reached goal"}
    checked = 0
    for _ in range(300):
        grid = draw_layout(rng.randint(2, 8), rng, hole_probability=0.3)
        problem = {"row": {"map_id": 0, "map_desc": grid}}
        state, actions = env.start(problem), []
        for action in rng.choices("UDLR", k=20):
            state, outcome = env.step(problem, state, action)
            actions.append(action)
            assert evaluator.score(problem["row"], env.answer(actions))["status"] == expected[outcome]
            checked += 1
            if outcome != ALIVE:
                break
    assert checked > 1000


def test_answers_use_the_released_format(data_root):
    env = FrozenLakeEnvironment(ReferenceEvaluator(data_root))
    assert env.answer("RD") == "\\boxed{RIGHT, DOWN}"
    problem = {"row": {"map_id": 0, "map_desc": GRID}}
    assert env.reward(problem, env.answer("RRDD")) == 1.0 and env.reward(problem, env.answer("RD")) == 0.0


def test_policy_reads_raw_move_probabilities_and_encodes_once(tiny_model, tiny_batch, monkeypatch):
    model = tiny_model.eval()
    model.config.livr_stage = 2
    encoded = []
    monkeypatch.setattr(route_search, "encode",
                        lambda *args, training=True: encoded.append(args[3]) or prompt_of(tiny_batch))
    policy = LivrMovePolicy(model, Processor(), "unused")
    problem = {"id": "test/1", "split": "test", "row_index": 1, "row": {"map_desc": WIDE}}
    replies = policy([(problem, ["R"], ["U", "D", "L", "R"]), (problem, [], ["U", "R"])])
    expected = move_probabilities(model, Tokenizer(), prompt_of(tiny_batch), "R", RAW)["raw"]["moves"]
    assert all(abs(replies[0][m] - math.log(expected[m])) < 1e-5 for m in "UDLR")
    assert set(replies[1]) == {"U", "R"} and len(encoded) == 1 and policy.calls == 2


def test_search_pilot_runs_end_to_end(tiny_model, tiny_batch, tmp_path, monkeypatch):
    model = tiny_model.eval()
    model.config.livr_stage = 2
    data = {"test": [{"map_id": 0, "map_desc": GRID}, {"map_id": 1, "map_desc": WIDE},
                     {"map_id": 2, "map_desc": WIDE}],
            "fresh/demo": [{"map_id": 0, "map_desc": WIDE}, {"map_id": 1, "map_desc": GRID}]}
    monkeypatch.setattr(route_search, "records", lambda root, split: data[split])
    monkeypatch.setattr(route_search, "encode", lambda *args, training=True: prompt_of(tiny_batch))
    probs = move_probabilities(model, Tokenizer(), prompt_of(tiny_batch), "R", RAW)["raw"]["moves"]
    point = {"split": "test", "row_index": 1, "kind": "forced", "taken_outcome": "hole", "route": "RRR", "k": 2,
             "p_moves": probs}
    ignored = [point | {"split": "train"}, point | {"row_index": 0}, point | {"taken_outcome": "optimal"}]
    points = tmp_path / "points.jsonl"
    points.write_text("".join(json.dumps(p) + "\n" for p in [point] + ignored))
    summary = search_pilot(model, Processor(), StatusEvaluator(), "unused", tmp_path / "run", points, "fresh/demo",
                           {}, 0, budget=8)
    assert summary["validity"]["valid"] and summary["validity"]["probe_points"] == 1
    agree, checked = summary["validity"]["simulator_agreement"]
    assert agree == checked > 0
    assert summary["primary"]["maps"] == 3 and set(summary["primary"]["by_split"]) == {"test", "fresh/demo"}
    assert summary["reading"] in ("ready", "insufficient") and summary["policy_calls"]["validity"] == 1
    rows = [json.loads(line) for line in (tmp_path / "run/searches.jsonl").read_text().splitlines()]
    assert [(r["split"], r["row_index"]) for r in rows] == [("test", 1), ("test", 2), ("fresh/demo", 0)]
    assert all(r["shortest"] == 5 and r["uniform"]["solved"] for r in rows)
    assert all(r["transitions"]["guided"] for r in rows)
    for r in rows:
        if r["guided"]["solved"]:
            assert r["guided"]["excess_length"] >= 0 and r["departures"]["moves"] >= 0
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["budget"] == 8 and manifest["maps"] == 3

    points.write_text(json.dumps(point | {"p_moves": {m: v + 0.5 for m, v in probs.items()}}) + "\n")
    summary = search_pilot(model, Processor(), StatusEvaluator(), "unused", tmp_path / "off", points, "fresh/demo",
                           {}, 0, budget=8)
    assert summary["reading"].startswith("withheld") and summary["guidance"]["reading"].startswith("withheld")


def test_a_misread_map_judges_every_move_into_a_cell_alike():
    p = {"id": "wide", "row": {"map_desc": WIDE}}
    exact = FrozenLakeEnvironment(StatusEvaluator())
    cells = [(r, c) for r in range(4) for c in range(4)]
    judged = lambda env: {(cell, a): env.step(p, cell, a) for cell in cells for a in "UDLR"}
    assert judged(MisperceivedFrozenLake(StatusEvaluator(), 0, 0, 0)) == judged(exact)
    blind = MisperceivedFrozenLake(StatusEvaluator(), 1, 0, 0)
    assert blind.step(p, (1, 1), "R") == ((1, 2), ALIVE)  # the only hole, read as safe
    assert blind.reward(p, "\\boxed{RIGHT, RIGHT, RIGHT}") == 0.0  # the verifier reads the true map
    wary = MisperceivedFrozenLake(StatusEvaluator(), 0, 1, 0)
    assert all(o in (DEAD, DONE) or child in (cell, (1, 0)) for (cell, _), (child, o) in judged(wary).items())
    assert wary.step(p, (1, 2), "R")[1] == DONE and wary.step(p, (0, 0), "R")[1] == DEAD
    misread = lambda rate: {k for k, v in judged(MisperceivedFrozenLake(StatusEvaluator(), rate, rate, 3)).items()
                            if v != judged(exact)[k]}
    assert misread(0.1) <= misread(0.4) and misread(0.4)


def test_toward_goal_stand_in_clips_and_scores_ending():
    p = {"id": "grid", "row": {"map_desc": GRID}}  # start (0, 0), goal (2, 2)
    [stuck, home] = toward_goal_policy([(p, ["U", "L"], ["D", "R", STOP]), (p, list("DDRR"), ["U", "L", STOP])])
    assert stuck == {"D": -9.0, "R": -9.0, STOP: -12.0}  # moves off the grid left the agent at the start
    assert home[STOP] == 0.0 and home["U"] == home["L"] == -3.0
