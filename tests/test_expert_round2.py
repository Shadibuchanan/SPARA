import json
import random

import pytest
import torch

from spara import detour_scaling, expert_iteration, expert_round2, training
from spara.expert_round2 import (away_moves, balanced, batching_check, bootstrap_difference, compare, latent_states,
                                 letters, masked, outcome, phases, probe_predictions, probe_score,
                                 require_round1_checkpoint, run_arm, run_search, signatures, training_rows,
                                 wall_bumps)
from spara.fresh_maps import answer_text, draw_detour_layouts, label_route
from spara.runtime import read_config
from test_detour_scaling import Evaluator, SavingProcessor
from test_expert_iteration import tiny
from test_planning_probe import GRID, Processor, prompt_of

SETTINGS = {"pool_split": "fresh/p78", "heldout_split": "fresh/h78", "round1_pool_split": "fresh/p",
            "round1_heldout_split": "fresh/h", "other_fresh": ["fresh/probe", "fresh/p", "fresh/h"],
            "maps": 4, "budget": 12, "max_depth": 12, "batch": 4, "read_precision": "tf32",
            "probe_tolerance": 0.01, "check_maps": 2,
            "seed": 0, "loading_agreement": 2, "round1_steps": 375, "extra_new": 6, "extra_old": 6,
            "variety_floor": 1, "phase_one_rows": 8, "seen_maps": 2, "epochs": 1, "learning_rate": 1e-5,
            "warmup_ratio": 0.05, "max_new_tokens": 12, "state_layers": [1, 2], "probe_layer": 1,
            "took_margin": 0.05, "bootstrap": 200}


def labelled(desc, i):
    return {"map_id": i, "map_desc": desc, "text_output": answer_text(label_route(desc)),
            "image_input": f"img/level_{len(desc)}/{i}.png"}  # IDs repeat across levels, as in the data


@pytest.fixture
def data(monkeypatch):
    small, _ = draw_detour_layouts(10, random.Random(0), set(), [4], [1])
    large, _ = draw_detour_layouts(8, random.Random(1), set(), [5, 6], [1, 1])
    sets = {"fresh/h": [labelled(d, i) for i, d in enumerate(small[:2])],
            "fresh/p": [labelled(d, i) for i, d in enumerate(small[2:6])],
            "fresh/probe": [labelled(small[6], 0)],
            "fresh/h78": [labelled(d, i) for i, d in enumerate(large[:4])],
            "fresh/p78": [labelled(d, i) for i, d in enumerate(large[4:8])],
            "train": [labelled(GRID, 0), labelled(small[7], 1)],
            "test": [labelled(GRID, 0), labelled(small[8], 1)]}
    for module in (expert_round2, detour_scaling, expert_iteration):
        monkeypatch.setattr(module, "records", lambda root, split: sets[split])
    return sets


@pytest.fixture
def prompts(tiny_batch, monkeypatch):
    prompt = lambda *args, **kwargs: prompt_of(tiny_batch)
    for module in (expert_round2, detour_scaling, expert_iteration):
        monkeypatch.setattr(module, "encode", prompt)
    monkeypatch.setattr(training, "encode", lambda *args, **kwargs: {k: v.clone() for k, v in tiny_batch.items()})


def test_require_round1_checkpoint_checks_the_recipe(tmp_path):
    config = read_config("configs/livr.json")
    progress = {"stage": 2, "epoch": 1, "next_row": 0, "optimizer_steps": 375,
                "config": config | {"epochs_per_stage": 1, "learning_rate": 1e-5, "warmup_ratio": 0.05}}
    (tmp_path / "complete.json").write_text(json.dumps({"progress": progress}))
    require_round1_checkpoint(tmp_path, config, SETTINGS)
    for change in ({"optimizer_steps": 2500}, {"stage": 1},
                   {"config": progress["config"] | {"learning_rate": 1e-4}}):
        (tmp_path / "complete.json").write_text(json.dumps({"progress": progress | change}))
        with pytest.raises(ValueError):
            require_round1_checkpoint(tmp_path, config, SETTINGS)


def test_masked_restores_the_stage_after_an_error(tiny_model):
    model = tiny(tiny_model)
    with pytest.raises(RuntimeError), masked(model):
        assert model.config.livr_stage == 1
        raise RuntimeError
    assert model.config.livr_stage == 2


def test_phases_shuffle_then_split():
    data = [("train", {"i": i}) for i in range(30)]
    one, two = phases(data, SETTINGS | {"phase_one_rows": 16}, 8)
    assert len(one) == 16 and len(two) == 14
    assert sorted(r["i"] for _, r in one + two) == list(range(30))
    assert [r["i"] for _, r in one] != list(range(16))  # shuffled before the split
    with pytest.raises(ValueError, match="whole number"):
        phases(data, SETTINGS | {"phase_one_rows": 12}, 8)


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def route_rows(rows, solved):
    return [{"row_index": i, "map_id": r["map_id"], "solved": i in solved,
             "answer": answer_text(label_route(r["map_desc"])) if i in solved else None} for i, r in enumerate(rows)]


def test_training_rows_draw_from_both_rounds(data, tmp_path):
    round1 = write_jsonl(tmp_path / "round1.jsonl", route_rows(data["fresh/p"], {0, 1}))
    search = tmp_path / "search"
    search.mkdir()
    write_jsonl(search / "routes.jsonl", route_rows(data["fresh/p78"], {1, 3}))
    write_jsonl(search / "frontier.jsonl", [r | {"row_index": i} for i, r in
                                            enumerate(route_rows(data["fresh/p"], {2}))][2:])
    rows, counts = training_rows("unused", SETTINGS, round1, search)
    assert counts == {"new_maps": 2, "old_maps": 3, "frontier_maps": 1, "rows": 2 + 12}
    assert rows[:2] == [("train", r) for r in data["train"]]
    new = [r["map_id"] for split, r in rows[2:8]]
    assert new == [1, 3, 1, 3, 1, 3] and {split for split, _ in rows[2:8]} == {"fresh/p78"}
    assert [r["map_id"] for _, r in rows[8:]] == [0, 1, 2, 0, 1, 2]
    assert all(r["text_output"] == answer_text(label_route(r["map_desc"])) for _, r in rows[2:])
    with pytest.raises(ValueError, match="variety floor"):
        training_rows("unused", SETTINGS | {"variety_floor": 3}, round1, search)
    write_jsonl(search / "frontier.jsonl", route_rows(data["fresh/p"], {0}))
    with pytest.raises(ValueError, match="already solved"):
        training_rows("unused", SETTINGS, round1, search)


def test_away_moves_and_signatures():
    grid = [[1, 0, 0], [0, -1, 0], [0, 0, 2]]
    assert away_moves(grid, "RRDD") == 0 and away_moves(grid, "URRDD") == 0  # a wall bump is not away
    assert away_moves(grid, "RL") == 1 and away_moves(grid, "RDLL") == 0  # counting stops at the hole
    assert letters("DOWN, RIGHT") == "DR" and letters(None) == ""
    descs = {0: grid, 1: grid, 2: grid, 3: grid}
    marks = signatures(descs, {0: "RRDD", 1: "RD", 2: "RR", 3: "URRDD"})
    assert marks == {"maps": 4, "solved": 2, "failed": 2, "hole": 1, "short_or_past": 1, "failed_with_away_move": 0}
    assert signatures({0: grid}, {0: "DDRUR"})["failed_with_away_move"] == 1


def test_outcomes_and_wall_bumps():
    grid = [[1, 0, 0], [-1, 0, 0], [0, 0, 2]]
    assert outcome(grid, "RRDD") == "shortest" and outcome(grid, "RR") == "stops short"
    assert outcome(grid, "D") == "hole" and outcome(grid, "URRDD") == "longer"
    assert outcome(grid, "RRDDL") == "passes the goal" and outcome(grid, "") == "stops short"
    assert wall_bumps(grid, "URRDD") == 1 and wall_bumps(grid, "RRDD") == 0


def test_balanced_accuracy_and_its_bootstrap():
    truth = torch.tensor([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=torch.bool)
    exact = truth.clone()
    wrong = truth.clone()
    wrong[0, 0] = False
    ones = torch.ones(1, 4, dtype=torch.double)
    assert float(balanced(exact, truth, ones)[0]) == 1.0
    assert abs(float(balanced(wrong, truth, ones)[0]) - (0.75 + 1.0) / 2) < 1e-9
    first = {4: {"predicted": exact, "truth": truth, "rows": [0, 1, 2, 3]}}
    second = {4: {"predicted": wrong, "truth": truth, "rows": [0, 1, 2, 3]}}
    interval = bootstrap_difference(first, second, 500, 0)
    assert 0 <= interval["low"] <= 0.125 <= interval["high"]
    with pytest.raises(ValueError, match="different maps"):
        bootstrap_difference(first, {4: second[4] | {"rows": [0, 1, 2, 4]}}, 10, 0)


def synthetic_states(noise, seed=0):
    """Latent states that encode each map's holes linearly, plus noise."""
    rng = random.Random(seed)
    generator = torch.Generator().manual_seed(seed)
    items, states = [], []
    projection = torch.randn(36, 2 * 4 * 16, generator=generator)
    for population, count in (("pool78", 60), ("heldout78", 30)):
        layouts, _ = draw_detour_layouts(count, rng, set(), [6], [1])
        for i, desc in enumerate(layouts):
            holes = torch.tensor([float(v == -1) for row in desc for v in row])
            state = holes @ projection + noise * torch.randn(2 * 4 * 16, generator=generator)
            states.append(state.view(2, 4, 16))
            items.append({"population": population, "split": "s", "row_index": i, "map_desc": desc})
    return {"layers": [1, 2], "states": torch.stack(states).half(), "items": items}


def test_the_hole_probe_reads_what_the_states_hold():
    clean = probe_score(probe_predictions(synthetic_states(0.0), 1))
    noisy = probe_score(probe_predictions(synthetic_states(30.0), 1))
    assert clean["hole_balanced_accuracy"] > 0.9 > noisy["hole_balanced_accuracy"] + 0.1
    assert set(clean["by_size"]) == {6}


def test_latent_states_read_the_four_latents(tiny_model, tiny_batch, prompts, tmp_path):
    model = tiny(tiny_model)
    items = [("pool78", "s", 0, {"map_desc": GRID}), ("heldout78", "s", 1, {"map_desc": GRID})]
    latent_states(model, Processor(), "unused", items, [1, 3], tmp_path / "states.pt")
    saved = torch.load(tmp_path / "states.pt")
    assert saved["states"].shape == (2, 2, 4, 32) and saved["layers"] == [1, 3]
    with torch.no_grad():
        hidden = model(**prompt_of(tiny_batch), output_hidden_states=True, use_cache=False).hidden_states
    assert torch.allclose(saved["states"][0, 0].float(), hidden[1][0, 5:9].half().float())
    with masked(model), pytest.raises(ValueError, match="unmasked"):
        latent_states(model, Processor(), "unused", items, [1], tmp_path / "other.pt")


def test_batching_check_runs_both_precisions(tiny_model, prompts, monkeypatch, tmp_path):
    rows = [{"map_desc": [[0] * n for _ in range(n)]} for n in (3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8)]
    monkeypatch.setattr(expert_round2, "records", lambda root, split: rows)
    result = batching_check(tiny(tiny_model), Processor(), "unused", "fresh/any", tmp_path)
    assert len(result["rows"]) == 16 and len(result["equal_rows"]) == 4
    assert result["fp32"]["max_difference"] < 1e-5 and result["fp32"]["equal_length_max_difference"] < 1e-5
    assert "bf16" in result and (tmp_path / "batching-check.json").exists()


def round1_run(tmp_path, data, success=True):
    round1 = write_jsonl(tmp_path / "round1-routes.jsonl", route_rows(data["fresh/p"], {0, 1}))
    greedy = [{"population": "heldout", "row_index": i, "success": success} for i in range(2)]
    greedy += [{"population": "test_detour", "row_index": 1, "success": success}]
    return round1, write_jsonl(tmp_path / "round1-greedy.jsonl", greedy)


def test_run_search_evaluates_e_then_searches_frontier_and_pool(data, tiny_model, prompts, tmp_path, monkeypatch):
    model = tiny(tiny_model)
    round1, _ = round1_run(tmp_path, data)
    # Round 1's recorded outcomes are E's own, so the loading check passes.
    first = detour_scaling.greedy_eval
    recorded = []

    def greedy(*args, **kwargs):
        rows = first(*args, **kwargs)
        recorded.extend(rows)
        return rows
    monkeypatch.setattr(expert_round2, "greedy_eval", greedy)
    expert_round2.evaluate(model, Processor(), Evaluator(), "unused", tmp_path, SETTINGS)
    old = write_jsonl(tmp_path / "e-greedy.jsonl", [r for r in recorded if r["population"] in ("heldout", "test_detour")])
    summary = run_search(model, Processor(), Evaluator(), "unused", tmp_path / "run", SETTINGS, {}, round1, old)
    checks = summary["checks"]
    assert checks["valid"] and checks["batch"] == 4 and checks["loading"]["same_outcome"] == 3
    assert checks["step_zero_max_difference"] < 1e-5 and checks["single_max_difference"] < 1e-5
    frontier = [json.loads(line) for line in (tmp_path / "run/frontier.jsonl").read_text().splitlines()]
    assert [r["row_index"] for r in frontier] == [2, 3] and summary["pool"]["maps"] == 4
    assert summary["evaluation"]["greedy"]["heldout78"][1] == 4 and summary["evaluation"]["masked"][1] == 4
    for name in ("greedy.jsonl", "masked.jsonl", "states.pt", "routes.jsonl", "summary-before-pool.json"):
        assert (tmp_path / "run" / name).exists()
    states = torch.load(tmp_path / "run/states.pt")
    assert [i["population"] for i in states["items"]] == ["pool78"] * 4 + ["heldout78"] * 4
    expert_round2.livr_greedy(model, Processor(), Evaluator(), "unused", tmp_path / "run", SETTINGS)
    assert len((tmp_path / "run/livr-greedy.jsonl").read_text().splitlines()) == 4


def test_run_search_stops_when_e_does_not_reproduce_round_1(data, tiny_model, prompts, tmp_path, monkeypatch):
    round1, old = round1_run(tmp_path, data)
    monkeypatch.setattr(expert_round2, "greedy_eval", lambda model, processor, evaluator, root, items, log, cap: [
        {"population": p, "row_index": i, "success": False} for p, _, i, _ in items])
    with pytest.raises(ValueError, match="validity check failed"):
        run_search(tiny(tiny_model), Processor(), Evaluator(), "unused", tmp_path / "run", SETTINGS, {}, round1, old)
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert not summary["checks"]["valid"] and summary["checks"]["loading"]["same_outcome"] == 0
    assert not (tmp_path / "run/routes.jsonl").exists()


@pytest.mark.parametrize("arm", ["e2", "eb2"])
def test_run_arm_trains_two_phases_and_evaluates(arm, data, tiny_model, prompts, tmp_path, monkeypatch):
    round1 = write_jsonl(tmp_path / "round1.jsonl", route_rows(data["fresh/p"], {0, 1}))
    search = tmp_path / "search"
    search.mkdir()
    write_jsonl(search / "routes.jsonl", route_rows(data["fresh/p78"], {1, 3}))
    write_jsonl(search / "frontier.jsonl", [])
    stages, benchmarks = [], []
    train = training.train_stage

    def tracked(model, processor, root, output, config, stage, **kwargs):
        stages.append((output.name, stage, kwargs.get("save_final", True), len(kwargs["data"])))
        return train(model, processor, root, output, config, stage, **kwargs)
    monkeypatch.setattr(expert_round2, "train_stage", tracked)

    def benchmark(model, processor, config, root, output, manifest):
        assert model.config.livr_stage == 2
        output.mkdir()
        benchmarks.append(manifest["arm"])
    monkeypatch.setattr(expert_round2, "run_benchmark", benchmark)
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False}
    summary = run_arm(tiny(tiny_model), SavingProcessor(), Evaluator(), config, "unused", tmp_path / arm,
                      SETTINGS, arm, round1, search, {})
    first_stage = 1 if arm == "eb2" else 2
    assert stages == [("phase1", first_stage, False, 8), ("phase2", 2, True, 6)]
    assert not (tmp_path / arm / "phase1/final").exists() and (tmp_path / arm / "phase2/final").exists()
    steps = [json.loads(line) for line in (tmp_path / arm / "phase1/steps.jsonl").read_text().splitlines()]
    assert {s["stage"] for s in steps} == {first_stage} and len(steps) == 1
    assert summary["training"]["phase1"]["stage"] == first_stage and summary["training"]["phase2"]["optimizer_steps"] == 1
    assert benchmarks == [arm] and summary["counts"]["rows"] == 14
    for name in ("greedy.jsonl", "masked.jsonl", "states.pt"):
        assert (tmp_path / arm / name).exists()


def write_run(path, data, held, masked_success, old=(1, 1, 1), seen=(1, 0), states=None, benchmark=None):
    path.mkdir()
    rows = [{"population": "heldout", "split": "fresh/h", "row_index": i, "grid_size": 4, "route": "",
             "success": old[i]} for i in range(2)]
    rows += [{"population": "test_detour", "split": "test", "row_index": 1, "grid_size": 4, "route": "",
              "success": old[2]}]
    rows += [{"population": "heldout78", "split": "fresh/h78", "row_index": i,
              "grid_size": len(data["fresh/h78"][i]["map_desc"]), "route": "R", "success": s}
             for i, s in enumerate(held)]
    rows += [{"population": "seen78", "split": "fresh/p78", "row_index": i, "grid_size": 5, "route": "",
              "success": s} for i, s in enumerate(seen)]
    write_jsonl(path / "greedy.jsonl", rows)
    write_jsonl(path / "masked.jsonl", [{"population": "heldout78", "row_index": i, "success": s}
                                        for i, s in enumerate(masked_success)])
    torch.save(states, path / "states.pt")
    (path / "summary.json").write_text(json.dumps({"checks": {"valid": True}, "seconds": 2.0,
                                                   "training": {"phase1": {}, "phase2": {}}}))
    if benchmark is not None:
        (path / "evaluation").mkdir()
        write_jsonl(path / "evaluation/responses.jsonl",
                    [{"row_index": i, "success": s} for i, s in enumerate(benchmark)])


def test_compare_applies_the_fixed_rules(data, tmp_path, monkeypatch):
    monkeypatch.setattr(expert_round2, "ALPHA", 1.01)  # four maps cannot reach 0.05: direction only
    clean, noisy = synthetic_states(0.0), synthetic_states(30.0)
    search = tmp_path / "search"
    write_run(search, data, [0, 0, 0, 0], [0, 0, 0, 0], states=clean)
    (search / "summary.json").write_text(json.dumps({"checks": {"valid": True}, "seconds": 3.0,
                                                     "frontier": {"solved": 1}, "pool": {"solved": 2}}))
    write_jsonl(search / "routes.jsonl", [{"row_index": i, "solved": i < 1, "route": "DR" if i < 1 else None,
                                           "equals_label": False, "longer_than_label": False} for i in range(4)])
    write_jsonl(search / "livr-greedy.jsonl", [{"row_index": i, "success": False, "route": "R"} for i in range(4)])
    write_run(tmp_path / "e2", data, [1, 0, 0, 0], [0, 0, 0, 0], states=noisy, benchmark=[1, 1, 0])
    write_run(tmp_path / "eb2", data, [1, 1, 1, 0], [1, 1, 0, 0], states=clean, benchmark=[1, 0, 0])
    livr = write_jsonl(tmp_path / "livr.jsonl", [{"row_index": i, "success": True} for i in range(3)])
    e = write_jsonl(tmp_path / "e.jsonl", [{"row_index": i, "success": i == 0} for i in range(3)])
    arms = {"e2": tmp_path / "e2", "eb2": tmp_path / "eb2"}
    result = compare("unused", search, arms, livr, e, SETTINGS)
    assert result["validity"]["valid"] and result["manipulation"]["took"]
    assert result["manipulation"]["difference"] > 0.1 and result["manipulation"]["interval_95"]["low"] > 0
    assert result["primary"]["only_first"] == 2 and result["primary"]["reading"] == "the bottleneck adds to expert iteration"
    assert result["secondary"]["reading"] == "round 2 helps on larger maps"
    assert result["descriptive"]["masked"]["eb2"] == {"unmasked": 3, "masked": 2}
    assert result["descriptive"]["held_out_success"] == {"e": 0, "e2": 1, "eb2": 3, "l": 0}
    assert result["descriptive"]["seen"]["maps"] == {"trained": 1, "untrained": 1}
    assert result["descriptive"]["benchmark"]["eb2"]["vs_livr"]["all"]["only_second"] == 2
    assert sum(result["descriptive"]["failures"]["eb2"].values()) == 4

    # The gate fails when EB2's latents hold no more than E2's.
    torch.save(noisy, tmp_path / "eb2/states.pt")
    failed = compare("unused", search, arms, livr, e, SETTINGS)
    assert not failed["manipulation"]["took"]
    assert failed["primary"]["reading"] == "the bottleneck did not take; no reading on perception"
    (search / "summary.json").write_text(json.dumps({"checks": {"valid": False}, "seconds": 3.0,
                                                     "frontier": {}, "pool": {}}))
    assert compare("unused", search, arms, livr, e, SETTINGS)["primary"]["reading"].startswith("withheld")


def test_run_search_falls_back_to_round_1_reads(data, tiny_model, prompts, tmp_path, monkeypatch):
    model = tiny(tiny_model)
    round1, old = round1_run(tmp_path, data)
    monkeypatch.setattr(expert_round2, "evaluate", lambda *args, **kwargs: {"loading": {"valid": True}})
    real = expert_round2.single_read
    precisions = []

    def single(policy, problem, prefix):
        precisions.append(policy.precision)
        probs = real(policy, problem, prefix)
        return {k: v + 0.5 for k, v in probs.items()} if policy.precision == "tf32" else probs
    monkeypatch.setattr(expert_round2, "single_read", single)
    summary = run_search(model, Processor(), Evaluator(), "unused", tmp_path / "run", SETTINGS, {}, round1, old)
    checks = summary["checks"]
    assert checks["batched_failed"]["step_zero_max_difference"] == pytest.approx(0.5)
    assert checks["batch"] == 1 and checks["read_precision"] == "bf16" and checks["valid"]
    assert set(precisions) == {"tf32", "bf16"}
