import json
import random

import pytest
import torch

from spara import detour_scaling, expert_iteration, expert_round2, training
from spara.expert_round2 import (away_moves, balanced, batching_check, bootstrap_difference, ceiling, compare,
                                 glimpse_check, glimpse_reading, hole_check, hole_walks, latent_states, letters, masked,
                                 oracle_read, outcome, phases, probe_predictions, probe_score, reading_check,
                                 require_base_checkpoint, round3, run_arm, run_search, signatures, text_check,
                                 training_rows, wall_bumps, with_map_text)
from spara.fresh_maps import answer_text, draw_detour_layouts, label_route
from spara.runtime import read_config
from test_detour_scaling import Evaluator, SavingProcessor
from test_expert_iteration import tiny
from test_planning_probe import GRID, Processor, prompt_of

SETTINGS = {"pool_split": "fresh/p78", "heldout_split": "fresh/h78", "round1_pool_split": "fresh/p",
            "round1_heldout_split": "fresh/h", "frontier_split": "fresh/p", "other_fresh": ["fresh/probe", "fresh/p", "fresh/h"],
            "maps": 4, "budget": 12, "max_depth": 12, "batch": 4, "read_precision": "tf32", "probe_tolerance": 0.01, "check_maps": 2,
            "seed": 0, "loading_agreement": 2, "base_steps": 375, "extra_new": 6, "extra_old": 6,
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
            "fresh/p78b": [labelled(d, i) for i, d in enumerate(draw_detour_layouts(3, random.Random(2), set(),
                                                                                    [5, 6], [1, 1])[0])],
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


def test_require_base_checkpoint_checks_the_recipe(tmp_path):
    config = read_config("configs/livr.json")
    progress = {"stage": 2, "epoch": 1, "next_row": 0, "optimizer_steps": 375,
                "config": config | {"epochs_per_stage": 1, "learning_rate": 1e-5, "warmup_ratio": 0.05}}
    (tmp_path / "complete.json").write_text(json.dumps({"progress": progress}))
    require_base_checkpoint(tmp_path, config, SETTINGS)
    for change in ({"optimizer_steps": 2500}, {"stage": 1},
                   {"config": progress["config"] | {"learning_rate": 1e-4}}):
        (tmp_path / "complete.json").write_text(json.dumps({"progress": progress | change}))
        with pytest.raises(ValueError):
            require_base_checkpoint(tmp_path, config, SETTINGS)


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


def search_run(path, settings, routes, frontier=()):
    path.mkdir()
    (path / "manifest.json").write_text(json.dumps({"settings": settings}))
    write_jsonl(path / "routes.jsonl", routes)
    write_jsonl(path / "frontier.jsonl", list(frontier))
    return path


def test_training_rows_draw_from_both_rounds(data, tmp_path):
    round1 = write_jsonl(tmp_path / "round1.jsonl", route_rows(data["fresh/p"], {0, 1}))
    old = {k: v for k, v in SETTINGS.items() if k != "frontier_split"}  # round 2's search predates the key
    search = search_run(tmp_path / "search", old, route_rows(data["fresh/p78"], {1, 3}),
                        route_rows(data["fresh/p"], {2})[2:])
    rows, counts = training_rows("unused", SETTINGS, round1, [search])
    assert counts["new_maps"] == 2 and counts["old_maps"] == 3 and counts["rows"] == 2 + 12
    assert list(counts["solved"].values()) == [2, 1, 2]
    assert rows[:2] == [("train", r) for r in data["train"]]
    new = [r["map_id"] for split, r in rows[2:8]]
    assert new == [1, 3, 1, 3, 1, 3] and {split for split, _ in rows[2:8]} == {"fresh/p78"}
    assert [r["map_id"] for _, r in rows[8:]] == [0, 1, 2, 0, 1, 2]
    assert all(r["text_output"] == answer_text(label_route(r["map_desc"])) for _, r in rows[2:])
    with pytest.raises(ValueError, match="variety floor"):
        training_rows("unused", SETTINGS | {"variety_floor": 3}, round1, [search])
    write_jsonl(search / "frontier.jsonl", route_rows(data["fresh/p"], {0}))
    with pytest.raises(ValueError, match="already solved"):
        training_rows("unused", SETTINGS, round1, [search])


def test_training_rows_chain_round_3_onto_rounds_1_and_2(data, tmp_path):
    round1 = write_jsonl(tmp_path / "round1.jsonl", route_rows(data["fresh/p"], {0, 1}))
    second = search_run(tmp_path / "second", SETTINGS, route_rows(data["fresh/p78"], {1, 3}),
                        route_rows(data["fresh/p"], {2})[2:])
    third_settings = SETTINGS | {"pool_split": "fresh/p78b", "frontier_split": "fresh/p78"}
    unsolved = [r for r in route_rows(data["fresh/p78"], {0}) if r["row_index"] in (0, 2)]
    third = search_run(tmp_path / "third", third_settings, route_rows(data["fresh/p78b"], {2}), unsolved)
    rows, counts = training_rows("unused", third_settings | {"extra_new": 8}, round1, [second, third])
    assert counts["new_maps"] == 4 and counts["old_maps"] == 3
    new = [(split, r["map_id"]) for split, r in rows[2:10]]
    assert new[:4] == [("fresh/p78", 0), ("fresh/p78", 1), ("fresh/p78", 3), ("fresh/p78b", 2)] and new[4:] == new[:4]
    assert [r["map_id"] for _, r in rows[10:]] == [0, 1, 2, 0, 1, 2]
    with pytest.raises(ValueError, match="variety floor"):
        training_rows("unused", third_settings | {"variety_floor": 2}, round1, [second, third])
    write_jsonl(third / "frontier.jsonl", route_rows(data["fresh/p78"], {1}))
    with pytest.raises(ValueError, match="already solved"):
        training_rows("unused", third_settings, round1, [second, third])


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


def test_hole_walks_read_the_fatal_hole_and_the_holes_passed_by():
    # Map 0 falls into (1, 0) after starting next to it; map 1 passes (0, 1) and reaches the goal.
    desc = {0: [[1, 0, 0], [-1, 0, 0], [0, 0, 2]], 1: [[1, -1, 0], [0, 0, 0], [0, 0, 2]]}
    routes = {0: "D", 1: "DRRD"}
    predicted = torch.zeros(2, 9, dtype=torch.bool)
    predicted[1, 1] = True  # the probe reads map 1's passed hole, and misses map 0's fatal one
    result = hole_walks({3: {"rows": [0, 1], "desc": [desc[0], desc[1]], "predicted": predicted}}, routes)
    assert result["maps"] == 2 and result["fatal"] == {"read_as_hole": 0, "holes": 1, "share": 0.0}
    assert result["avoided"] == {"read_as_hole": 1, "holes": 1, "share": 1.0}


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


@pytest.mark.parametrize("arm", ["e2", "eb2", "s3"])
def test_run_arm_trains_two_phases_and_evaluates(arm, data, tiny_model, prompts, tmp_path, monkeypatch):
    round1 = write_jsonl(tmp_path / "round1.jsonl", route_rows(data["fresh/p"], {0, 1}))
    search = search_run(tmp_path / "search", SETTINGS, route_rows(data["fresh/p78"], {1, 3}))
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
                      SETTINGS, arm, round1, [search], {})
    first_stage = 1 if arm in ("eb2", "s3") else 2
    assert stages == [("phase1", first_stage, False, 8), ("phase2", 2, True, 6)]
    assert not (tmp_path / arm / "phase1/final").exists() and (tmp_path / arm / "phase2/final").exists()
    steps =[json.loads(line) for line in (tmp_path / arm / "phase1/steps.jsonl").read_text().splitlines()]
    assert {s["stage"] for s in steps} == {first_stage} and len(steps) == 1
    assert summary["training"]["phase1"]["stage"] == first_stage and summary["training"]["phase2"]["optimizer_steps"] == 1
    assert benchmarks == [arm] and summary["counts"]["rows"] == 14
    for name in ("greedy.jsonl", "masked.jsonl", "states.pt"):
        assert (tmp_path / arm / name).exists()
    greedy = [json.loads(line) for line in (tmp_path / arm / "greedy.jsonl").read_text().splitlines()]
    assert all(isinstance(r["answer"], str) and 0 < r["new_tokens"] <= SETTINGS["max_new_tokens"] for r in greedy)
    manifest = json.loads((tmp_path / arm / "manifest.json").read_text())
    assert ("hole_check_rows" in manifest["checks"]) == (arm == "s3")
    unsaved = tmp_path / f"{arm}-unsaved"
    run_arm(tiny(tiny_model), SavingProcessor(), Evaluator(), config, "unused", unsaved,
            SETTINGS | {"save_arm_weights": False}, arm, round1, [search], {})
    assert not (unsaved / "phase2/final").exists() and (unsaved / "greedy.jsonl").exists()


def test_s3r_trains_reading_rows_first_and_probes_them(data, tiny_model, prompts, tmp_path, monkeypatch):
    for rows in data.values():
        for row in rows:
            row["text_input"] = "Map: <image> Go."
    round1 = write_jsonl(tmp_path / "round1.jsonl", route_rows(data["fresh/p"], {0, 1}))
    search = search_run(tmp_path / "search", SETTINGS, route_rows(data["fresh/p78"], {1, 3}))
    stages = []
    train = training.train_stage

    def tracked(model, processor, root, output, config, stage, **kwargs):
        stages.append((output.name, stage, [row["text_output"] for _, row in kwargs["data"]]))
        return train(model, processor, root, output, config, stage, **kwargs)
    monkeypatch.setattr(expert_round2, "train_stage", tracked)
    monkeypatch.setattr(expert_round2, "run_benchmark", lambda *args: None)
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False}
    summary = run_arm(tiny(tiny_model), SavingProcessor(), Evaluator(), config, "unused", tmp_path / "s3r",
                      SETTINGS | {"save_arm_weights": False}, "s3r", round1, [search], {})
    assert [(name, stage) for name, stage, _ in stages] == [("phase0", 2), ("phase1", 1), ("phase2", 2)]
    reading, s3 = stages[0][2], stages[1][2] + stages[2][2]
    assert all(" > " not in answer and "boxed" not in answer for answer in reading)
    assert len(s3) == 14 and all(" > " in answer and "boxed" in answer for answer in s3)
    safe = sum(value == 0 for row in data["fresh/p78"] for line in row["map_desc"] for value in line)
    assert sum(len(answer.splitlines()) - 1 for answer in reading) == safe  # every safe pool cell, once
    assert summary["training"]["phase0"]["rows"] == len(reading) and summary["counts"]["rows"] == 14
    probes = summary["evaluation"]["reading"]
    assert probes["after_phase0"]["answers"] == probes["after_phase2"]["answers"] == 4  # one row per held-out map
    assert (tmp_path / "s3r/reading-phase0.jsonl").exists() and (tmp_path / "s3r/reading-phase2.jsonl").exists()


def test_g3_trains_s3_rows_with_glimpse_latents(data, tiny_model, prompts, tmp_path, monkeypatch):
    from spara import glimpse
    from spara.model import LATENT_TOKENS
    round1 = write_jsonl(tmp_path / "round1.jsonl", route_rows(data["fresh/p"], {0, 1}))
    search = search_run(tmp_path / "search", SETTINGS, route_rows(data["fresh/p78"], {1, 3}))
    answers = []
    train = training.train_stage

    def tracked(model, processor, root, output, config, stage, **kwargs):
        answers.extend(row["text_output"] for _, row in kwargs["data"])
        return train(model, processor, root, output, config, stage, **kwargs)
    monkeypatch.setattr(expert_round2, "train_stage", tracked)
    monkeypatch.setattr(expert_round2, "run_benchmark", lambda *args: None)
    monkeypatch.setattr(glimpse, "install",  # the stand-in tokenizer spells no cells
                        lambda model, processor: setattr(model.config, "livr_glimpse", {"digits": [], "comma": -1}))
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False}
    model = tiny(tiny_model)
    summary = run_arm(model, SavingProcessor(), Evaluator(), config, "unused", tmp_path / "g3",
                      SETTINGS | {"save_arm_weights": False}, "g3", round1, [search], {})
    four = "".join(LATENT_TOKENS)
    assert len(answers) == 14 and model.config.livr_glimpse
    assert all(answer.count(four) == answer.count(" hole ") > 0 for answer in answers)  # after every line's cell
    assert summary["checks"]["hole_check_rows"] == {"rows": 14, "replay": 14}  # checked before the latents go in


def test_g3r_trains_reading_rows_with_glimpses_and_stops_after_the_probes(data, tiny_model, prompts, tmp_path,
                                                                          monkeypatch):
    from spara import glimpse
    from spara.model import LATENT_TOKENS
    for rows in data.values():
        for row in rows:
            row["text_input"] = "Map: <image> Go."
    round1 = write_jsonl(tmp_path / "round1.jsonl", route_rows(data["fresh/p"], {0, 1}))
    search = search_run(tmp_path / "search", SETTINGS, route_rows(data["fresh/p78"], {1, 3}))
    stages = []
    train = training.train_stage

    def tracked(model, processor, root, output, config, stage, **kwargs):
        stages.append((output.name, stage, [row["text_output"] for _, row in kwargs["data"]]))
        return train(model, processor, root, output, config, stage, **kwargs)
    monkeypatch.setattr(expert_round2, "train_stage", tracked)
    monkeypatch.setattr(glimpse, "install",  # the stand-in tokenizer spells no cells
                        lambda model, processor: setattr(model.config, "livr_glimpse", {"digits": [], "comma": -1}))
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False}
    model = tiny(tiny_model)
    summary = run_arm(model, SavingProcessor(), Evaluator(), config, "unused", tmp_path / "g3r",
                      SETTINGS | {"save_arm_weights": False, "reading_only": True}, "g3r", round1, [search], {})
    assert [(name, stage) for name, stage, _ in stages] == [("phase0", 2)]  # no route phases
    four = "".join(LATENT_TOKENS)
    assert all(answer.count(four) == answer.count(" hole ") > 1 and " > " not in answer for answer in stages[0][2])
    probes = summary["evaluation"]["reading"]
    assert probes["after_phase0"]["answers"] == probes["after_phase0_blind"]["answers"] == 4
    assert set(summary["training"]) == {"phase0"} and "blind" not in model.config.livr_glimpse
    assert (tmp_path / "g3r/reading-phase0-blind.jsonl").exists() and not (tmp_path / "g3r/greedy.jsonl").exists()


def test_glimpse_reading_applies_the_fixed_rules(data, tmp_path):
    from spara.hole_check import holes_around
    maps = data["fresh/h78"]

    def answer(desc, listed):  # the start's line, then one named cell that has a hole beside it
        start = next((r, c) for r, line in enumerate(desc) for c, v in enumerate(line) if v == 1)
        cell = next((r, c) for r, line in enumerate(desc) for c, v in enumerate(line)
                    if v == 0 and holes_around(desc, (r, c)))
        return (f"{start[0]},{start[1]} hole {holes_around(desc, start) or '-'}\n"
                f"{cell[0]},{cell[1]} hole {holes_around(desc, cell) if listed else '-'}")

    def run(name, listed, latents=8):
        (tmp_path / name).mkdir()
        for file, said in (("reading-phase0.jsonl", listed), ("reading-phase0-blind.jsonl", False)):
            write_jsonl(tmp_path / name / file, [{"row_index": i, "answer": answer(row["map_desc"], said),
                                                  "new_tokens": 20, "latents": latents} for i, row in enumerate(maps)])
        (tmp_path / name / "summary.json").write_text("{}")
        return glimpse_reading("unused", tmp_path / name, tmp_path / name, SETTINGS | {"max_new_tokens": 384})
    reads = run("reads", True)
    assert reads["validity"]["valid"] and reads["primary"]["named_cells_listed"] == 1
    assert reads["primary"]["next"].startswith("glimpse latents read under labels")
    assert reads["descriptive"]["shares"]["blind"]["named"] == 0
    assert max(reads["descriptive"]["shares"]["g3r"]["by_direction"].values()) == 1
    assert run("none", False)["primary"]["next"].startswith("glimpse latents do not read")
    assert run("unwritten", True, latents=0)["primary"]["reading"].startswith("withheld")


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
    assert result["descriptive"]["hole_walks"]["eb2"]["maps"] == 4

    # The gate fails when EB2's latents hold no more than E2's.
    torch.save(noisy, tmp_path / "eb2/states.pt")
    failed = compare("unused", search, arms, livr, e, SETTINGS)
    assert not failed["manipulation"]["took"]
    assert failed["primary"]["reading"] == "the bottleneck did not take; no reading on perception"
    (search / "summary.json").write_text(json.dumps({"checks": {"valid": False}, "seconds": 3.0,
                                                     "frontier": {}, "pool": {}}))
    assert compare("unused", search, arms, livr, e, SETTINGS)["primary"]["reading"].startswith("withheld")


def test_with_map_text_writes_the_map_after_the_image():
    row = {"text_input": "Here is the map: <image>\n\nPlease answer.", "map_desc": GRID, "map_id": 3}
    text = with_map_text(row)["text_input"]
    assert text.count("<image>") == 1 and text.startswith("Here is the map: <image>\n")
    assert "S F F\nF H F\nF F G\n" in text and text.endswith("\n\nPlease answer.")
    assert row["text_input"].endswith("<image>\n\nPlease answer.")  # the source row is unchanged


@pytest.mark.parametrize("arm, blank", [("o2", False), ("t2", True), ("o3", False), ("s3t", False)])
def test_text_arms_train_and_evaluate_with_the_map_as_text(arm, blank, data, tiny_model, prompts, tmp_path,
                                                           monkeypatch):
    for rows in data.values():
        for row in rows:
            row["text_input"] = "Map: <image> Go."
    round1 = write_jsonl(tmp_path / "round1.jsonl", route_rows(data["fresh/p"], {0, 1}))
    search = search_run(tmp_path / "search", SETTINGS, route_rows(data["fresh/p78"], {1, 3}))
    trained, generated = [], []
    train = training.train_stage

    def tracked(model, processor, root, output, config, stage, **kwargs):
        trained.extend(row for _, row in kwargs["data"])
        return train(model, processor, root, output, config, stage, **kwargs)
    monkeypatch.setattr(expert_round2, "train_stage", tracked)
    monkeypatch.setattr(expert_round2, "run_benchmark", lambda *args: pytest.fail("no benchmark"))
    greedy = expert_round2.greedy_eval

    def seen(model, processor, evaluator, root, items, log, max_new_tokens):
        generated.extend(row for *_, row in items)
        return greedy(model, processor, evaluator, root, items, log, max_new_tokens)
    monkeypatch.setattr(expert_round2, "greedy_eval", seen)
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False}
    summary = run_arm(tiny(tiny_model), SavingProcessor(), Evaluator(), config, "unused", tmp_path / arm,
                      SETTINGS | {"save_arm_weights": False}, arm, round1, [search], {})
    assert len(trained) == 14 and all("The same map as text" in row["text_input"] for row in trained + generated)
    assert {row["blank_image"] for row in trained + generated} == {blank}
    zero_shot = [json.loads(line) for line in (tmp_path / arm / "zero-shot.jsonl").read_text().splitlines()]
    assert {r["population"] for r in zero_shot} == {"heldout78"} and len(zero_shot) == 4
    assert set(summary["evaluation"]["greedy"]) == {"heldout", "test_detour", "heldout78", "seen78"}
    assert summary["training"]["phase1"]["stage"] == (1 if arm in ("o3", "s3t") else 2)
    assert all((" hole " in row["text_output"]) == (arm == "s3t") for row in trained)
    assert ("hole_check_rows" in summary["checks"]) == (arm == "s3t")
    assert not (tmp_path / arm / "masked.jsonl").exists() and not (tmp_path / arm / "states.pt").exists()


@pytest.mark.parametrize("blank", [False, True])
def test_encode_blanks_the_image_only_when_asked(blank, tmp_path, monkeypatch):
    from PIL import Image
    from spara import serialization
    Image.new("RGB", (6, 4), "black").save(tmp_path / "map.png")
    monkeypatch.setattr(serialization, "image_path", lambda root, split, row: tmp_path / "map.png")
    shown = []

    def vision(messages):
        shown.append(messages[0]["content"][1]["image"])
        return [], None
    monkeypatch.setattr(serialization, "process_vision_info", vision)

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            return {"input_ids": torch.ones(1, 3, dtype=torch.long)}
    serialization.encode(Processor(), "root", "split", {"text_input": "A <image> B", "blank_image": blank},
                         training=False)
    assert shown[0].size == (6, 4) and shown[0].getpixel((0, 0)) == ((255, 255, 255) if blank else (0, 0, 0))


def test_ceiling_applies_the_fixed_rules(data, tmp_path, monkeypatch):
    counts = {"rows": 14}
    write_run(tmp_path / "search", data, [0, 0, 0, 0], [0, 0, 0, 0])
    for name, held in (("e2", [1, 1, 0, 0]), ("o2", [1, 1, 1, 1]), ("t2", [0, 0, 0, 0])):
        write_run(tmp_path / name, data, held, [0, 0, 0, 0])
        (tmp_path / name / "summary.json").write_text(json.dumps({"counts": counts}))
        write_jsonl(tmp_path / name / "zero-shot.jsonl", [{"population": "heldout78", "row_index": i, "grid_size": 5,
                                                           "route": "R", "success": i == 0} for i in range(4)])
    run = lambda margin, t2=None: ceiling("unused", tmp_path / "search", tmp_path / "e2", tmp_path / "o2",
                                          SETTINGS | {"ceiling_margin": margin}, t2)
    assert run(1)["primary"]["reading"] == "no detectable difference: choosing moves limits"  # 2 maps: p = 0.5
    assert "text_only" not in run(1)
    monkeypatch.setattr(expert_round2, "ALPHA", 1.01)  # four maps cannot reach 0.05: direction only
    result = run(2, tmp_path / "t2")
    assert result["validity"]["valid"] and result["primary"]["gain"] == 2
    assert result["primary"]["reading"] == "perfect perception helps a lot"
    assert run(3)["primary"]["reading"] == "perfect perception helps a little"
    assert result["text_only"]["gain"] == -2
    assert result["text_only"]["reading"] == "the policy plans worse from the text map: the ceiling is inconclusive"
    assert result["descriptive"]["held_out_success"] == {"e": 0, "e2": 2, "o2": 4, "t2": 0, "o2_zero_shot": 1,
                                                         "t2_zero_shot": 1}
    assert result["descriptive"]["t2_vs_o2"]["only_second"] == 4
    assert sum(result["descriptive"]["failures"]["e2"].values()) == 4
    (tmp_path / "t2/summary.json").write_text(json.dumps({"counts": {"rows": 13}}))
    assert not run(2, tmp_path / "t2")["validity"]["valid"]


def test_round3_applies_the_fixed_rules(data, tmp_path, monkeypatch):
    counts = {"new_maps": 2, "old_maps": 3, "rows": 14}
    summary = {"checks": {"valid": True}, "seconds": 2.0, "training": {}, "counts": counts,
               "frontier": {"solved": 1}, "pool": {"solved": 2}}
    for name, held in (("search", [0, 0, 0, 0]), ("base", [1, 0, 0, 0]), ("eb3", [1, 1, 1, 1]),
                       ("e", [0, 0, 0, 0]), ("eb2", [1, 0, 0, 0])):
        write_run(tmp_path / name, data, held, [0, 0, 0, 0], benchmark=[1, 0])
        (tmp_path / name / "summary.json").write_text(json.dumps(summary))
    write_jsonl(tmp_path / "search/routes.jsonl", [{"row_index": i, "solved": i == 0} for i in range(2)])
    livr = write_jsonl(tmp_path / "livr.jsonl", [{"row_index": i, "success": s} for i, s in enumerate([0, 1])])
    run = lambda: round3("unused", tmp_path / "search", tmp_path / "base", tmp_path / "eb3",
                         {"e": tmp_path / "e", "eb2": tmp_path / "eb2"}, livr, SETTINGS)
    result = run()
    assert result["validity"]["valid"]
    assert result["primary"]["only_first"] == 4 and result["primary"]["reading"] == "no detectable difference"
    described = result["descriptive"]
    assert described["held_out_success"] == {"b": 0, "eb3": 4, "e": 0, "eb2": 1}
    assert described["b_rerun_same_outcome"] == 4  # the base job's answers against round 2's EB2
    assert described["seen"]["maps"] == {"trained": 1, "untrained": 1}
    assert described["benchmark"]["eb3_vs_livr"]["detour"]["only_second"] == 1
    monkeypatch.setattr(expert_round2, "ALPHA", 1.01)  # four maps cannot reach 0.05: direction only
    assert run()["primary"]["reading"] == "round 3 helps"
    (tmp_path / "base/summary.json").write_text(json.dumps(summary | {"counts": counts | {"rows": 13}}))
    assert run()["primary"]["reading"].startswith("withheld")  # B did not train on EB2's rows


def test_hole_check_applies_the_fixed_rules(data, tmp_path, monkeypatch):
    from spara.hole_check import check_lines
    counts = {"new_maps": 2, "old_maps": 3, "rows": 14}
    for name, held in (("s3", [1, 1, 1, 0]), ("eb3", [0, 0, 0, 0]), ("search", [0, 0, 0, 0])):
        write_run(tmp_path / name, data, held, [0, 0, 0, 0], benchmark=[1, 0])
        checks = {"valid": True} | ({"hole_check_rows": {"rows": 14, "replay": 14}} if name == "s3" else {})
        (tmp_path / name / "summary.json").write_text(json.dumps({"checks": checks, "seconds": 2.0, "training": {},
                                                                   "counts": counts}))
    # S3's answers: three solved with right lines; map 3 lists the hole ahead and walks into it.
    rows = [json.loads(line) for line in (tmp_path / "s3/greedy.jsonl").read_text().splitlines()]
    for r in rows:
        r |= {"answer": "", "new_tokens": 5}
        if r["population"] == "heldout78":
            desc = data["fresh/h78"][r["row_index"]]["map_desc"]
            route = label_route(desc)
            if not r["success"]:  # the label's route up to the first cell beside a hole, then into it
                lines = check_lines(desc, route)
                k, holes = next((k, line.split(" hole ")[1][0]) for k, line in enumerate(lines)
                                if line.split(" hole ")[1][0] != "-")
                route = route[:k] + holes
            r |= {"route": route, "answer": "\n".join(check_lines(desc, route)) + "\n\\boxed{}"}
    write_jsonl(tmp_path / "s3/greedy.jsonl", rows)
    livr = write_jsonl(tmp_path / "livr.jsonl", [{"row_index": i, "success": s} for i, s in enumerate([0, 1])])
    run = lambda: hole_check("unused", tmp_path / "s3", tmp_path / "eb3", tmp_path / "search", livr, SETTINGS)
    result = run()
    assert result["validity"]["valid"] and result["validity"]["capped"] == 0
    assert result["primary"]["only_first"] == 3 and result["primary"]["reading"] == "no detectable difference"
    assert result["descriptive"]["fatal_moves"] == {"listed and taken": 1}
    assert result["primary"]["next"] == "train on failures as well as successes"
    assert result["descriptive"]["format"] == {"follows": 4, "answers": 4}
    lines = result["descriptive"]["lines"]
    assert lines["cell_right"] == lines["lines"] == lines["holes_exact_given_cell"]
    # G3's table on the same runs: S3's answers stand in for G3's, with four latents a line, seeing and blind.
    for name in ("greedy", "blind"):
        write_jsonl(tmp_path / f"s3/{name}.jsonl", [r | {"latents": 4 * r["answer"].count(" hole ")} for r in rows])
    glimpsed = glimpse_check("unused", tmp_path / "s3", tmp_path / "s3", tmp_path / "eb3", tmp_path / "search", livr,
                             SETTINGS)
    assert glimpsed["validity"]["valid"] and glimpsed["validity"]["four_latents_a_line"] == [4, 4]
    assert glimpsed["glimpse"]["lines_read"] and glimpsed["glimpse"]["g3_vs_blind"]["p"] == 1
    assert glimpsed["primary"]["next"].startswith("the reading is right and the routes are not")
    write_jsonl(tmp_path / "s3/greedy.jsonl", [r | {"latents": 0} for r in rows])
    assert not glimpse_check("unused", tmp_path / "s3", tmp_path / "s3", tmp_path / "eb3", tmp_path / "search", livr,
                             SETTINGS)["validity"]["valid"]  # the model did not write the latents
    write_jsonl(tmp_path / "s3/greedy.jsonl", rows)
    # The oracle check on the same runs: S3's answers, whose lines are all right, stand in for every pass.
    for name in ("free", "holes", "cell"):
        write_jsonl(tmp_path / f"s3/oracle-{name}.jsonl", rows)
    oracle = oracle_read("unused", tmp_path / "s3", tmp_path / "s3", tmp_path / "eb3", SETTINGS)
    assert oracle["validity"]["valid"] and oracle["validity"]["free_routes_as_s3"] == [4, 4]
    assert oracle["primary"]["first"] == 3 and oracle["primary"]["next"].startswith("the policy does not turn")
    assert oracle["descriptive"]["fatal_moves"]["cell"] == {"listed and taken": 1}
    # S3R's table on the same runs: S3's answers stand in for S3R's, and their lines are all right.
    probe = lambda listed: {"answers": 4, "start_right": 4, "hole_directions": 10, "hole_directions_listed": listed,
                            "capped": 0}
    summary = json.loads((tmp_path / "s3/summary.json").read_text())

    r, c = next((r, c) for r, line in enumerate(data["fresh/h78"][0]["map_desc"]) for c, v in enumerate(line) if v == 1)
    for when, answer in (("phase0", f"{r},{c} hole -\n9,9 hole U\nnot a line"), ("phase2", "a route")):
        write_jsonl(tmp_path / f"s3/reading-{when}.jsonl", [{"row_index": 0, "answer": answer}])

    def s3r(listed):
        (tmp_path / "s3/summary.json").write_text(json.dumps(summary | {"evaluation": {"reading": {
            "after_phase0": probe(listed), "after_phase2": probe(0)}}}))
        return reading_check("unused", tmp_path / "s3", tmp_path / "s3", tmp_path / "eb3", tmp_path / "search", livr,
                             SETTINGS)
    read_right = s3r(9)
    assert read_right["validity"]["valid"] and read_right["reading"]["probe_reads"]
    assert read_right["reading"]["answers_read"]
    assert read_right["primary"]["next"].startswith("the check does not help even when read right")
    assert read_right["reading"]["probe"]["after_phase2"]["hole_directions_listed"] == 0
    split = read_right["reading"]["probe_by_line"]  # the start line apart from the cells named by coordinates
    assert split["phase0"]["start_lines"] == split["phase0"]["named_lines"] == split["phase0"]["named_invented"] == 1
    assert split["phase2"] == {}
    with monkeypatch.context() as patch:
        patch.setattr(expert_round2, "line_reading",
                      lambda answers: {"answers": 4, "first_line_offset": {}, "lines": {}})
        assert s3r(9)["primary"]["next"].startswith("the reading is learned and not used")
        assert s3r(8)["primary"]["next"].endswith("report and hold")
    monkeypatch.setattr(expert_round2, "ALPHA", 1.01)  # four maps cannot reach 0.05: direction only
    assert run()["primary"]["reading"] == "writing the check helps"
    assert s3r(0)["primary"]["next"].startswith("the check becomes the policy's form")
    for r in rows:
        r["new_tokens"] = SETTINGS["max_new_tokens"]
    write_jsonl(tmp_path / "s3/greedy.jsonl", rows)
    assert run()["primary"]["reading"].startswith("withheld")  # every answer reached the cap


def test_text_check_applies_the_fixed_rules(data, tmp_path, monkeypatch):
    from spara.hole_check import check_lines
    counts = {"new_maps": 2, "old_maps": 3, "rows": 14}
    for name, held in (("s3t", [1, 1, 1, 0]), ("o3", [0, 0, 0, 0]), ("s3", [0, 0, 0, 0]), ("eb3", [0, 0, 0, 0])):
        write_run(tmp_path / name, data, held, [0, 0, 0, 0])
        checks = {"valid": True} | ({"hole_check_rows": {"rows": 14, "replay": 14}} if name == "s3t" else {})
        (tmp_path / name / "summary.json").write_text(json.dumps({"checks": checks, "seconds": 2.0, "training": {},
                                                                   "counts": counts}))
        rows = [json.loads(line) for line in (tmp_path / name / "greedy.jsonl").read_text().splitlines()]
        for r in rows:
            r |= {"answer": "", "new_tokens": 5}
            if r["population"] == "heldout78" and name in ("s3t", "s3"):
                desc = data["fresh/h78"][r["row_index"]]["map_desc"]
                route = label_route(desc)
                lines = check_lines(desc, route)
                if name == "s3":  # S3 names the start one column off, and lists no holes
                    lines = [f"0,9 hole - > {letter}" for letter in route]
                r |= {"route": route, "answer": "\n".join(lines) + "\n\boxed{}"}
        write_jsonl(tmp_path / name / "greedy.jsonl", rows)
        if name in ("s3t", "o3"):
            write_jsonl(tmp_path / name / "zero-shot.jsonl",
                        [{"population": "heldout78", "row_index": i, "success": 0} for i in range(4)])
    run = lambda: text_check("unused", tmp_path / "s3t", tmp_path / "o3", tmp_path / "s3", tmp_path / "eb3", SETTINGS)
    result = run()
    assert result["validity"]["valid"] and result["validity"]["zero_shot_agree"] == [4, 4]
    assert result["primary"]["only_first"] == 3 and result["primary"]["reading"] == "no detectable difference"
    assert result["lines_read_the_text"]["reads"] and result["lines_read_the_text"]["start_right"] == 1.0
    assert result["primary"]["next"].startswith("the check does not help even when read right")
    reading = result["descriptive"]["line_reading"]
    assert reading["s3t"]["lines"]["right_cell"] > 0 and "right_cell" not in reading["s3"]["lines"]
    assert result["descriptive"]["held_out_success"] == {"s3t": 3, "o3": 0, "s3": 0, "eb3": 0}
    monkeypatch.setattr(expert_round2, "READS", 1.01)
    assert run()["primary"]["next"].startswith("the model cannot place itself")
    monkeypatch.setattr(expert_round2, "ALPHA", 1.01)  # four maps cannot reach 0.05: direction only
    helped = run()["primary"]
    assert helped["reading"] == "the check helps when the map is exact" and helped["next"].startswith("image reading")
    write_jsonl(tmp_path / "o3/zero-shot.jsonl",
                [{"population": "heldout78", "row_index": i, "success": 1} for i in range(4)])
    withheld = run()["primary"]
    assert withheld["reading"].startswith("withheld") and "next" not in withheld  # the jobs read different bases
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
