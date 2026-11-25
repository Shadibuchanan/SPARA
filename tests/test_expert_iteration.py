import json
import random

import pytest
import torch

from spara import detour_scaling, expert_iteration, route_search, training
from spara.expert_iteration import (BatchedLivr, batched_decide, compare, point_differences, route_of, run_search,
                                    run_train, training_rows)
from spara.fresh_maps import answer_text, draw_detour_layouts, label_route
from spara.jepa import decide
from spara.runtime import read_config
from spara.search import STOP
from test_detour_probe import WIDE
from test_detour_scaling import Evaluator, SavingProcessor
from test_planning_probe import GRID, Processor, Tokenizer, prompt_of

SETTINGS = {"pool_split": "fresh/p", "heldout_split": "fresh/h", "fresh_split": "fresh/probe",
            "other_fresh": ["fresh/probe"], "maps": 3, "budget": 12, "max_depth": 12, "batch": 4,
            "probe_tolerance": 0.01, "seed": 0, "extra_examples": 12, "seen_maps": 2, "epochs": 1,
            "learning_rate": 1e-5, "warmup_ratio": 0.05, "max_new_tokens": 12}


def tiny(model):
    model = model.eval()
    model.config.livr_stage = 2
    return model


def wide_prompt(batch):
    """A second prompt with a longer text and a two-token image, so rows need padding."""
    ids = torch.tensor([[1, 2, 3, 3, 5, 7, 7, 10, 11, 12, 13, 8, 9]])
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.randn(8, batch["pixel_values"].shape[1], generator=torch.Generator().manual_seed(3)),
            "image_grid_thw": torch.tensor([[1, 2, 4]])}


def test_route_of_reads_the_released_format():
    assert route_of("\\boxed{LEFT, DOWN, DOWN, RIGHT}") == "LDDR" and route_of("\\boxed{}") == ""
    assert route_of(answer_text("UDLR")) == "UDLR"


def test_batched_reads_match_single_passes_across_prompts(tiny_model, tiny_batch):
    model = tiny(tiny_model)
    prompts = [prompt_of(tiny_batch), wide_prompt(tiny_batch), prompt_of(tiny_batch)]
    prefixes = ["R", "", "DDL"]
    batched = batched_decide(model, Tokenizer(), prompts, prefixes)
    for prompt, prefix, probs in zip(prompts, prefixes, batched):
        single = decide(model, Tokenizer(), prompt, prefix, 0)[0]
        assert set(single) == set(probs) == {"U", "D", "L", "R", "end"}
        assert all(abs(single[k] - probs[k]) < 1e-5 for k in single)
    with pytest.raises(ValueError, match="Unexpected prompt fields"):
        batched_decide(model, Tokenizer(), [prompt_of(tiny_batch) | {"labels": None}], [""])


def problems(n):
    return [{"id": f"s/{i}", "split": "s", "row_index": i, "row": {"map_desc": WIDE}} for i in range(n)]


def test_the_batched_policy_caches_and_counts(tiny_model, tiny_batch, monkeypatch):
    prompts = {0: prompt_of(tiny_batch), 1: wide_prompt(tiny_batch), 2: prompt_of(tiny_batch)}
    monkeypatch.setattr(expert_iteration, "encode", lambda processor, root, split, row, training=True: prompts[row["i"]])
    ps = [{"id": f"s/{i}", "split": "s", "row_index": i, "row": {"i": i}} for i in range(3)]
    policy = BatchedLivr(tiny(tiny_model), Processor(), "unused", batch=2)
    replies = policy([(ps[0], [], ["U", "D", STOP]), (ps[1], ["R"], ["L", STOP]), (ps[2], [], ["U"])])
    assert [sorted(r) for r in replies] == [[STOP, "D", "U"], [STOP, "L"], ["U"]]
    assert policy.sequences == 3 and policy.forward_passes == 2
    single = decide(policy.model, Tokenizer(), prompts[1], "R", 0)[0]
    assert abs(replies[1]["L"] - torch.tensor(single["L"]).log().item()) < 1e-5
    policy([(ps[0], [], ["U"])])  # cached
    assert policy.sequences == 3
    policy.reset(1)
    assert policy.cache == {} and policy.batch == 1


def test_point_differences_select_forced_failures_on_these_maps(tiny_model, tiny_batch, tmp_path, monkeypatch):
    monkeypatch.setattr(expert_iteration, "encode", lambda *args, training=True: prompt_of(tiny_batch))
    model = tiny(tiny_model)
    probs = decide(model, Tokenizer(), prompt_of(tiny_batch), "R", 0)[0]
    base = {"split": "s", "row_index": 0, "kind": "forced", "taken_outcome": "hole", "route": "RRR", "k": 2,
            "p_moves": {m: probs[m] for m in "UDLR"}}
    lines = [base, base | {"row_index": 1, "p_moves": {m: probs[m] + 0.5 * (m == "U") for m in "UDLR"}},
             base | {"taken_outcome": "optimal"}, base | {"split": "train"}, base | {"row_index": 9}]
    path = tmp_path / "points.jsonl"
    path.write_text("".join(json.dumps(p) + "\n" for p in lines))
    policy = BatchedLivr(model, Processor(), "unused", batch=4)
    differences = point_differences(policy, problems(2), path)
    assert len(differences) == 2 and differences[0] < 1e-5 and abs(differences[1] - 0.5) < 1e-5
    assert policy.forward_passes == 1


def pool_rows(n, seed=0):
    layouts, _ = draw_detour_layouts(n, random.Random(seed), set(), [4], [1])
    return [{"map_id": i, "map_desc": d, "text_output": answer_text(label_route(d))} for i, d in enumerate(layouts)]


@pytest.fixture
def data(monkeypatch):
    rows = pool_rows(9)
    sets = {"fresh/h": rows[:2], "fresh/p": rows[2:6], "fresh/probe": [rows[6] | {"map_desc": WIDE}],
            "train": [{"map_id": 0, "map_desc": GRID, "text_output": answer_text(label_route(GRID))}, rows[7]],
            "test": [{"map_id": 0, "map_desc": GRID, "text_output": answer_text(label_route(GRID))}, rows[8]]}
    for module in (expert_iteration, route_search, detour_scaling):
        monkeypatch.setattr(module, "records", lambda root, split: sets[split])
    return sets


def test_run_search_writes_routes_and_reproduction(data, tiny_model, tiny_batch, tmp_path, monkeypatch):
    monkeypatch.setattr(expert_iteration, "encode", lambda *args, training=True: prompt_of(tiny_batch))
    model = tiny(tiny_model)
    detours = route_search.problems_of("unused", "test") + route_search.problems_of("unused", "fresh/probe")
    probs = decide(model, Tokenizer(), prompt_of(tiny_batch), "", 0)[0]
    point = {"split": detours[0]["split"], "row_index": detours[0]["row_index"], "kind": "forced",
             "taken_outcome": "hole", "route": "R", "k": 1, "p_moves": {m: probs[m] for m in "UDLR"}}
    points = tmp_path / "points.jsonl"
    points.write_text(json.dumps(point) + "\n")
    reference = tmp_path / "reference.jsonl"
    reference.write_text("".join(json.dumps({"id": p["id"], "solved": False, "route": False}) + "\n" for p in detours))
    summary = run_search(model, Processor(), Evaluator(), "unused", tmp_path / "run", SETTINGS, {}, points, reference)
    assert summary["checks"]["valid"] and summary["checks"]["batch"] == 4
    assert summary["checks"]["probe_points"] == 1 and summary["checks"]["single_max_difference"] < 1e-5
    rows = [json.loads(line) for line in (tmp_path / "run/routes.jsonl").read_text().splitlines()]
    assert [r["row_index"] for r in rows] == [0, 1, 2] and summary["pool"]["maps"] == 3
    for r in rows:
        assert r["label_route"] == label_route(data["fresh/p"][r["row_index"]]["map_desc"])
        if r["solved"]:
            assert Evaluator().score(data["fresh/p"][r["row_index"]], r["answer"])["success"]
            assert route_of(r["answer"]) == r["route"]
    reproduced = summary["detour_maps"]["reproduction"]
    assert reproduced["recorded"] == len(detours) and reproduced["recorded_solved"] == 0
    assert reproduced["same_outcome_and_route"] == sum(not r["solved"] for r in
                                                       map(json.loads, (tmp_path / "run/detour-searches.jsonl")
                                                           .read_text().splitlines()))
    for name in ("manifest.json", "summary-before-pool.json", "detour-searches.jsonl"):
        assert (tmp_path / "run" / name).exists()


def test_run_search_falls_back_to_single_passes_when_batches_disagree(data, tiny_model, tiny_batch, tmp_path,
                                                                     monkeypatch):
    monkeypatch.setattr(expert_iteration, "encode", lambda *args, training=True: prompt_of(tiny_batch))
    point = {"split": "test", "row_index": 1, "kind": "forced", "taken_outcome": "hole", "route": "R", "k": 1,
             "p_moves": {m: -1.0 for m in "UDLR"}}  # never matched: the check fails at every batch size
    points = tmp_path / "points.jsonl"
    points.write_text(json.dumps(point) + "\n")
    summary = run_search(tiny(tiny_model), Processor(), Evaluator(), "unused", tmp_path / "run", SETTINGS, {}, points)
    assert summary["checks"]["batch"] == 1 and "batched_failed" in summary["checks"] and not summary["checks"]["valid"]


def test_run_search_without_recorded_points_checks_single_passes(data, tiny_model, tiny_batch, tmp_path, monkeypatch):
    monkeypatch.setattr(expert_iteration, "encode", lambda *args, training=True: prompt_of(tiny_batch))
    summary = run_search(tiny(tiny_model), Processor(), Evaluator(), "unused", tmp_path / "run", SETTINGS, {}, None)
    checks = summary["checks"]
    assert checks["valid"] and checks["batch"] == 4 and checks["probe_points"] is None
    assert checks["single_max_difference"] < 1e-5


def routes_for(rows, solved):
    return [{"row_index": i, "map_id": r["map_id"], "solved": i in solved,
             "answer": answer_text("DDRR") if i in solved else None} for i, r in enumerate(rows)]


def test_training_rows_use_found_routes_or_labels_on_solved_maps(data):
    routes = routes_for(data["fresh/p"], {1, 3})
    found, distinct = training_rows("unused", SETTINGS, routes, "search")
    assert distinct == 2 and len(found) == 2 + 12 and found[:2] == [("train", r) for r in data["train"]]
    ids = [data["fresh/p"][i]["map_id"] for i in (1, 3)]
    assert sorted(row["map_id"] for _, row in found[2:]) == [ids[0]] * 6 + [ids[1]] * 6
    assert {row["text_output"] for _, row in found[2:]} == {answer_text("DDRR")}
    labelled, _ = training_rows("unused", SETTINGS, routes, "shortest")
    assert [row["map_id"] for _, row in labelled] == [row["map_id"] for _, row in found]
    assert [row["text_output"] for _, row in labelled[2:4]] == [data["fresh/p"][1]["text_output"],
                                                                 data["fresh/p"][3]["text_output"]]
    with pytest.raises(ValueError, match="no map"):
        training_rows("unused", SETTINGS, routes_for(data["fresh/p"], set()), "search")
    routes[1]["map_id"] = 99
    with pytest.raises(ValueError, match="does not belong"):
        training_rows("unused", SETTINGS, routes, "search")
    with pytest.raises(ValueError, match="labels"):
        training_rows("unused", SETTINGS, routes, "other")


def test_search_pool_writes_the_routes_found_for_part_of_the_pool(data, tiny_model, tiny_batch, tmp_path, monkeypatch):
    from spara.expert_iteration import search_pool
    monkeypatch.setattr(expert_iteration, "encode", lambda *args, training=True: prompt_of(tiny_batch))
    monkeypatch.setattr(expert_iteration, "CHUNK", 2)
    recipe = SETTINGS | {"search": {"budget": 8, "max_depth": 8, "seed": 0}}
    summary = search_pool(tiny(tiny_model), Processor(), Evaluator(), "unused", tmp_path / "s", recipe, 1, 4, {})
    rows = [json.loads(line) for line in (tmp_path / "s/routes.jsonl").read_text().splitlines()]
    assert [r["row_index"] for r in rows] == [1, 2, 3] and summary["pool"]["maps"] == 3 and summary["maps"] == [1, 4]
    assert summary["pool"]["cost"]["sequences"] == sum(r["policy_calls"] for r in rows)
    for r in rows:
        assert r["policy_calls"] <= 8
        if r["solved"]:
            assert Evaluator().score(data["fresh/p"][r["row_index"]], r["answer"])["success"]
    data_rows, distinct = training_rows("unused", recipe | {"extra_examples": sum(r["solved"] for r in rows)}, rows,
                                        "search") if any(r["solved"] for r in rows) else ([], 0)
    assert distinct == sum(r["solved"] for r in rows)
    assert sum(split == "fresh/p" for split, _ in data_rows) == distinct  # one example a solved map
    with pytest.raises(ValueError, match="pool has"):
        search_pool(tiny(tiny_model), Processor(), Evaluator(), "unused", tmp_path / "t", recipe, 0, 9, {})


@pytest.mark.parametrize("labels", ["search", "shortest"])
def test_run_train_end_to_end(labels, data, tiny_model, tiny_batch, tmp_path, monkeypatch):
    prompt = lambda *args, **kwargs: {k: v.clone() for k, v in tiny_batch.items()}
    monkeypatch.setattr(training, "encode", prompt)
    monkeypatch.setattr(detour_scaling, "encode", lambda *args, training=True: prompt_of(tiny_batch))
    benchmarks = []

    def benchmark(model, processor, config, root, output, manifest):
        output.mkdir()
        (output / "responses.jsonl").write_text(json.dumps({"row_index": 0, "success": True}) + "\n")
        benchmarks.append(manifest["arm"])
    monkeypatch.setattr(detour_scaling, "run_benchmark", benchmark)
    routes = tmp_path / "routes.jsonl"
    routes.write_text("".join(json.dumps(r) + "\n" for r in routes_for(data["fresh/p"], {0, 2})))
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False}
    summary = run_train(tiny(tiny_model), SavingProcessor(), Evaluator(), config, "unused", tmp_path / labels,
                        SETTINGS, labels, routes, {})
    assert summary["distinct"] == 2 and summary["training"]["optimizer_steps"] == 2  # 14 rows, accumulation 8
    assert benchmarks == (["search"] if labels == "search" else [])
    manifest = json.loads((tmp_path / labels / "manifest.json").read_text())
    assert manifest["labels"] == labels and manifest["distinct"] == 2
    assert (tmp_path / labels / "stage2/final").is_dir() == (labels == "search")  # E's weights start round 2


def write_arm(path, held, seen, benchmark=None, training=None):
    """greedy.jsonl with `held` over 8 held-out and 2 test detour maps, `seen` over 4 seen maps."""
    path.mkdir()
    rows = [{"population": "heldout", "row_index": i, "grid_size": 4 + i % 2, "route": "R", "success": held[i]}
            for i in range(8)]
    rows += [{"population": "test_detour", "row_index": 100 + i, "grid_size": 6, "route": "R", "success": held[8 + i]}
             for i in range(2)]
    rows += [{"population": "seen", "row_index": i, "grid_size": 4, "route": "R", "success": seen[i]} for i in range(4)]
    (path / "greedy.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path / "summary.json").write_text(json.dumps({"checks": {"valid": True}, "training": training, "seconds": 2.0}))
    if benchmark is not None:
        (path / "evaluation").mkdir()
        (path / "evaluation/responses.jsonl").write_text(
            "".join(json.dumps({"row_index": i, "success": s}) + "\n" for i, s in enumerate(benchmark)))


def test_compare_applies_the_fixed_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(detour_scaling, "ALPHA", 0.6)  # ten maps cannot reach 0.05
    write_arm(tmp_path / "search", [1] * 10, [1, 1, 0, 0], benchmark=[1] * 101 + [0], training={"loss": 1})
    write_arm(tmp_path / "shortest", [1] * 9 + [0], [1, 0, 0, 0], training={"loss": 1})
    write_arm(tmp_path / "livr", [0] * 10, [0, 0, 0, 0])
    write_arm(tmp_path / "n500", [1] * 5 + [0] * 5, [1, 1, 1, 0])
    search = tmp_path / "searchrun"
    search.mkdir()
    (search / "summary.json").write_text(json.dumps({"checks": {"valid": True}, "pool": {"solved": 2},
                                                     "seconds": 3.0}))
    (search / "routes.jsonl").write_text("".join(json.dumps({"row_index": i, "solved": i < 2}) + "\n" for i in range(4)))
    outcomes = tmp_path / "livr.jsonl"
    outcomes.write_text("".join(json.dumps({"row_index": i, "success": True}) + "\n" for i in range(102)))
    runs = {n: tmp_path / n for n in ("search", "shortest", "livr", "n500")}
    result = compare(runs, search, outcomes)
    assert result["validity"]["valid"]
    assert result["primary"]["reading"] == "expert iteration helps" and result["primary"]["only_first"] == 10
    assert result["secondary"]["only_first"] == 1 and result["secondary"]["only_second"] == 0
    assert result["descriptive"]["held_out_success"] == {"search": 10, "shortest": 9, "livr": 0, "n500": 5}
    assert result["descriptive"]["seen"]["search"] == {"trained": 2, "untrained": 0}
    assert result["descriptive"]["seen"]["maps"] == {"trained": 2, "untrained": 2}
    assert result["descriptive"]["benchmark_vs_livr"]["all"]["only_second"] == 1
    assert result["seconds_in_jobs"] == {"routes": 3.0, "search": 2.0, "shortest": 2.0}
    (search / "summary.json").write_text(json.dumps({"checks": {"valid": False}, "pool": {}, "seconds": 3.0}))
    assert compare(runs, search, outcomes)["primary"]["reading"].startswith("withheld")


def test_reads_take_a_precision_and_restore_matmul_settings(tiny_model, tiny_batch, monkeypatch):
    model = tiny(tiny_model)
    before = torch.get_float32_matmul_precision()
    prompts, prefixes = [prompt_of(tiny_batch), wide_prompt(tiny_batch)], ["R", ""]
    default = batched_decide(model, Tokenizer(), prompts, prefixes)
    finer = batched_decide(model, Tokenizer(), prompts, prefixes, "tf32")
    assert torch.get_float32_matmul_precision() == before
    assert all(abs(a[k] - b[k]) < 1e-5 for a, b in zip(default, finer) for k in a)  # CPU: both FP32
    with pytest.raises(ValueError, match="precision"):
        batched_decide(model, Tokenizer(), prompts, prefixes, "fp16")
    seen = []
    monkeypatch.setattr(expert_iteration, "batched_decide",
                        lambda model, tokenizer, ps, xs, precision: seen.append(precision) or [{}] * len(ps))
    monkeypatch.setattr(expert_iteration, "encode", lambda *args, training=True: prompt_of(tiny_batch))
    policy = BatchedLivr(model, Processor(), "unused", batch=2, precision="tf32")
    policy.many([({"id": "a", "split": "s", "row": {}}, "")])
    policy.reset(1, "bf16")
    policy.many([({"id": "a", "split": "s", "row": {}}, "")])
    policy.reset(1)
    assert seen == ["tf32", "bf16"] and policy.precision == "bf16"
