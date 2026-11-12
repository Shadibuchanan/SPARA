import json
import random

import pytest
from mathruler.grader import extract_boxed_content

from spara import detour_scaling, training
from spara.detour_scaling import check_sets, compare, populations, run_arm, training_data
from spara.fresh_maps import answer_text, draw_detour_layouts, label_route
from spara.planning_probe import walk
from spara.runtime import read_config
from test_planning_probe import GRID, Processor, prompt_of

RECIPE = {"heldout_split": "fresh/h", "pool_split": "fresh/p", "other_fresh": ["fresh/probe"], "extra_examples": 12,
          "seen_maps": 2, "epochs": 1, "learning_rate": 1e-5, "warmup_ratio": 0.05, "max_new_tokens": 12,
          "arms": {"livr": 0, "n2": 2, "n4": 4}}


class Evaluator:
    """The reference rules on the boxed answer, with the reference's status strings."""
    def score(self, row, text):
        extracted = extract_boxed_content(text.split("<|im_start|>assistant")[-1])
        status = walk(row["map_desc"], "".join(ch for ch in extracted.upper() if ch in "UDRL"))[1]
        return {"extracted_answer": extracted, "success": status == "goal", "status": status}


class SavingProcessor(Processor):
    def save_pretrained(self, path):
        pass


def labelled(desc, i):
    return {"map_id": i, "map_desc": desc, "text_output": answer_text(label_route(desc))}


@pytest.fixture
def data(monkeypatch):
    layouts, _ = draw_detour_layouts(9, random.Random(0), set(), [4], [1])
    sets = {"fresh/h": [labelled(d, i) for i, d in enumerate(layouts[:2])],
            "fresh/p": [labelled(d, i) for i, d in enumerate(layouts[2:6])],
            "fresh/probe": [labelled(layouts[6], 0)],
            "train": [labelled(GRID, 0), labelled(layouts[7], 1)],
            "test": [labelled(GRID, 0), labelled(layouts[8], 1)]}
    monkeypatch.setattr(detour_scaling, "records", lambda root, split: sets[split])
    return sets


def test_training_data_mixes_released_maps_with_repeated_pool_maps(data):
    mixed = training_data("unused", RECIPE, 2)
    assert mixed[:2] == [("train", r) for r in data["train"]] and len(mixed) == 2 + 12
    extra = [row["map_id"] for split, row in mixed[2:]]
    assert sorted(extra) == [0] * 6 + [1] * 6  # nested: only the first two pool maps
    assert {split for split, _ in mixed[2:]} == {"fresh/p"}
    with pytest.raises(ValueError, match="pool"):
        training_data("unused", RECIPE, 5)


def test_set_checks_catch_bad_labels_and_shared_layouts(data):
    checks = check_sets(Evaluator(), "unused", RECIPE)
    assert checks["valid"] and checks["labels_accepted"] == [6, 6]
    assert "train|test" not in checks["shared_layouts"]  # the released splits' own overlap is not checked
    data["fresh/p"][0]["text_output"] = "\\boxed{UP}"
    assert check_sets(Evaluator(), "unused", RECIPE)["labels_accepted"] == [5, 6]
    data["fresh/p"][0] = labelled(data["fresh/h"][0]["map_desc"], 0)
    checks = check_sets(Evaluator(), "unused", RECIPE)
    assert not checks["valid"] and checks["shared_layouts"]["fresh/h|fresh/p"] == 1


def test_populations_cover_held_out_test_detours_and_seen_maps(data):
    items = [(p, s, i) for p, s, i, _ in populations("unused", RECIPE)]
    assert items == [("heldout", "fresh/h", 0), ("heldout", "fresh/h", 1), ("test_detour", "test", 1),
                     ("seen", "fresh/p", 0), ("seen", "fresh/p", 1)]


def test_train_stage_reads_rows_from_several_splits(tiny_model, tiny_batch, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(training, "encode", lambda processor, root, split, row: seen.append(split) or
                        {k: v.clone() for k, v in tiny_batch.items()})
    config = read_config("configs/livr.json") | {"epochs_per_stage": 1, "gradient_checkpointing": False}
    data = [("train", {"text_output": "x"})] * 5 + [("fresh/p", {"text_output": "x"})] * 4
    training.train_stage(tiny_model, SavingProcessor(), "unused", tmp_path / "run", config, 2, data=data,
                         final_weights_only=True)
    assert sorted(seen) == ["fresh/p"] * 4 + ["train"] * 5
    assert len((tmp_path / "run/steps.jsonl").read_text().splitlines()) == 2  # 9 rows, accumulation 8
    assert (tmp_path / "run/final").is_dir()
    with pytest.raises(ValueError, match="grounding"):
        training.train_stage(tiny_model, SavingProcessor(), "unused", tmp_path / "other", config, 2,
                             grounding=object(), data=data)


@pytest.mark.parametrize("arm", ["livr", "n2"])
def test_run_arm_end_to_end(arm, data, tiny_model, tiny_batch, tmp_path, monkeypatch):
    model = tiny_model.eval()
    model.config.livr_stage = 2
    prompt = lambda *args, **kwargs: {k: v.clone() for k, v in tiny_batch.items()}
    monkeypatch.setattr(training, "encode", prompt)
    monkeypatch.setattr(detour_scaling, "encode", lambda *args, training=True: prompt_of(tiny_batch))
    benchmarks = []

    def benchmark(model, processor, config, root, output, manifest):
        output.mkdir()
        (output / "responses.jsonl").write_text(json.dumps({"row_index": 0, "success": True}) + "\n")
        benchmarks.append(manifest["arm"])
    monkeypatch.setattr(detour_scaling, "run_benchmark", benchmark)
    config = read_config("configs/livr.json") | {"gradient_checkpointing": False}
    summary = run_arm(model, SavingProcessor(), Evaluator(), config, "unused", tmp_path / arm, arm, RECIPE, {})
    rows = [json.loads(line) for line in (tmp_path / arm / "greedy.jsonl").read_text().splitlines()]
    assert [r["population"] for r in rows] == ["heldout", "heldout", "test_detour", "seen", "seen"]
    assert summary["checks"]["valid"] and summary["greedy"]["heldout"][1] == 2
    if arm == "livr":
        assert summary["training"] is None and benchmarks == [] and not (tmp_path / arm / "stage2").exists()
    else:
        assert summary["training"]["optimizer_steps"] == 2 and benchmarks == ["n2"]  # 14 rows, accumulation 8
        assert (tmp_path / arm / "stage2/final").is_dir()


def write_run(path, successes, seen, routes=None, training=None, benchmark=None):
    """A run directory with greedy results: `successes` over 10 held-out maps and 2 test
    detour maps, `seen` over 4 seen maps."""
    path.mkdir()
    rows = [{"population": "heldout", "row_index": i, "grid_size": 5, "route": "R", "success": successes[i]}
            for i in range(10)]
    rows += [{"population": "test_detour", "row_index": 100 + i, "grid_size": 6,
              "route": (routes or ["R", "D"])[i], "success": successes[10 + i]} for i in range(2)]
    rows += [{"population": "seen", "row_index": i, "grid_size": 4, "route": "R", "success": seen[i]} for i in range(4)]
    (path / "greedy.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (path / "summary.json").write_text(json.dumps({"checks": {"valid": True}, "training": training, "seconds": 1.0}))
    if benchmark is not None:
        (path / "evaluation").mkdir()
        (path / "evaluation/responses.jsonl").write_text(
            "".join(json.dumps({"row_index": i, "success": s}) + "\n" for i, s in enumerate(benchmark)))


def test_compare_applies_the_fixed_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(detour_scaling, "AGREEMENT", 2)  # both recorded test routes must reproduce
    planning = tmp_path / "routes.jsonl"
    planning.write_text("".join(json.dumps({"split": "test", "row_index": 100 + i, "route": r}) + "\n"
                                for i, r in enumerate(["R", "D"])))
    outcomes = tmp_path / "livr.jsonl"
    outcomes.write_text("".join(json.dumps({"row_index": i, "success": i % 2 == 0}) + "\n" for i in range(102)))
    bench = [True] * 102
    none, all_ = [False] * 12, [True] * 12
    runs = {"livr": tmp_path / "l", "n98": tmp_path / "a", "n500": tmp_path / "b", "n2000": tmp_path / "c"}
    write_run(runs["livr"], none, [False] * 4)
    write_run(runs["n98"], none, [True] * 4, training={"seconds": 1}, benchmark=bench)
    write_run(runs["n500"], [True] * 5 + [False] * 7, [True] * 4, training={"seconds": 1}, benchmark=bench)
    write_run(runs["n2000"], all_, [False] * 4, training={"seconds": 1}, benchmark=bench)
    result = compare(runs, planning, outcomes)
    assert result["validity"]["valid"] and result["validity"]["reproduced_test_routes"] == [2, 2]
    assert result["primary"]["only_first"] == 12 and result["primary"]["reading"] == "variety helps"
    assert result["primary"]["by_population"]["test_detour"]["maps"] == 2
    assert result["secondary"]["reading"] == "no detectable difference"
    assert result["validity"]["seen_success"]["n2000"] == 0.0
    assert result["descriptive"]["benchmark_vs_livr"]["n98"]["detour"]["maps"] == 2

    write_run(tmp_path / "a2", none, [True, False, False, False], training={"seconds": 1}, benchmark=bench)
    result = compare(runs | {"n98": tmp_path / "a2"}, planning, outcomes)
    assert not result["validity"]["valid"] and result["primary"]["reading"].startswith("withheld")
