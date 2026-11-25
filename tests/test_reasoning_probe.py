import json

import torch

from spara import reasoning_probe
from spara.fresh_maps import answer_text
from spara.reasoning_probe import away, run_probe, summarise


def test_away_counts_the_moves_away_from_the_goal_a_route_needs():
    assert away([[1, 0], [0, 2]]) == 0
    assert away([[1, -1, 2], [0, -1, 0], [0, 0, 0]]) == 2


def test_summarise_counts_maps_solved_by_any_sample():
    rows = [{"grid_size": 7, "away": 1, "successes": [False, True], "tokens": [10, 20], "capped": [False, False]},
            {"grid_size": 8, "away": 3, "successes": [False, False], "tokens": [30, 30], "capped": [True, False]}]
    s = summarise(rows)
    assert s["all"]["solved_by_any"] == 1 and s["all"]["mean_success"] == 0.25 and s["all"]["capped"] == 1
    assert s["size 7"]["solved_by_any"] == 1 and s["away 3"]["maps"] == 1


class Tokens:
    eos_token_id = 0

    def convert_tokens_to_ids(self, token):
        return 0

    def decode(self, ids, skip_special_tokens=True):
        return answer_text("R") if ids.tolist() and ids.tolist()[0] == 1 else answer_text("D")


class Processor:
    tokenizer = Tokens()


class Model:
    """Samples two answers per map: the first writes route R, the second route D."""
    config = type("C", (), {"livr_latent_tokens": []})()
    device = torch.device("cpu")

    def eval(self):
        return self

    def generate(self, input_ids, **kwargs):
        rows = input_ids.shape[0]
        out = torch.tensor([[1, 0]] + [[2, 2]] * (rows - 1))  # row 0 ends; the rest hit the cap
        return torch.cat([input_ids, out], dim=1)


def test_run_probe_scores_each_sample_and_asks_for_reasoning(tmp_path, monkeypatch):
    rows = [{"map_id": 0, "map_desc": [[1, 2]], "text_input": "map <image>"},
            {"map_id": 1, "map_desc": [[1], [2]], "text_input": "map <image>"}]
    asked = []

    def encode(processor, root, split, row, training, latents, generation_prompt):
        asked.append((row["text_input"], latents, generation_prompt))
        return {"input_ids": torch.tensor([[5, 6]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}

    class Evaluator:
        provenance = {}

        def __init__(self, root):
            pass

        def score(self, row, text):
            return {"success": text == answer_text("R" if row["map_id"] == 0 else "D")}

    monkeypatch.setattr(reasoning_probe, "records", lambda root, split: rows)
    monkeypatch.setattr(reasoning_probe, "encode", encode)
    monkeypatch.setattr(reasoning_probe, "ReferenceEvaluator", Evaluator)
    monkeypatch.setattr(reasoning_probe, "away", lambda desc: 0)
    summary = run_probe(Model(), Processor(), "unused", tmp_path / "p", "fresh/x", [0, 1], 2, {}, True, {})
    maps = [json.loads(line) for line in (tmp_path / "p/maps.jsonl").read_text().splitlines()]
    assert [m["successes"] for m in maps] == [[True, False], [False, True]]
    assert [m["capped"] for m in maps] == [[False, True], [False, True]] and maps[0]["tokens"] == [2, 2]
    assert summary["by"]["all"]["solved_by_any"] == 2
    assert all(text.endswith(reasoning_probe.REASON) and latents == [] and header for text, latents, header in asked)
