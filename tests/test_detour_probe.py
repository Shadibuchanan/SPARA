import json

import torch
from transformers.generation.logits_process import TopPLogitsWarper

from spara import detour_probe
from spara.detour_probe import alignment, decision_point, is_detour, move_probabilities, processed
from spara.planning_probe import answer_ids
from test_planning_probe import GRID, Processor, Tokenizer, prompt_of

# Start (1, 0), goal (1, 3), a hole between: the shortest route (5 moves) is longer than
# the grid distance (3). From the start, R heads toward the goal; U and D are equally short.
WIDE = [[0, 0, 0, 0], [1, 0, -1, 2], [0, 0, 0, 0], [0, 0, 0, 0]]
SETTINGS = {"greedy": {"repetition_penalty": 1.05},
            "sampled": {"repetition_penalty": 1.05, "temperature": 0.7, "top_k": 50, "top_p": 0.9}}


def test_detour_maps():
    assert is_detour(WIDE) and not is_detour(GRID)


def test_decision_points():
    kind = lambda grid, route: (lambda p: p and (p["kind"], p["k"]))(decision_point(grid, route))
    assert kind(GRID, "RRDD") is None  # optimal throughout, no detour needed
    assert kind(GRID, "RDRD") == ("deviation", 2)  # into the hole
    assert kind(GRID, "RR") == ("stopped", 3)
    assert kind(GRID, "RRDDD") == ("deviation", 5)  # a move after reaching the goal
    assert kind(WIDE, "RRR") == ("forced", 2)  # every optimal move from (1, 1) leads away
    assert kind(WIDE, "RURRD") == ("forced", 2)  # the detour taken where it was forced
    assert kind(WIDE, "URRRD") == ("chosen", 1)  # an optimal away move while R was also optimal
    assert kind(WIDE, "L") == ("deviation", 1)  # off the grid
    point = decision_point(WIDE, "RRR")
    assert point["cell"] == [1, 1] and point["outcomes"] == {"U": "optimal", "D": "optimal", "L": "safe", "R": "hole"}


def test_alignment_finds_the_move_and_the_end_decision():
    tokenizer = Tokenizer()
    context, at, split, tokens, end, cont = alignment(tokenizer, "")
    assert context.tolist() == [14] and at == split == 1 and end == 20  # "\boxed{" then a move or "}"
    context, at, split, tokens, end, cont = alignment(tokenizer, "RD")
    assert context.tolist() == [14, 18, 19, 16, 19] and at == 5 and split == 4
    assert tokens == {"U": 15, "D": 16, "L": 17, "R": 18} and (end, cont) == (20, 19)


def test_top_p_gives_moves_outside_the_nucleus_zero_probability():
    logits = torch.tensor([5.0, 4.0, 0.0, -3.0])
    probs = torch.softmax(processed(logits, torch.tensor([7]), {"temperature": 0.7, "top_p": 0.9}), -1)
    reference = torch.softmax(TopPLogitsWarper(0.9)(None, (logits / 0.7)[None])[0], -1)
    torch.testing.assert_close(probs, reference)
    assert probs[2] == 0 and probs[3] == 0 and probs[:2].sum() > 0.999


def test_repetition_penalty_lowers_tokens_already_present():
    logits = torch.tensor([2.0, 2.0, -1.0])
    scores = processed(logits, torch.tensor([0, 2]), {"repetition_penalty": 2.0})
    assert scores.tolist() == [1.0, 2.0, -2.0]


def test_move_probabilities_match_teacher_forced_routes(tiny_model, tiny_batch):
    model = tiny_model.eval()
    model.config.livr_stage = 2
    prompt, tokenizer = prompt_of(tiny_batch), Tokenizer()
    probs = move_probabilities(model, tokenizer, prompt, "RD", SETTINGS)
    raw = probs["raw"]
    assert abs(sum(raw["moves"].values()) + raw["end"] + raw["other"] - 1) < 1e-5
    for move, value in raw["moves"].items():  # P(", ") x P(move), from one teacher-forced pass each
        ids = answer_ids(tokenizer, "RD" + move)
        full = torch.cat([prompt["input_ids"], ids[None]], -1)
        with torch.no_grad():
            logp = torch.log_softmax(model(**prompt | {"input_ids": full, "attention_mask": torch.ones_like(full)})
                                     .logits[0].float(), -1)
        length = prompt["input_ids"].shape[-1]
        expected = logp[length + 3, ids[4]] + logp[length + 4, ids[5]]
        assert abs(value - expected.exp().item()) < 1e-5
    sampled = probs["sampled"]
    assert abs(sum(sampled["moves"].values()) + sampled["end"] + sampled["other"] - 1) < 1e-5
    assert probs["argmax"] in ("U", "D", "L", "R", "end", "other")


def test_detour_probe_runs_end_to_end(tiny_model, tiny_batch, tmp_path, monkeypatch):
    model = tiny_model.eval()
    model.config.livr_stage = 2
    model.generation_config.repetition_penalty = 1.05
    data = {"test": [{"map_desc": GRID}, {"map_desc": WIDE}, {"map_desc": WIDE}],
            "fresh/demo": [{"map_desc": WIDE}, {"map_desc": GRID}],
            "train": [{"map_desc": WIDE, "text_output": "\\boxed{RIGHT, UP, RIGHT, RIGHT, DOWN}"},
                      {"map_desc": GRID, "text_output": "\\boxed{RIGHT, RIGHT, DOWN, DOWN}"}]}
    monkeypatch.setattr(detour_probe, "records", lambda root, split: data[split])
    monkeypatch.setattr(detour_probe, "encode", lambda *args, training=True: prompt_of(tiny_batch))
    test_routes = tmp_path / "routes.jsonl"
    test_routes.write_text("".join(json.dumps(r) + "\n" for r in [
        {"split": "test", "row_index": 0, "route": "RRDD", "completion": "\\boxed{RIGHT, RIGHT, DOWN, DOWN}<|im_end|>"},
        {"split": "test", "row_index": 1, "route": "RRR", "completion": "\\boxed{RIGHT, RIGHT, RIGHT}<|im_end|>"},
        {"split": "test", "row_index": 2, "route": "URRRD", "completion": "\\boxed{UP, RIGHT}<|im_end|>"},
        {"split": "train", "row_index": 0, "route": "RURRD", "completion": ""}]))
    fresh_routes = tmp_path / "sweep.jsonl"
    fresh_routes.write_text(json.dumps({"row_index": 0, "greedy_route": "RR"}) + "\n"
                            + json.dumps({"row_index": 1, "greedy_route": "RDRD"}) + "\n")
    summary = detour_probe.detour_probe(model, Processor(), "unused", tmp_path / "run", test_routes, fresh_routes,
                                        "fresh/demo", {"temperature": 0.7, "top_p": 0.9}, {}, 0)
    points = [json.loads(line) for line in (tmp_path / "run/points.jsonl").read_text().splitlines()]
    unseen = [(p["split"], p["row_index"], p["kind"], p["k"], p["taken"]) for p in points if p["split"] != "train"]
    assert unseen == [("test", 1, "forced", 2, "R"), ("test", 2, "chosen", 1, "U"),
                      ("fresh/demo", 0, "forced", 2, "R"), ("fresh/demo", 1, "deviation", 2, "D")]
    assert [p["canonical"] for p in points if p["split"] == "test"] == [True, False]
    released = [p for p in points if p["split"] == "train"]
    assert len(released) == 1 and released[0]["kind"] == "forced" and released[0]["taken"] == "U"
    assert 0 <= released[0]["p_taken"] <= 1
    assert summary["primary"]["points"] == 2 and summary["validity"]["training_maps"] == 1
    assert summary["kinds"]["test"] == {"forced": 1, "deviation": 0, "stopped": 0, "chosen": 1}
    assert summary["reading"] in ("tractable", "partial", "hard") or summary["reading"].startswith("withheld")
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["decoding"]["sampled"] == {"repetition_penalty": 1.05, "temperature": 0.7, "top_k": 50,
                                               "top_p": 0.9}
