import json
import random

import pytest
import torch
from mathruler.grader import extract_boxed_content

from spara import planning_probe
from spara.latent_rl import LatentAdjuster
from spara.planning_probe import (CLASSES, completion_nll, control_routes, move_tokens, neighbours, optimise,
                                  shortest_routes, state_probe, teacher_batch, tracking_test, walk)

SUPPRESS = [2, 3, 4, 5, 10, 11, 12, 13]  # image markers and latent tokens
PIECES = {14: "\\boxed{", 15: "UP", 16: "DOWN", 17: "LEFT", 18: "RIGHT", 19: ", ", 20: "}", 6: "<|im_end|>"}


class Tokenizer:
    """Fixed pieces per token ID; anything else decodes to '#'."""
    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=False):
        return "".join(PIECES.get(int(i), "#") for i in ids)

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        ids = []
        while text:
            token, piece = max(((t, p) for t, p in PIECES.items() if text.startswith(p)), key=lambda x: len(x[1]))
            ids.append(token)
            text = text[len(piece):]
        return {"input_ids": torch.tensor([ids])}


class Processor:
    tokenizer = Tokenizer()


class Evaluator:
    """The reference rules on the boxed answer."""
    def __init__(self, root=None):
        pass

    def score(self, row, text):
        extracted = extract_boxed_content(text.split("<|im_start|>assistant")[-1])
        route = "".join(ch for ch in extracted.upper() if ch in "UDRL")
        return {"extracted_answer": extracted, "success": walk(row["map_desc"], route)[1] == "goal"}


def prompt_of(batch):
    prompt = {k: v for k, v in batch.items() if k != "labels"}
    prompt["input_ids"] = prompt["input_ids"][:, :11]
    prompt["attention_mask"] = prompt["attention_mask"][:, :11]
    return prompt


GRID = [[1, 0, 0], [0, -1, 0], [0, 0, 2]]


def test_move_tokens_follow_the_reference_parser():
    tokenizer = Tokenizer()
    ids = tokenizer("\\boxed{DOWN, LEFT, UP}<|im_end|>")["input_ids"][0]
    assert move_tokens(tokenizer, ids) == [("D", 1), ("L", 3), ("U", 5)]
    assert move_tokens(tokenizer, torch.tensor([7, 15, 6])) == []  # no boxed answer
    assert move_tokens(tokenizer, torch.tensor([14, 15, 19, 16])) == []  # unclosed


def test_walk_and_neighbours():
    steps, status = walk(GRID, "RD")
    assert status == "hole" and [s["cell"] for s in steps] == [[0, 0], [0, 1]] and steps[-1]["fatal"]
    steps, status = walk(GRID, "URRDD")  # the first move leaves the grid and is ignored
    assert status == "goal" and steps[1]["cell"] == [0, 0] and not any(s["fatal"] for s in steps)
    assert walk(GRID, "R")[1] == "short"
    assert neighbours(GRID, [0, 1]) == [None, 1.0, 0.0, 0.0]  # U, D, L, R


def test_shortest_routes_and_controls():
    routes, capped = shortest_routes(GRID, 64)
    assert sorted(routes) == ["DDRR", "RRDD"] and not capped
    assert shortest_routes(GRID, 1) == (["DDRR"], True)
    open_grid = [[1, 0, 0], [0, 0, 0], [0, 0, 2]]
    assert len(shortest_routes(open_grid, 64)[0]) == 6
    controls = control_routes(GRID, "RRDD", own="RDDD")
    assert "RDDD" not in controls and "DRDD" in controls  # DRDD enters the hole
    assert all(walk(GRID, r)[1] != "goal" for r in controls) and all(len(r) == 4 for r in controls)


def step(size, k, solved, fatal=False):
    return {"size": size, "k": k, "solved": solved, "fatal": fatal}


def test_tracking_rule():
    solved = [step(4, k, True) for k in (1, 2, 2, 3) for _ in range(10)]
    fatal = [step(4, 2, False, True)] * 10 + [step(4, 5, False, True)] * 10 + [step(4, 1, False, True)] * 3
    intact = tracking_test(solved + fatal, [True] * len(solved) + [True] * len(fatal))
    assert intact["reading"] == "tracking intact" and intact["tested"] == 20 and intact["fatal_first_moves"] == 3
    assert intact["nearest_k_used"] == 10  # k = 5 falls back to k = 3
    broken = tracking_test(solved + fatal, [True] * len(solved) + [False] * len(fatal))
    assert broken["reading"] == "tracking contributes" and broken["p_one_sided"] < 1e-6


SAFE = [[r, c] for r in range(4) for c in range(4) if [r, c] not in ([1, 2],)]
LAYOUT = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 0], [0, 0, 0, 2]]


def synthetic_split(maps, failed, rng, corrupt):
    """Steps whose layer-2 state encodes the cell (row and column one-hot); failed maps end
    with a fatal move from (1, 1) into the hole at (1, 2)."""
    records, steps, states = [], [], []
    for m in range(maps):
        fails = m < failed
        cells = [rng.choice(SAFE) for _ in range(2)] + ([[1, 1]] if fails else [rng.choice(SAFE)])
        for k, cell in enumerate(cells, 1):
            fatal = fails and k == 3
            move = "R" if fatal else rng.choice("UDLR")
            shown = [(cell[0] + 2) % 4, cell[1]] if fatal and corrupt else cell
            code = torch.zeros(2 * CLASSES)
            code[shown[0]] = code[CLASSES + shown[1]] = 3.0
            states.append(torch.stack([torch.randn(2 * CLASSES), code + 0.1 * torch.randn(2 * CLASSES)]))
            steps.append({"map": m, "row_index": m, "size": 4, "k": k, "solved": not fails, "cell": cell,
                          "move": move, "fatal": fatal, "neighbours": neighbours(LAYOUT, cell)})
        records.append({"success": not fails, "dropped_steps": 0})
    return {"maps": records, "steps": steps, "states": torch.stack(states).half()}


@pytest.mark.parametrize("corrupt,reading", [(False, "tracking intact"), (True, "tracking contributes")])
def test_state_probe_on_synthetic_states(corrupt, reading):
    torch.manual_seed(0)
    rng = random.Random(0)
    features = {"layers": [0, 1], "train": synthetic_split(80, 0, rng, False),
                "test": synthetic_split(40, 15, rng, corrupt)}
    grids = {"train": [LAYOUT] * 80, "test": [LAYOUT] * 40}
    report = state_probe(features, grids, seed=0)
    assert report["layer"] == 2 and report["held_out_position_accuracy"]["2"] > 0.95
    test = report["splits"]["test"]
    assert test["solved_step_accuracy"] > 0.95 and test["S1"]["reading"] == reading
    assert test["S2"]["fatal_steps_with_wrong_cell"] == (15 if corrupt else 0)
    assert test["S3"]["fatal_holes"] == 15 and len(report["steps"]) == 120


def test_padded_rows_score_like_single_rows(tiny_model, tiny_batch):
    model = tiny_model.eval()
    model.config.livr_stage = 2
    prompt = prompt_of(tiny_batch)
    short, long = torch.tensor([14, 6]), torch.tensor([14, 15, 19, 16, 20, 6])
    with torch.no_grad():
        batch, ids, mask = teacher_batch(prompt, [short, long], 0)
        together, _ = completion_nll(model, batch, ids, mask, 11)
        alone = [completion_nll(model, *teacher_batch(prompt, [c], 0), 11)[0][0] for c in (short, long)]
    torch.testing.assert_close(together, torch.stack(alone), atol=1e-5, rtol=0)


def tiny_basis(hidden, k=2):
    directions = torch.stack([torch.linalg.qr(torch.randn(hidden, k, generator=torch.Generator().manual_seed(s)))[0]
                              for s in range(4)])
    return {"directions": directions, "scales": torch.ones(4, k) * 5.0, "captured": [1.0] * 4, "k": k, "maps": 1}


def test_optimise_stays_in_the_ball_and_lowers_nll(tiny_model, tiny_batch):
    model = tiny_model.eval()
    model.config.livr_stage = 2
    for p in model.parameters():
        p.requires_grad_(False)
    model.lm_head.weight.mul_(100)  # random init leaves every logit near zero, whatever the latents
    adjuster = LatentAdjuster(model, 2)
    target = torch.tensor([14, 15, 19, 16, 20, 6])
    kept, start = optimise(model, adjuster, tiny_basis(32), prompt_of(tiny_batch), [target, target], [0.5, 50.0],
                           steps=30, learning_rate=0.5, pad_id=0)
    assert all(p.grad is None for p in model.parameters())
    assert max(kept[0]["a"].norm(dim=-1)) <= 0.5 + 1e-5
    assert kept[1]["nll"] < start[1] - 0.5 and kept[0]["nll"] < start[0]


def test_planning_probe_runs_end_to_end(tiny_model, tiny_batch, tmp_path, monkeypatch):
    model = tiny_model.eval()
    model.config.livr_stage = 2
    model.generation_config.suppress_tokens = SUPPRESS
    open_grid = [[1, 0, 0], [0, 0, 0], [0, 0, 2]]
    data = {"test": [{"map_desc": GRID}, {"map_desc": open_grid}],
            "train": [{"map_desc": grid, "text_output": answer}
                      for grid, answer in [(GRID, "\\boxed{RIGHT, RIGHT, DOWN, DOWN}"),
                                           (open_grid, "\\boxed{DOWN, RIGHT, DOWN, RIGHT}")] * 3],
            "fresh/demo": [{"map_desc": [[1, 2, 0, 0, 0]] + [[0] * 5] * 4}]}
    monkeypatch.setattr(planning_probe, "records", lambda root, split: data[split])
    monkeypatch.setattr(planning_probe, "encode", lambda *args, training=True: prompt_of(tiny_batch))
    monkeypatch.setattr(planning_probe, "ReferenceEvaluator", Evaluator)
    torch.save(tiny_basis(32), tmp_path / "basis.pt")
    torch.manual_seed(0)
    summary = planning_probe.planning_probe(model, Processor(), "unused", tmp_path / "run", tmp_path / "basis.pt", 2,
                                            {}, 0, fresh_split="fresh/demo", radii=(1.0, 4.0), steps=2,
                                            max_new_tokens=4)
    routes = [json.loads(line) for line in (tmp_path / "run/routes.jsonl").read_text().splitlines()]
    assert [r["split"] for r in routes] == ["test"] * 2 + ["fresh/demo"] + ["train"] * 6
    assert all(r["success"] for r in routes if r["split"] == "train")
    assert summary["state_probe"]["layer"] == 3  # the tiny model's only probe layer
    failed = [r for r in routes if r["split"] == "test" and not r["success"]]
    headroom = [json.loads(line) for line in (tmp_path / "run/headroom.jsonl").read_text().splitlines()]
    assert failed and len(headroom) == len(failed) == summary["headroom"]["maps"]
    for record in headroom:
        assert walk(data["test"][record["row_index"]]["map_desc"], record["target"])[1] == "goal"
        kinds = [(r["radius"], r["kind"]) for r in record["rows"]]
        assert kinds[:2] == [(1.0, "target"), (4.0, "target")]
        assert all(max(r["norms"]) <= r["radius"] + 1e-4 for r in record["rows"])
    assert set(summary["headroom"]["per_radius"]) == {"1.0", "4.0"}
    features = torch.load(tmp_path / "run/features.pt")
    assert features["train"]["states"].shape == (24, 1, 32)  # 6 maps x 4 moves, one probe layer


def test_planning_probe_without_headroom_needs_no_basis(tiny_model, tiny_batch, tmp_path, monkeypatch):
    model = tiny_model.eval()
    model.config.livr_stage = 2
    model.generation_config.suppress_tokens = SUPPRESS
    data = {"test": [{"map_desc": GRID}],
            "train": [{"map_desc": GRID, "text_output": "\\boxed{RIGHT, RIGHT, DOWN, DOWN}"}] * 4}
    monkeypatch.setattr(planning_probe, "records", lambda root, split: data[split])
    monkeypatch.setattr(planning_probe, "encode", lambda *args, training=True: prompt_of(tiny_batch))
    monkeypatch.setattr(planning_probe, "ReferenceEvaluator", Evaluator)
    summary = planning_probe.planning_probe(model, Processor(), "unused", tmp_path / "run", None, 2, {}, 0,
                                            radii=(), max_new_tokens=4)
    assert summary["headroom"] is None and summary["state_probe"]["layer"] == 3
    assert not (tmp_path / "run/headroom.jsonl").exists()
    assert json.loads((tmp_path / "run/manifest.json").read_text())["basis"] is None
