import copy
import json

import pytest
import torch

from spara import training
from spara.grounding import (CLASSES, OUTCOMES, GroundingHead, StepGrounding, decision_states, grounding_loss,
                             load_head, step_labels, step_targets)
from spara.planning_probe import move_tokens, step_states
from spara.runtime import read_config
from test_planning_probe import GRID, Tokenizer

OPEN = [[1, 0, 0], [0, 0, 0], [0, 0, 2]]
ROWS = [{"map_desc": grid, "text_output": answer}
        for grid, answer in [(GRID, "\\boxed{RIGHT, RIGHT, DOWN, DOWN}"), (OPEN, "\\boxed{DOWN, RIGHT, DOWN, RIGHT}"),
                             (OPEN, "\\boxed{RIGHT, DOWN, DOWN, RIGHT}"), (GRID, "\\boxed{DOWN, DOWN, RIGHT, RIGHT}")]]
SETTINGS = {"layer": 2, "weight": 1.0, "head_learning_rate": 1e-2}
PROMPT_LENGTH = 11


class Processor:
    tokenizer = Tokenizer()

    def save_pretrained(self, path):
        pass


def batch_for(tiny_batch, row):
    """The tiny prompt followed by the row's answer, labelled as spara.serialization.encode does."""
    answer = Tokenizer()(row["text_output"] + "<|im_end|>")["input_ids"]
    ids = torch.cat([tiny_batch["input_ids"][:, :PROMPT_LENGTH], answer], -1)
    labels = ids.clone()
    labels[:, :PROMPT_LENGTH] = -100
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels,
            "pixel_values": tiny_batch["pixel_values"], "image_grid_thw": tiny_batch["image_grid_thw"]}


def test_targets_follow_the_reference_walk():
    rows, cols, outcomes = step_targets(GRID, "RRDD")
    assert rows.tolist() == [0, 0, 0, 1] and cols.tolist() == [0, 1, 2, 2]
    names = [[OUTCOMES[i] for i in step] for step in outcomes.tolist()]  # U, D, L, R
    assert names == [["edge", "safe", "edge", "safe"], ["edge", "hole", "safe", "safe"],
                     ["edge", "safe", "safe", "edge"], ["safe", "goal", "hole", "edge"]]
    for failing in ("DR", "R"):  # into the hole; ends short of the goal
        with pytest.raises(ValueError):
            step_targets(GRID, failing)


def test_labels_find_the_decision_tokens():
    labels = step_labels(Tokenizer(), ROWS[0])
    # \boxed{ RIGHT , RIGHT , DOWN , DOWN } <|im_end|>: letters at answer tokens 1, 3, 5, 7.
    assert labels["positions"].tolist() == [0, 2, 4, 6] and labels["dropped"] == 0
    assert labels["rows"].tolist() == [0, 0, 0, 1] and labels["size"] == 3


def test_training_states_are_the_probes_states(tiny_model, tiny_batch):
    model = tiny_model.eval()
    model.config.livr_stage = 2
    batch = batch_for(tiny_batch, ROWS[0])
    labels = step_labels(Tokenizer(), ROWS[0])
    with torch.no_grad():
        hidden = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
    states = decision_states(hidden, SETTINGS["layer"], PROMPT_LENGTH, labels["positions"])
    prompt = {k: v for k, v in batch.items() if k != "labels"}
    prompt["input_ids"], prompt["attention_mask"] = batch["input_ids"][:, :PROMPT_LENGTH], batch["attention_mask"][:, :PROMPT_LENGTH]
    completion = batch["input_ids"][0, PROMPT_LENGTH:]
    tokens = [t for _, t in move_tokens(Tokenizer(), completion)]
    probe = step_states(model, prompt, completion, tokens, [SETTINGS["layer"] - 1])  # 0-based decoder index
    torch.testing.assert_close(states, probe[:, 0], atol=1e-6, rtol=0)


def test_masked_logits_and_gradients_stop_at_the_layer(tiny_model, tiny_batch):
    model = tiny_model.train()
    model.config.livr_stage = 2
    grounding = StepGrounding(SETTINGS, 32, 3)
    grounding.prepare(Tokenizer(), ROWS)
    batch = batch_for(tiny_batch, ROWS[0])

    def aux_backward():
        model.zero_grad(set_to_none=True)
        grounding.head.zero_grad(set_to_none=True)
        hidden = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
        loss, _ = grounding.loss(hidden, 0, batch)
        loss.backward()
        return loss

    # A zero head pushes nothing into the model, so the first update equals plain Stage 2's.
    loss = aux_backward()
    assert loss.item() == pytest.approx(0.5 * torch.log(torch.tensor(3.0)).item() + 0.5 * torch.log(torch.tensor(4.0)).item())
    assert all(p.grad is None or not p.grad.any() for p in model.parameters())
    assert grounding.head.linear.weight.grad.abs().sum() > 0

    torch.manual_seed(1)
    torch.nn.init.normal_(grounding.head.linear.weight)
    rows, cols, _ = grounding.head(torch.randn(5, 32))
    beyond = torch.arange(CLASSES) >= 3
    masked = torch.softmax(rows.masked_fill(beyond, float("-inf")), -1)
    assert torch.all(masked[:, 3:] == 0)
    aux_backward()
    layers = model.model.layers
    grads = lambda module: [p.grad for p in module.parameters() if p.requires_grad]
    assert all(g is not None and g.abs().sum() > 0 for g in grads(layers[1]))  # feeds hidden_states[2]
    assert all(g is None or not g.any() for g in grads(layers[2]))  # after the layer
    assert model.lm_head.weight.grad is None or not model.lm_head.weight.grad.any()


def test_answer_mismatch_is_rejected(tiny_model, tiny_batch):
    model = tiny_model.eval()
    model.config.livr_stage = 2
    grounding = StepGrounding(SETTINGS, 32, 3)
    grounding.prepare(Tokenizer(), ROWS)
    batch = batch_for(tiny_batch, ROWS[1])
    hidden = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
    with pytest.raises(ValueError, match="answer tokens differ"):
        grounding.loss(hidden, 0, batch)


def train(model, tiny_batch, tmp_path, monkeypatch, name, grounding=None):
    config = read_config("configs/livr.json") | {"epochs_per_stage": 2, "gradient_accumulation": 2,
                                                 "gradient_checkpointing": False}
    monkeypatch.setattr(training, "records", lambda *_: ROWS)
    monkeypatch.setattr(training, "encode", lambda *args, **kwargs: batch_for(tiny_batch, args[-1]))
    training.train_stage(model, Processor(), "unused", tmp_path / name, config, 2, grounding=grounding)
    return [json.loads(line) for line in (tmp_path / name / "steps.jsonl").read_text().splitlines()]


def test_zero_weight_matches_plain_stage_two(tiny_model, tiny_batch, tmp_path, monkeypatch):
    plain, grounded = tiny_model, copy.deepcopy(tiny_model)
    plain_log = train(plain, tiny_batch, tmp_path, monkeypatch, "plain")
    grounding = StepGrounding(SETTINGS | {"weight": 0.0}, 32, 3)
    grounded_log = train(grounded, tiny_batch, tmp_path, monkeypatch, "grounded", grounding)
    assert [r["loss"] for r in plain_log] == [r["loss"] for r in grounded_log]
    # Clipping's norm also sums the head's zero gradients, which reorders the float sum: the
    # clip factor can differ in its last bit, and Adam magnifies that where gradients are
    # near zero. Weights agree to under 0.1% of one learning-rate step (1e-5), not bitwise.
    for key, expected in plain.state_dict().items():
        torch.testing.assert_close(grounded.state_dict()[key], expected, atol=1e-8, rtol=1e-6)


def test_grounded_run_records_settings_and_saves_its_head(tiny_model, tiny_batch, tmp_path, monkeypatch):
    grounding = StepGrounding(SETTINGS, 32, 3)
    log = train(tiny_model, tiny_batch, tmp_path, monkeypatch, "run", grounding)
    assert all(r["grounding_loss"] is not None and 0 <= r["cell_accuracy"] <= 1 for r in log)
    progress = json.loads((tmp_path / "run/final/complete.json").read_text())["progress"]
    assert progress["grounding"] == SETTINGS | {"labels": {"rows": 4, "decision_states": 16, "dropped_steps": 0}}
    head, settings = load_head(tmp_path / "run/grounding_head.pt")
    assert settings == SETTINGS and head.linear.weight.abs().sum() > 0  # moved from zero
    torch.testing.assert_close(head.linear.weight, grounding.head.linear.weight.detach().cpu())
    states = torch.randn(3, 32)
    expected = grounding_loss(grounding.head, states, step_labels(Tokenizer(), ROWS[0]) | {
        k: v[:3] for k, v in step_labels(Tokenizer(), ROWS[0]).items() if k in ("rows", "cols", "outcomes")})[0]
    reloaded = grounding_loss(head, states, step_labels(Tokenizer(), ROWS[0]) | {
        k: v[:3] for k, v in step_labels(Tokenizer(), ROWS[0]).items() if k in ("rows", "cols", "outcomes")})[0]
    torch.testing.assert_close(reloaded, expected.detach())


def test_settings_are_checked():
    with pytest.raises(ValueError):
        StepGrounding(SETTINGS | {"layer": 4}, 32, 3)
    with pytest.raises(ValueError):
        StepGrounding(SETTINGS | {"head_learning_rate": 0}, 32, 3)
    assert GroundingHead(32).linear.out_features == 2 * CLASSES + 4 * len(OUTCOMES)
