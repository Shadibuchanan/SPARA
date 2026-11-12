"""Step grounding (docs/STEP_GROUNDING_SPEC.md): an auxiliary Stage 2 objective.

At each move of the teacher-forced released answer, a linear head reads one layer's
hidden state and predicts the agent's cell and what each of the four moves would do from
there (safe, hole, off the grid, goal). Labels come from the training row's map layout
and answer; the model never receives them. Decision states and the route walk are the
planning probe's (spara.planning_probe), so the probe measures what this trains.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from mathruler.grader import extract_boxed_content

from .latent_rl import moves
from .planning_probe import CLASSES, MOVES, move_tokens, walk

OUTCOMES = ("safe", "hole", "edge", "goal")
HEAD_FILE = "grounding_head.pt"


def outcome(grid, cell, move):
    """Index into OUTCOMES of taking `move` from `cell` under the reference rules."""
    n, (r, c) = len(grid), cell
    dr, dc = MOVES[move]
    if not (0 <= r + dr < n and 0 <= c + dc < n):
        return OUTCOMES.index("edge")
    return {-1: OUTCOMES.index("hole"), 2: OUTCOMES.index("goal")}.get(grid[r + dr][c + dc], OUTCOMES.index("safe"))


def step_targets(grid, route):
    """Per move of a successful route: the cell before it (row, column) and the outcome
    of each of U, D, L, R from that cell."""
    steps, status = walk(grid, route)
    if status != "goal":
        raise ValueError(f"Step grounding needs a route that reaches the goal, not one that ends in '{status}'")
    rows = torch.tensor([s["cell"][0] for s in steps], dtype=torch.long)
    cols = torch.tensor([s["cell"][1] for s in steps], dtype=torch.long)
    outcomes = torch.tensor([[outcome(grid, s["cell"], m) for m in MOVES] for s in steps], dtype=torch.long)
    return rows, cols, outcomes


def answer_ids(tokenizer, row):
    """The supervised continuation, tokenized as spara.serialization.encode does."""
    return tokenizer(row["text_output"] + "<|im_end|>", add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def step_labels(tokenizer, row):
    """Decision positions in the answer (index of the token whose output predicts each
    move's letter, relative to the answer's first token; -1 is the last prompt token) and
    the targets of the steps kept."""
    ids = answer_ids(tokenizer, row)
    letters = move_tokens(tokenizer, ids)
    route = "".join(m for m, _ in letters)
    if route != moves(extract_boxed_content(row["text_output"])):
        raise ValueError("Token alignment disagrees with the reference parser on a training answer")
    rows, cols, outcomes = step_targets(row["map_desc"], route)
    tokens = [t for _, t in letters]
    kept = torch.tensor([k for k, t in enumerate(tokens) if tokens.count(t) == 1], dtype=torch.long)
    return {"answer_ids": ids, "positions": torch.tensor([tokens[k] - 1 for k in kept.tolist()], dtype=torch.long),
            "rows": rows[kept], "cols": cols[kept], "outcomes": outcomes[kept], "size": len(row["map_desc"]),
            "dropped": len(tokens) - len(kept)}


def decision_states(hidden_states, layer, prefix_length, positions):
    """States of one sequence at the decision positions; hidden_states[layer] is the output
    of decoder layer `layer` (1-based), as the planning probe indexes it."""
    return hidden_states[layer][0, prefix_length + positions.to(hidden_states[layer].device)]


class GroundingHead(torch.nn.Module):
    """Parameter-free layer norm, then one zero-initialised linear map to row, column and
    per-direction outcome logits. Runs in FP32 under autocast."""

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, 2 * CLASSES + len(MOVES) * len(OUTCOMES))
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, states):
        with torch.autocast(states.device.type, enabled=False):
            out = self.linear(F.layer_norm(states.float(), states.shape[-1:]))
        return out[:, :CLASSES], out[:, CLASSES:2 * CLASSES], out[:, 2 * CLASSES:].view(-1, len(MOVES), len(OUTCOMES))


def grounding_loss(head, states, labels):
    """½ · mean(row, column CE) + ½ · mean outcome CE over the four directions, averaged
    over the decision states; with accuracies for logging."""
    rows, cols, outcomes = head(states)
    beyond = torch.arange(CLASSES, device=states.device) >= labels["size"]
    rows, cols = rows.masked_fill(beyond, float("-inf")), cols.masked_fill(beyond, float("-inf"))
    target = {k: labels[k].to(states.device) for k in ("rows", "cols", "outcomes")}
    position = 0.5 * (F.cross_entropy(rows, target["rows"]) + F.cross_entropy(cols, target["cols"]))
    move = F.cross_entropy(outcomes.reshape(-1, len(OUTCOMES)), target["outcomes"].reshape(-1))
    with torch.no_grad():
        cell = ((rows.argmax(-1) == target["rows"]) & (cols.argmax(-1) == target["cols"])).float().mean()
        ahead = (outcomes.argmax(-1) == target["outcomes"]).float().mean()
    return 0.5 * (position + move), {"cell_accuracy": float(cell), "outcome_accuracy": float(ahead)}


class StepGrounding:
    """The head, its settings and per-row labels, for spara.training.train_stage."""

    def __init__(self, settings, hidden_size, num_layers):
        if not 1 <= settings["layer"] <= num_layers:
            raise ValueError(f"Grounding layer must be a decoder layer output, 1 to {num_layers}")
        if settings["weight"] < 0 or settings["head_learning_rate"] <= 0:
            raise ValueError("Grounding weight must be non-negative and the head learning rate positive")
        self.settings, self.layer, self.weight = settings, settings["layer"], settings["weight"]
        self.head = GroundingHead(hidden_size)
        self.labels = None

    def prepare(self, tokenizer, rows):
        self.labels = [step_labels(tokenizer, row) for row in rows]
        return {"rows": len(rows), "decision_states": sum(len(x["positions"]) for x in self.labels),
                "dropped_steps": sum(x["dropped"] for x in self.labels)}

    def parameter_group(self):
        return {"params": list(self.head.parameters()), "lr": self.settings["head_learning_rate"], "weight_decay": 0.0}

    def loss(self, hidden_states, index, batch):
        labels = self.labels[index]
        supervised = (batch["labels"][0] != -100).nonzero()
        prefix_length = int(supervised[0])
        answer = batch["input_ids"][0, prefix_length:].cpu()
        if not torch.equal(answer, labels["answer_ids"]):
            raise ValueError(f"Training row {index}: the batch's answer tokens differ from the grounding labels'")
        if not len(labels["positions"]):
            zero = hidden_states[self.layer].sum() * 0
            return zero, {"cell_accuracy": None, "outcome_accuracy": None}
        return grounding_loss(self.head, decision_states(hidden_states, self.layer, prefix_length, labels["positions"]),
                              labels)

    def save(self, path):
        torch.save({"settings": self.settings, "hidden_size": self.head.linear.in_features,
                    "state_dict": {k: v.cpu() for k, v in self.head.state_dict().items()}}, path)


def load_head(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    head = GroundingHead(saved["hidden_size"])
    head.load_state_dict(saved["state_dict"])
    return head.eval(), saved["settings"]


def read_settings(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = {"layer", "weight", "head_learning_rate"} - settings.keys()
    if missing:
        raise ValueError(f"Grounding settings lack {sorted(missing)}")
    return settings
