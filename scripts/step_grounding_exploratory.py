"""Post hoc analyses of a step-grounding run, not among the pre-declared measures in
docs/STEP_GROUNDING_SPEC.md. CPU only. Writes RUN/exploratory.json.

- Fresh-map position at layer 20 by map size: 5 x 5 and 6 x 6 (the sizes of LIVR's fresh
  probe), and 7 x 7 and 8 x 8 split by whether the cell's row and column lie in 0-5, the
  range that training maps (3 x 3 to 6 x 6) label. Both the ridge probe of the
  manipulation check and the trained head.
- Hole ahead in the direction taken (as in scripts/analyze_planning_probe.py): how often the
  probe's hole score for the move about to be written exceeds 0.5, at fatal steps against
  solved steps. It reads the probe's selected layer, which for G need not be LIVR's.
- Matched start: G's first Stage 2 answer losses against the LIVR rerun's.

Usage: python scripts/step_grounding_exploratory.py RUN [--livr-steps PATH]
(by hand on a PACE login node, set OMP_NUM_THREADS=1)
"""
import argparse
import json
from pathlib import Path

import torch

from spara.grounding import load_head
from spara.planning_probe import MOVES, decoded_cells, position_targets, ridge

EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "experiments"
LAYER = 20
TRAINED_RANGE = 6  # training maps are at most 6 x 6, so labels cover rows and columns 0-5


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def head_cells(head, states, steps):
    with torch.no_grad():
        rows, cols, _ = head(states)
    return [[int(rows[i, :s["size"]].argmax()), int(cols[i, :s["size"]].argmax())] for i, s in enumerate(steps)]


def by_group(cells, steps):
    groups = {"5x5_6x6": lambda s: s["size"] in (5, 6),
              "7x7_8x8_cell_in_trained_range": lambda s: s["size"] in (7, 8) and max(s["cell"]) < TRAINED_RANGE,
              "7x7_8x8_row_or_column_beyond": lambda s: s["size"] in (7, 8) and max(s["cell"]) >= TRAINED_RANGE}
    kinds = {"solved_steps": lambda s: s["solved"], "fatal_steps_k2": lambda s: s["fatal"] and s["k"] >= 2}
    result = {}
    for group, in_group in groups.items():
        result[group] = {}
        for kind, keep in kinds.items():
            hits = [c == s["cell"] for c, s in zip(cells, steps) if in_group(s) and keep(s)]
            result[group][kind] = {"steps": len(hits), "accuracy": sum(hits) / len(hits) if hits else None}
    return result


def hole_ahead(steps):
    taken = lambda s: s["hole_scores"][list(MOVES).index(s["move"])]
    result = {}
    for split, sizes in (("test", None), ("fresh", None), ("fresh", (5, 6))):
        rows = [s for s in steps if s["split"] == split and (sizes is None or s["size"] in sizes)]
        fatal = [taken(s) > 0.5 for s in rows if s["fatal"]]
        safe = [taken(s) > 0.5 for s in rows if s["solved"]]
        name = split if sizes is None else f"{split}_5x5_6x6"
        result[name] = {"fatal_steps": len(fatal), "fatal_read_as_hole": sum(fatal) / len(fatal) if fatal else None,
                        "safe_steps": len(safe), "safe_read_as_hole": sum(safe) / len(safe) if safe else None}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--livr-steps", default=EVIDENCE / "livr-6253300" / "steps.jsonl")
    args = parser.parse_args()
    run = Path(args.run)
    features = torch.load(run / "probe" / "features.pt", weights_only=False)
    position = [layer + 1 for layer in features["layers"]].index(LAYER)
    train, fresh = features["train"], features["fresh"]
    x = fresh["states"][:, position].float()
    ridge_cells = decoded_cells(ridge(train["states"][:, position].float(), position_targets(train["steps"]), x),
                                fresh["steps"])
    head, settings = load_head(run / "stage2" / "grounding_head.pt")
    if settings["layer"] != LAYER:
        raise ValueError("The head was trained at another layer than this analysis reads")
    grounded = [r for r in jsonl(run / "stage2" / "steps.jsonl") if r["stage"] == 2]
    livr = [r for r in jsonl(args.livr_steps) if r["stage"] == 2]
    report = {
        "note": "Post hoc; not among the pre-declared measures",
        "fresh_position_layer_20": {"ridge": by_group(ridge_cells, fresh["steps"]),
                                    "head": by_group(head_cells(head, x, fresh["steps"]), fresh["steps"])},
        "hole_ahead_direction_taken": {
            "layer": json.loads((run / "probe" / "summary.json").read_text())["state_probe"]["layer"],
            **hole_ahead(jsonl(run / "probe" / "steps.jsonl"))},
        "matched_start": [{"step": g["step"], "grounded_answer_loss": g["loss"], "livr_answer_loss": l["loss"],
                           "grounded_grad_norm_with_head": g["grad_norm"], "livr_grad_norm": l["grad_norm"]}
                          for g, l in zip(grounded[:2], livr[:2])]}
    (run / "exploratory.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
