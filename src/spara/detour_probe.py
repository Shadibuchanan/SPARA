"""Detour reach check (docs/DETOUR_LEARNING_SPEC.md, step 0) on a final LIVR checkpoint.

On the model's own recorded greedy routes, find the first step where the route needs a
detour or leaves a shortest safe route, teacher-force the route up to that step, and read
how much probability the policy gives each move: raw, and under the RL rollout sampling
settings. The reading rules are fixed in the specification and applied here.
"""
import json
import statistics
import time
from pathlib import Path

import torch
from mathruler.grader import extract_boxed_content
from transformers.generation.logits_process import (RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper,
                                                    TopKLogitsWarper, TopPLogitsWarper)

from .data import records
from .latent_rl import moves
from .planning_probe import MOVES, answer_ids, distances, move_tokens
from .runtime import json_write, seed_all
from .serialization import encode
from .training import amp, to_device

REACH = 0.01  # within reach: optimal moves have at least this raw probability
TRACTABLE, HARD = 0.5, 0.2  # reading thresholds on the share within reach
AGREEMENT, MEMORISED = 0.9, 0.9  # validity thresholds


# Maps and decision points ----------------------------------------------------------

def find(grid, value):
    n = len(grid)
    return next((r, c) for r in range(n) for c in range(n) if grid[r][c] == value)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step(cell, move):
    return cell[0] + MOVES[move][0], cell[1] + MOVES[move][1]


def is_detour(grid):
    """The shortest safe route is longer than the grid distance from start to goal."""
    start, goal = find(grid, 1), find(grid, 2)
    return distances(grid)[start] > manhattan(start, goal)


def outcomes(grid, dist, cell):
    """For each move from `cell`: optimal (on a shortest safe route), hole, edge or safe."""
    n, result = len(grid), {}
    for move in MOVES:
        r, c = step(cell, move)
        if not (0 <= r < n and 0 <= c < n):
            result[move] = "edge"
        elif grid[r][c] == -1:
            result[move] = "hole"
        elif dist.get((r, c), -2) == dist.get(cell, -1) - 1:
            result[move] = "optimal"
        else:
            result[move] = "safe"
    return result


def decision_point(grid, route):
    """The first step of `route` that decides a detour or a failure; None for a route
    that stays on shortest safe routes on a map needing no detour.

    Returns {"kind", "k" (1-based index of the move decided), "cell", "outcomes"} with kind
    forced (every optimal move leads away from the goal), deviation (the move taken is not
    optimal), stopped (the route ends before the goal with every move optimal), or chosen
    (the first move away from the goal of a fully optimal route on a detour map)."""
    dist = distances(grid)
    goal, cell = find(grid, 2), find(grid, 1)
    first_away = None
    for k in range(1, len(route) + 2):
        if dist.get(cell) == 0:
            if k <= len(route):  # moves after reaching the goal
                return {"kind": "deviation", "k": k, "cell": list(cell), "outcomes": outcomes(grid, dist, cell)}
            return first_away
        here = outcomes(grid, dist, cell)
        optimal = [m for m, o in here.items() if o == "optimal"]
        point = {"k": k, "cell": list(cell), "outcomes": here}
        if optimal and all(manhattan(step(cell, m), goal) > manhattan(cell, goal) for m in optimal):
            return point | {"kind": "forced"}
        if k > len(route):
            return point | {"kind": "stopped"}
        move = route[k - 1]
        if here[move] != "optimal":
            return point | {"kind": "deviation"}
        if first_away is None and manhattan(step(cell, move), goal) > manhattan(cell, goal):
            first_away = point | {"kind": "chosen"}
        cell = step(cell, move)
    raise AssertionError("unreachable")


# Policy probabilities --------------------------------------------------------------

def decoding_settings(model, generation):
    """What generate applies on top of the logits: the checkpoint's repetition penalty and
    top-k (Hugging Face's default when unset) for both modes, and the RL rollouts'
    temperature and top-p for sampling."""
    config = model.generation_config
    penalty = config.repetition_penalty if config.repetition_penalty not in (None, 1.0) else None
    return {"greedy": {"repetition_penalty": penalty},
            "sampled": {"repetition_penalty": penalty, "temperature": generation["temperature"],
                        "top_k": generation.get("top_k", config.top_k), "top_p": generation["top_p"]}}


def processed(logits, input_ids, settings):
    """Scores after Hugging Face's processors in generate's order: repetition penalty, then
    temperature, top-k and top-p. `input_ids` is the sequence before the predicted token."""
    scores = logits[None].float()
    if settings.get("repetition_penalty"):
        scores = RepetitionPenaltyLogitsProcessor(settings["repetition_penalty"])(input_ids[None], scores)
    if settings.get("temperature") is not None:
        scores = TemperatureLogitsWarper(settings["temperature"])(input_ids[None], scores)
    if settings.get("top_k"):
        scores = TopKLogitsWarper(settings["top_k"])(input_ids[None], scores)
    if settings.get("top_p") is not None:
        scores = TopPLogitsWarper(settings["top_p"])(input_ids[None], scores)
    return scores[0]


def alignment(tokenizer, prefix):
    """Token IDs of the answer up to the move after `prefix`, the position of that move's
    token, the position where continuing and ending the route diverge, and the tokens for
    each move and for ending."""
    k = len(prefix)
    ids = {m: answer_ids(tokenizer, prefix + m) for m in MOVES}
    positions = {m: move_tokens(tokenizer, ids[m])[k][1] for m in MOVES}
    at = positions["U"]
    if any(positions[m] != at or not torch.equal(ids[m][:at], ids["U"][:at]) for m in MOVES):
        raise ValueError("The move alternatives do not share their answer prefix")
    tokens = {m: int(ids[m][at]) for m in MOVES}
    if len(set(tokens.values())) != len(MOVES):
        raise ValueError("Two moves share their first token")
    end = answer_ids(tokenizer, prefix)
    split = next(i for i in range(min(len(end), at + 1)) if i == at or end[i] != ids["U"][i])
    if not torch.equal(end[:split], ids["U"][:split]):
        raise ValueError("Ending and continuing the route do not share their prefix")
    return ids["U"][:at], at, split, tokens, int(end[split]), int(ids["U"][split])


@torch.no_grad()
def move_probabilities(model, tokenizer, prompt, prefix, settings):
    """Probability of each move after `prefix` (a route string), of ending the route there,
    and of anything else: raw (the model's softmax) and under the sampling settings; and the
    greedy choice under the greedy settings (`settings` from decoding_settings)."""
    context, at, split, tokens, end_token, continue_token = alignment(tokenizer, prefix)
    length = prompt["input_ids"].shape[-1]
    ids = torch.cat([prompt["input_ids"], context[None].to(prompt["input_ids"].device)], -1)
    with amp(model.device):
        logits = model(**prompt | {"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                       use_cache=False).logits[0].float()
    rows = {"move": length + at - 1, "choice": length + split - 1}  # positions predicting each token
    dist = lambda name, mode: (torch.softmax(logits[rows[name]], -1) if mode == "raw" else
                               torch.softmax(processed(logits[rows[name]], ids[0, :rows[name] + 1],
                                                       settings["sampled"]), -1))
    result = {}
    for mode in ("raw", "sampled"):
        move_dist = dist("move", mode)
        if split == at:  # ending and the next move are alternatives of the same token
            carry, end = 1.0, float(move_dist[end_token])
        else:
            choice = dist("choice", mode)
            carry, end = float(choice[continue_token]), float(choice[end_token])
        probs = {m: carry * float(move_dist[t]) for m, t in tokens.items()}
        result[mode] = {"moves": probs, "end": end, "other": max(0.0, 1.0 - sum(probs.values()) - end)}
    greedy = lambda name: int(processed(logits[rows[name]], ids[0, :rows[name] + 1], settings["greedy"]).argmax())
    if split != at and greedy("choice") != continue_token:
        result["argmax"] = "end" if greedy("choice") == end_token else "other"
    else:
        best = greedy("move")
        result["argmax"] = next((m for m, t in tokens.items() if t == best), "end" if best == end_token else "other")
    return result


def describe(point, probs, taken):
    """One decision point's record: outcomes, probabilities, rank of the best optimal move."""
    raw, sampled = probs["raw"], probs["sampled"]
    optimal = [m for m, o in point["outcomes"].items() if o == "optimal"]
    order = sorted(MOVES, key=lambda m: -raw["moves"][m])
    return point | {
        "taken": taken, "taken_outcome": point["outcomes"].get(taken, "end"),
        "p_moves": {m: round(v, 6) for m, v in raw["moves"].items()}, "p_end": raw["end"], "p_other": raw["other"],
        "p_optimal": sum(raw["moves"][m] for m in optimal),
        "p_optimal_sampled": sum(sampled["moves"][m] for m in optimal),
        "p_moves_sampled": {m: round(v, 6) for m, v in sampled["moves"].items()},
        "rank_best_optimal": min(order.index(m) for m in optimal) + 1 if optimal else None,
        "argmax": probs["argmax"], "argmax_matches_taken": probs["argmax"] == taken}


# Summary ---------------------------------------------------------------------------

def share(values):
    return sum(values) / len(values) if values else None


def group(points):
    p = [x["p_optimal"] for x in points]
    return {"points": len(points), "within_reach": share([v >= REACH for v in p]),
            "median_p_optimal": statistics.median(p) if p else None,
            "zero_under_sampling": share([x["p_optimal_sampled"] == 0 for x in points]),
            "rank_best_optimal": {str(r): sum(x["rank_best_optimal"] == r for x in points) for r in (1, 2, 3, 4)}}


def summarise(points, released):
    unseen = [x for x in points if x["split"] != "train"]
    primary = [x for x in unseen if x["kind"] == "forced" and x["taken_outcome"] != "optimal"]
    agreement = share([x["argmax_matches_taken"] for x in unseen])
    memorised = statistics.median([x["p_taken"] for x in released]) if released else None
    valid = agreement is not None and agreement >= AGREEMENT and memorised is not None and memorised >= MEMORISED
    reach = group(primary)["within_reach"]
    reading = ("withheld: the measurement failed a validity check" if not valid or reach is None else
               "tractable" if reach >= TRACTABLE else "hard" if reach < HARD else "partial")
    by = lambda rows, key: {str(v): group([x for x in rows if x[key] == v]) for v in sorted({x[key] for x in rows})}
    deviations = [x for x in unseen if x["kind"] == "deviation"]
    stopped = [x for x in unseen if x["kind"] == "stopped"]
    return {
        "validity": {"argmax_matches_greedy": agreement, "training_released_move_median": memorised,
                     "training_maps": len(released), "valid": valid,
                     "canonical_test_completions": share([x["canonical"] for x in unseen if x.get("canonical") is not None])},
        "primary": group(primary) | {"by_split": by(primary, "split"), "by_size": by(primary, "grid_size")},
        "reading": reading,
        "kinds": {split: {kind: sum(x["kind"] == kind for x in unseen if x["split"] == split)
                          for kind in ("forced", "deviation", "stopped", "chosen")}
                  for split in sorted({x["split"] for x in unseen})},
        "forced_points_taken_optimal": group([x for x in unseen if x["kind"] == "forced"
                                              and x["taken_outcome"] == "optimal"]),
        "deviation": group(deviations) | {"detour_maps": group([x for x in deviations if x["detour"]]),
                                          "no_detour_maps": group([x for x in deviations if not x["detour"]]),
                                          "taken_into_hole": share([x["taken_outcome"] == "hole" for x in deviations])},
        "stopped": {"points": len(stopped), "median_p_end": statistics.median([x["p_end"] for x in stopped])
                    if stopped else None, "median_p_optimal": statistics.median([x["p_optimal"] for x in stopped])
                    if stopped else None},
        "chosen": group([x for x in unseen if x["kind"] == "chosen"])}


# Run -------------------------------------------------------------------------------

def read_routes(path, split):
    """row_index -> (route, recorded completion or None) from a planning-probe routes.jsonl
    (rows of `split`) or an exploration-probe sweep.jsonl (greedy_route)."""
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "greedy_route" in row:
            result[row["row_index"]] = (row["greedy_route"], None)
        elif row.get("split") == split:
            result[row["row_index"]] = (row["route"], row.get("completion"))
    return result


def detour_probe(model, processor, data_root, output, test_routes, fresh_routes, fresh_split, generation,
                 run_manifest, seed):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    json_write(output / "manifest.json", run_manifest | {
        "test_routes": str(test_routes), "fresh_routes": str(fresh_routes), "fresh_split": fresh_split,
        "generation": generation, "decoding": decoding_settings(model, generation), "reach": REACH, "thresholds": {"tractable": TRACTABLE, "hard": HARD},
        "validity": {"agreement": AGREEMENT, "memorised": MEMORISED}, "seed": seed})
    seed_all(seed)
    model.eval()
    tokenizer = processor.tokenizer
    settings = decoding_settings(model, generation)
    began, points, released = time.time(), [], []
    sets = [("test", "test", read_routes(test_routes, "test"))]
    if fresh_split:
        sets.append((fresh_split, fresh_split, read_routes(fresh_routes, fresh_split)))
    with (output / "points.jsonl").open("w", encoding="utf-8") as log:
        for name, split, routes in sets:
            rows = records(data_root, split)
            for index, (route, completion) in sorted(routes.items()):
                row = rows[index]
                point = decision_point(row["map_desc"], route)
                if point is None:
                    continue
                prompt = to_device(encode(processor, data_root, split, row, training=False), model.device)
                prefix = route[:point["k"] - 1]
                probs = move_probabilities(model, tokenizer, prompt, prefix, settings)
                taken = route[point["k"] - 1] if point["k"] <= len(route) else "end"
                canonical = None
                if completion is not None:
                    recorded = tokenizer(completion, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                    canonical = torch.equal(recorded, answer_ids(tokenizer, route))
                record = {"split": name, "row_index": index, "grid_size": len(row["map_desc"]),
                          "detour": is_detour(row["map_desc"]), "route": route, "canonical": canonical}
                record |= describe(point, probs, taken)
                points.append(record)
                log.write(json.dumps(record) + "\n"); log.flush()
        # Validity: released answers on the training detour maps, memorised by Stage 2.
        for index, row in enumerate(records(data_root, "train")):
            if not is_detour(row["map_desc"]):
                continue
            route = moves(extract_boxed_content(row["text_output"]))
            point = decision_point(row["map_desc"], route)
            if point is None or point["k"] > len(route):
                continue
            prompt = to_device(encode(processor, data_root, "train", row, training=False), model.device)
            probs = move_probabilities(model, tokenizer, prompt, route[:point["k"] - 1], settings)
            taken = route[point["k"] - 1]
            record = {"split": "train", "row_index": index, "grid_size": len(row["map_desc"]), "detour": True,
                      "route": route, "canonical": None} | describe(point, probs, taken)
            record["p_taken"] = probs["raw"]["moves"][taken]
            released.append(record)
            log.write(json.dumps(record) + "\n"); log.flush()
    summary = summarise(points, released) | {
        "seconds": time.time() - began,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    json_write(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("validity", "primary", "reading")}, indent=1), flush=True)
    return summary
