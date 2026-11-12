"""Planning probe (docs/PLANNING_PROBE_SPEC.md): latent headroom and state tracking on a
final LIVR checkpoint, in one model load.

1. routes: greedy answers on the test maps and on a fresh replication set, with the
   hidden state that predicts each move; the same states for the training maps with
   their released answers teacher-forced (features.pt);
2. state probe: ridge probes from those states to the agent's cell and to whether each
   neighbour is a hole, fitted on training maps and applied to the model's own routes;
3. headroom: for each failed test map, latent adjustments in the structured basis,
   bounded in norm, optimised toward a correct route and toward a matched wrong one.

The reading rules are fixed in the specification and applied here. The state probe
can be recomputed on CPU from a run directory: python -m spara.planning_probe RUN_DIR
"""
import json
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import torch

from .data import records
from .evaluation import ReferenceEvaluator
from .latent_battery import mcnemar, moves, probe_layers
from .latent_rl import LatentAdjuster
from .runtime import json_write, seed_all
from .serialization import encode
from .text_rl import repeat_prompt
from .training import amp, to_device

GREEDY = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
WORDS = {"U": "UP", "D": "DOWN", "L": "LEFT", "R": "RIGHT"}
CLASSES = 8  # one-hot row and column classes for the position probe
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
HELD_OUT = 0.2


# Routes and steps ---------------------------------------------------------------

def move_tokens(tokenizer, completion):
    """(letter, token index) for every move letter the reference parser reads from the
    boxed answer of a completion, located by the decoded characters of each token."""
    ids = [int(t) for t in completion]
    ends, text = [], ""
    for i in range(len(ids)):
        prefix = tokenizer.decode(ids[:i + 1], skip_special_tokens=False)
        if not prefix.startswith(text):
            raise ValueError("Decoded prefixes are not nested; token spans are undefined")
        text = prefix
        ends.append(len(text))
    start = text.rfind("\\boxed{")
    if start < 0:
        return []
    start += len("\\boxed{")
    depth, end = 0, None
    for j in range(start, len(text)):  # brace matching as in extract_boxed_content
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == -1:
            end = j
            break
    if end is None:
        return []
    return [(text[j].upper(), next(i for i, e in enumerate(ends) if e > j))
            for j in range(start, end) if text[j].upper() in MOVES]


def walk(grid, route):
    """Per move, the cell before it and whether it enters a hole (the walk stops there),
    with the reference evaluator's rules; and the outcome."""
    n = len(grid)
    r, c = next((r, c) for r in range(n) for c in range(n) if grid[r][c] == 1)
    steps = []
    for letter in route:
        dr, dc = MOVES[letter]
        inside = 0 <= r + dr < n and 0 <= c + dc < n
        steps.append({"cell": [r, c], "move": letter, "fatal": inside and grid[r + dr][c + dc] == -1})
        if inside:
            r, c = r + dr, c + dc
            if grid[r][c] == -1:
                return steps, "hole"
    return steps, "goal" if grid[r][c] == 2 else "short"


def neighbours(grid, cell):
    """For U, D, L, R: 1.0 if that neighbour is a hole, 0.0 if safe, None if off the grid."""
    n, (r, c) = len(grid), cell
    return [None if not (0 <= r + dr < n and 0 <= c + dc < n) else float(grid[r + dr][c + dc] == -1)
            for dr, dc in MOVES.values()]


def answer_ids(tokenizer, route):
    """A route in the model's answer format, as token IDs."""
    text = "\\boxed{" + ", ".join(WORDS[m] for m in route) + "}<|im_end|>"
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


@torch.no_grad()
def step_states(model, prompt, completion, tokens, layers):
    """Hidden states (steps, layers, d) at the positions that predict each token in
    `tokens` (completion indices); hidden_states[l + 1] is decoder layer l's output."""
    ids = torch.cat([prompt["input_ids"], completion[None].to(prompt["input_ids"].device)], -1)
    with amp(model.device):
        hidden = model(**prompt | {"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                       output_hidden_states=True, use_cache=False).hidden_states
    positions = torch.tensor([prompt["input_ids"].shape[-1] + t - 1 for t in tokens], device=ids.device)
    return torch.stack([hidden[layer + 1][0, positions] for layer in layers], 1).float().cpu()


def collect(model, processor, data_root, split, rows, layers, evaluator, generate, max_new_tokens, log):
    """Per map: the route (greedy if `generate`, else the released answer), its outcome and
    steps up to the fatal one; per step, the states that predict its move."""
    maps, steps, states = [], [], []
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    for count, (index, row) in enumerate(rows):
        prompt = to_device(encode(processor, data_root, split, row, training=False), model.device)
        length = prompt["input_ids"].shape[-1]
        if generate:
            with torch.no_grad(), amp(model.device):
                generated = model.generate(**prompt, **GREEDY, max_new_tokens=max_new_tokens, use_cache=True)
            completion = generated[0, length:]
            ends = [i for i, t in enumerate(completion.tolist()) if t in eos]
            completion = completion[:ends[0] + 1] if ends else completion
        else:
            completion = processor.tokenizer(row["text_output"] + "<|im_end|>", add_special_tokens=False,
                                             return_tensors="pt")["input_ids"][0].to(model.device)
        text = processor.tokenizer.decode(torch.cat([prompt["input_ids"][0], completion]), skip_special_tokens=False)
        scored = evaluator.score(row, text)
        letters = move_tokens(processor.tokenizer, completion)
        route = "".join(m for m, _ in letters)
        if route != moves(scored["extracted_answer"]):
            raise ValueError(f"Token alignment disagrees with the reference parser on {split} row {index}")
        walked, status = walk(row["map_desc"], route)
        if (status == "goal") != bool(scored["success"]):
            raise ValueError(f"Walk disagrees with the evaluator on {split} row {index}")
        tokens = [t for _, t in letters[:len(walked)]]
        shared = {t for t in tokens if tokens.count(t) > 1}
        kept = [k for k, t in enumerate(tokens) if t not in shared]
        record = {"split": split, "row_index": index, "grid_size": len(row["map_desc"]), "route": route,
                  "success": status == "goal", "status": status, "dropped_steps": len(tokens) - len(kept),
                  "completion": processor.tokenizer.decode(completion, skip_special_tokens=False)}
        maps.append(record)
        log.write(json.dumps(record) + "\n"); log.flush()
        if kept:
            states.append(step_states(model, prompt, completion, [tokens[k] for k in kept], layers).half())
            steps += [{"map": len(maps) - 1, "row_index": index, "size": len(row["map_desc"]), "k": k + 1,
                       "solved": status == "goal", **walked[k],
                       "neighbours": neighbours(row["map_desc"], walked[k]["cell"])} for k in kept]
        if count % 100 == 0:
            print(f"routes {split} {count + 1}/{len(rows)}", flush=True)
    states = torch.cat(states) if states else torch.zeros(0, len(layers), model.config.hidden_size).half()
    return {"maps": maps, "steps": steps, "states": states}


# State probe --------------------------------------------------------------------

def ridge(train_x, train_y, test_x):
    """Ridge on standardised features via a thin SVD; strength by leave-one-out error
    (as in scripts/analyze_latent_battery.py)."""
    train_x, train_y, test_x = train_x.double(), train_y.double(), test_x.double()
    mean, std = train_x.mean(0), train_x.std(0).clamp(min=1e-6)
    x, z = (train_x - mean) / std, (test_x - mean) / std
    y_mean = train_y.mean(0)
    y = train_y - y_mean
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    s2, uy = s.square(), u.T @ y
    best = None
    for alpha in ALPHAS:
        lam = alpha * s2.sum() / len(x)
        shrink = s2 / (s2 + lam)
        leverage = (u.square() * shrink).sum(-1)
        loo = ((y - u @ (shrink[:, None] * uy)) / (1 - leverage).clamp(min=1e-9)[:, None]).square().mean()
        if best is None or loo < best[0]:
            best = (loo, lam)
    return z @ (vh.T @ ((s / (s2 + best[1]))[:, None] * uy)) + y_mean


def position_targets(steps):
    y = torch.zeros(len(steps), 2 * CLASSES)
    for i, step in enumerate(steps):
        y[i, step["cell"][0]] = y[i, CLASSES + step["cell"][1]] = 1
    return y


def decoded_cells(scores, steps):
    return [[int(scores[i, :s["size"]].argmax()), int(scores[i, CLASSES:CLASSES + s["size"]].argmax())]
            for i, s in enumerate(steps)]


def select_layer(train, layers, seed):
    """Held-out position accuracy per layer, with the probe fitted on the other maps."""
    maps = sorted({s["map"] for s in train["steps"]})
    held = set(random.Random(seed).sample(maps, round(HELD_OUT * len(maps))))
    fit = [i for i, s in enumerate(train["steps"]) if s["map"] not in held]
    test = [i for i, s in enumerate(train["steps"]) if s["map"] in held]
    y = position_targets([train["steps"][i] for i in fit])
    accuracy = {}
    for position, layer in enumerate(layers):
        x = train["states"][:, position].float()
        cells = decoded_cells(ridge(x[fit], y, x[test]), [train["steps"][i] for i in test])
        accuracy[str(layer + 1)] = sum(c == train["steps"][i]["cell"] for c, i in zip(cells, test)) / len(test)
    return accuracy


def hole_scores(train, x_train, x_test):
    """Per direction, a ridge fitted only on steps where that neighbour is on the grid and
    is not the next move; scores for every test step (None columns stay NaN-free)."""
    scores = []
    for j, letter in enumerate(MOVES):
        index = [i for i, s in enumerate(train["steps"]) if s["neighbours"][j] is not None and s["move"] != letter]
        y = torch.tensor([[train["steps"][i]["neighbours"][j]] for i in index])
        scores.append(ridge(x_train[index], y, x_test)[:, 0])
    return torch.stack(scores, 1)


def poisson_binomial_lower(probs, observed):
    """P(at most `observed` successes) for independent Bernoulli trials."""
    dist = [1.0]
    for p in probs:
        dist = [a * (1 - p) + b * p for a, b in zip(dist + [0.0], [0.0] + dist)]
    return sum(dist[:observed + 1])


def tracking_test(steps, correct):
    """S1: fatal steps with k >= 2 decoded correctly against the accuracy on solved steps of
    the same size and k (nearest k if absent); fatal first moves separately."""
    solved = defaultdict(list)
    for step, ok in zip(steps, correct):
        if step["solved"]:
            solved[step["size"], step["k"]].append(ok)
    fatal = [(s, ok) for s, ok in zip(steps, correct) if s["fatal"]]
    first = [ok for s, ok in fatal if s["k"] == 1]
    expected, observed, fallback, unmatched = [], 0, 0, 0
    for step, ok in [(s, ok) for s, ok in fatal if s["k"] >= 2]:
        ks = [k for size, k in solved if size == step["size"]]
        if not ks:
            unmatched += 1
            continue
        k = min(ks, key=lambda k: (abs(k - step["k"]), -k))
        fallback += k != step["k"]
        expected.append(sum(solved[step["size"], k]) / len(solved[step["size"], k]))
        observed += ok
    result = {"fatal_steps": len(fatal), "fatal_first_moves": len(first),
              "fatal_first_move_accuracy": sum(first) / len(first) if first else None,
              "tested": len(expected), "nearest_k_used": fallback, "unmatched": unmatched}
    if not expected:
        return result | {"reading": "no fatal steps with k >= 2"}
    accuracy, base = observed / len(expected), sum(expected) / len(expected)
    p = poisson_binomial_lower(expected, observed)
    gap = base - accuracy
    reading = ("tracking contributes" if p < 0.05 and gap >= 0.15 else
               "tracking intact" if gap < 0.10 else "inconclusive")
    return result | {"correct": observed, "accuracy": accuracy, "expected_accuracy": base, "gap": gap,
                     "p_one_sided": p, "reading": reading}


def safe_from(grid, cell, move):
    n, (r, c), (dr, dc) = len(grid), cell, MOVES[move]
    if 0 <= r + dr < n and 0 <= c + dc < n:
        r, c = r + dr, c + dc
    return grid[r][c] != -1


def secondary(steps, cells, holes, grids):
    """S2 (wrong decoded cell: would the fatal move be safe from it?) and S3 (hole-ahead)."""
    wrong = [(s, c) for s, c in zip(steps, cells) if s["fatal"] and s["k"] >= 2 and c != s["cell"]]
    from_decoded, chance = [], []
    for step, cell in wrong:
        grid = grids[step["map"]]
        n = len(grid)
        others = [[r, c] for r in range(n) for c in range(n) if grid[r][c] != -1 and [r, c] != step["cell"]]
        from_decoded.append(safe_from(grid, cell, step["move"]))
        chance.append(sum(safe_from(grid, o, step["move"]) for o in others) / len(others))
    letters = list(MOVES)
    fatal = [float(holes[i, letters.index(s["move"])]) for i, s in enumerate(steps) if s["fatal"]]
    avoided, taken, other_safe = [], [], []
    for i, step in enumerate(steps):
        if not step["solved"]:
            continue
        for j, letter in enumerate(letters):
            status = step["neighbours"][j]
            if status is None:
                continue
            if letter == step["move"]:
                taken.append(float(holes[i, j]))
            elif status == 1.0:
                avoided.append(float(holes[i, j]))
            else:
                other_safe.append(float(holes[i, j]))
    mean = lambda v: sum(v) / len(v) if v else None
    share = lambda v: sum(x > 0.5 for x in v) / len(v) if v else None
    return {"S2": {"fatal_steps_with_wrong_cell": len(wrong), "safe_from_decoded_cell": mean(from_decoded),
                   "safe_from_other_cells": mean(chance)},
            "S3": {"fatal_holes": len(fatal), "fatal_read_as_hole": share(fatal),
                   "avoided_holes": len(avoided), "avoided_read_as_hole": share(avoided),
                   "bias_check": {"taken_direction_mean_score": mean(taken),
                                  "other_safe_mean_score": mean(other_safe)}}}


def state_probe(features, grids, seed):
    """Layer selection on training maps, then S1 to S3 and the validity check on each
    evaluated split. `grids[split]` lists each map's layout in `features[split]["maps"]` order."""
    layers, train = features["layers"], features["train"]
    held_out = select_layer(train, layers, seed)
    best = max(held_out, key=held_out.get)
    position = [layer + 1 for layer in layers].index(int(best))
    x_train = train["states"][:, position].float()
    y = position_targets(train["steps"])
    report = {"held_out_position_accuracy": held_out, "layer": int(best), "splits": {}, "steps": []}
    for split in (s for s in ("test", "fresh") if s in features and features[s]["steps"]):
        data = features[split]
        x = data["states"][:, position].float()
        cells = decoded_cells(ridge(x_train, y, x), data["steps"])
        holes = hole_scores(train, x_train, x)
        correct = [c == s["cell"] for c, s in zip(cells, data["steps"])]
        solved = [ok for ok, s in zip(correct, data["steps"]) if s["solved"]]
        report["splits"][split] = {
            "maps": len(data["maps"]), "failed_maps": sum(not m["success"] for m in data["maps"]),
            "steps": len(data["steps"]), "dropped_steps": sum(m["dropped_steps"] for m in data["maps"]),
            "solved_step_accuracy": sum(solved) / len(solved) if solved else None,
            "S1": tracking_test(data["steps"], correct), **secondary(data["steps"], cells, holes, grids[split])}
        report["steps"] += [{"split": split, **{k: s[k] for k in ("row_index", "size", "k", "cell", "move", "fatal",
                                                                  "solved")},
                             "decoded_cell": c, "hole_scores": [round(float(v), 4) for v in h]}
                            for s, c, h in zip(data["steps"], cells, holes)]
    return report


# Headroom -----------------------------------------------------------------------

def distances(grid):
    """Steps from each cell to the goal through non-hole cells."""
    n = len(grid)
    goal = next((r, c) for r in range(n) for c in range(n) if grid[r][c] == 2)
    dist, queue = {goal: 0}, deque([goal])
    while queue:
        r, c = queue.popleft()
        for dr, dc in MOVES.values():
            cell = (r + dr, c + dc)
            if 0 <= cell[0] < n and 0 <= cell[1] < n and grid[cell[0]][cell[1]] != -1 and cell not in dist:
                dist[cell] = dist[r, c] + 1
                queue.append(cell)
    return dist


def shortest_routes(grid, limit):
    """Up to `limit` shortest successful routes in a fixed move order, and whether more exist."""
    dist = distances(grid)
    n = len(grid)
    start = next((r, c) for r in range(n) for c in range(n) if grid[r][c] == 1)
    routes, capped = [], False

    def extend(cell, route):
        nonlocal capped
        if len(routes) >= limit:
            capped = True  # every remaining branch leads to another shortest route
            return
        if dist[cell] == 0:
            routes.append(route)
            return
        for letter, (dr, dc) in MOVES.items():
            step = (cell[0] + dr, cell[1] + dc)
            if dist.get(step) == dist[cell] - 1:
                extend(step, route + letter)

    extend(start, "")
    return routes, capped


def control_routes(grid, target, own):
    """Failing routes that differ from `target` in exactly one move and are not `own`."""
    edits = {target[:i] + m + target[i + 1:] for i in range(len(target)) for m in MOVES if m != target[i]}
    return sorted(r for r in edits if r != own and walk(grid, r)[1] != "goal")


def teacher_batch(prompt, completions, pad_id):
    """One prompt followed by different completions, right-padded. Every row shares the
    prompt, so the latent positions found from row 0 hold for all rows."""
    length = max(len(c) for c in completions)
    ids = torch.full((len(completions), length), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, c in enumerate(completions):
        ids[i, :len(c)], mask[i, :len(c)] = c.cpu(), 1
    device = prompt["input_ids"].device
    base = repeat_prompt(prompt, len(completions))
    batch = base | {"input_ids": torch.cat([base["input_ids"], ids.to(device)], -1),
                    "attention_mask": torch.cat([base["attention_mask"], mask.to(device)], -1)}
    return batch, ids.to(device), mask.to(device).bool()


def completion_nll(model, batch, ids, mask, prompt_length):
    """Mean token NLL per row, and whether the teacher-forced argmax matches every token."""
    logits = model(**batch, use_cache=False).logits[:, prompt_length - 1:-1].float()
    nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), ids, reduction="none")
    return (nll * mask).sum(-1) / mask.sum(-1), ((logits.argmax(-1) == ids) | ~mask).all(-1)


@torch.no_grad()
def score_routes(model, prompt, tokenizer, routes, chunk=16):
    length, values = prompt["input_ids"].shape[-1], []
    for start in range(0, len(routes), chunk):
        batch, ids, mask = teacher_batch(prompt, [answer_ids(tokenizer, r) for r in routes[start:start + chunk]],
                                         tokenizer.pad_token_id or 0)
        with amp(model.device):
            values += completion_nll(model, batch, ids, mask, length)[0].tolist()
    return values


def optimise(model, adjuster, basis, prompt, completions, radii, steps, learning_rate, pad_id):
    """Coefficients a (rows, K, k), ||a|| <= radius per slot, minimising each row's
    completion NLL; a row keeps the coefficients of the first step at which its
    teacher-forced argmax matches every token."""
    device, length = prompt["input_ids"].device, prompt["input_ids"].shape[-1]
    batch, ids, mask = teacher_batch(prompt, completions, pad_id)
    directions, scales = basis["directions"].to(device), basis["scales"].to(device)
    radius = torch.tensor(radii, device=device, dtype=torch.float32)[:, None, None]
    a = torch.zeros(len(completions), *scales.shape, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([a], lr=learning_rate)
    kept, start_nll = [None] * len(completions), None
    for step in range(steps + 1):
        offsets = torch.einsum("sdk,sk,nsk->nsd", directions, scales, a)
        with adjuster.offset(offsets), amp(device):
            nll, matched = completion_nll(model, batch, ids, mask, length)
        start_nll = nll.detach().tolist() if start_nll is None else start_nll
        for i in range(len(kept)):
            if kept[i] is None and (matched[i] or step == steps):
                kept[i] = {"a": a[i].detach().clone(), "step": step if matched[i] else None, "nll": float(nll[i])}
        if all(k is not None for k in kept):
            break
        optimizer.zero_grad()
        a.grad, = torch.autograd.grad(nll.sum(), [a])
        optimizer.step()
        with torch.no_grad():
            a.mul_((radius / a.norm(dim=-1, keepdim=True).clamp(min=1e-12)).clamp(max=1.0))
    return kept, start_nll


def headroom(model, processor, data_root, failed, basis, adjuster, evaluator, radii, steps, learning_rate,
             max_candidates, max_new_tokens, log):
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id or 0
    for p in model.parameters():
        p.requires_grad_(False)
    results = []
    for count, (index, row, own) in enumerate(failed):
        began = time.time()
        grid = row["map_desc"]
        prompt = to_device(encode(processor, data_root, "test", row, training=False), model.device)
        candidates, capped = shortest_routes(grid, max_candidates)
        nll = score_routes(model, prompt, tokenizer, candidates)
        target, target_nll = min(zip(candidates, nll), key=lambda item: item[1])
        controls = control_routes(grid, target, own)
        control = control_nll = None
        if controls:
            control, control_nll = min(zip(controls, score_routes(model, prompt, tokenizer, controls)),
                                       key=lambda item: abs(item[1] - target_nll))
        plan = [(r, "target", target) for r in radii] + ([(r, "control", control) for r in radii] if control else [])
        kept, start_nll = optimise(model, adjuster, basis, prompt, [answer_ids(tokenizer, route) for _, _, route in plan],
                                   [r for r, _, _ in plan], steps, learning_rate, pad_id)
        coefficients = torch.stack([k["a"] for k in kept])
        offsets = torch.einsum("sdk,sk,nsk->nsd", basis["directions"].to(model.device),
                               basis["scales"].to(model.device), coefficients)
        length = prompt["input_ids"].shape[-1]
        with torch.no_grad(), adjuster.offset(offsets), amp(model.device):
            generated = model.generate(**repeat_prompt(prompt, len(plan)), **GREEDY, max_new_tokens=max_new_tokens,
                                       use_cache=True)
        rows = []
        for (radius, kind, route), keep, before, sequence in zip(plan, kept, start_nll, generated):
            scored = evaluator.score(row, tokenizer.decode(torch.cat([prompt["input_ids"][0], sequence[length:]]),
                                                           skip_special_tokens=False))
            answer = moves(scored["extracted_answer"])
            rows.append({"radius": radius, "kind": kind, "matched_at_step": keep["step"], "nll_start": before,
                         "nll_end": keep["nll"], "norms": keep["a"].norm(dim=-1).tolist(), "route": answer,
                         "success": bool(scored["success"]), "reproduced": answer == route})
        record = {"row_index": index, "grid_size": len(grid), "own_route": own, "target": target,
                  "target_nll": target_nll, "candidates": len(candidates), "candidates_capped": capped,
                  "control": control, "control_nll": control_nll, "controls_available": len(controls),
                  "seconds": time.time() - began, "rows": rows}
        results.append(record)
        log.write(json.dumps(record) + "\n"); log.flush()
        if count % 10 == 0:
            print(f"headroom {count + 1}/{len(failed)} {record['seconds']:.1f}s", flush=True)
    return results


def summarise_headroom(results, radii, primary=4.0):
    per_radius = {}
    for radius in radii:
        target = [next(r for r in m["rows"] if r["radius"] == radius and r["kind"] == "target") for m in results]
        control = [next((r for r in m["rows"] if r["radius"] == radius and r["kind"] == "control"), None)
                   for m in results]
        paired = [(t, c) for t, c in zip(target, control) if c is not None]
        per_radius[str(radius)] = {
            "maps": len(target), "fixed": sum(t["success"] for t in target) / len(target) if target else None,
            "target_reproduced": sum(t["reproduced"] for t in target) / len(target) if target else None,
            "maps_with_control": len(paired),
            "target_reproduced_paired": sum(t["reproduced"] for t, _ in paired) / len(paired) if paired else None,
            "control_reproduced": sum(c["reproduced"] for _, c in paired) / len(paired) if paired else None,
            "mean_target_nll_start": sum(t["nll_start"] for t in target) / len(target) if target else None,
            "mean_target_nll_end": sum(t["nll_end"] for t in target) / len(target) if target else None}
    summary = {"maps": len(results), "per_radius": per_radius}
    fixed_primary = per_radius.get(str(primary), {}).get("fixed")
    fixed_largest = per_radius[str(max(radii))]["fixed"] if results else None
    summary["headroom"] = (None if fixed_primary is None else "present" if fixed_primary >= 0.5 else
                           "absent" if fixed_largest is not None and fixed_largest < 0.2 else "partial")
    pairs = [(next(r for r in m["rows"] if r["radius"] == primary and r["kind"] == "target"),
              next(r for r in m["rows"] if r["radius"] == primary and r["kind"] == "control"))
             for m in results if m["control"] and any(r["radius"] == primary for r in m["rows"])]
    if pairs:
        b = sum(t["reproduced"] and not c["reproduced"] for t, c in pairs)
        c = sum(c["reproduced"] and not t["reproduced"] for t, c in pairs)
        target_rate = sum(t["reproduced"] for t, _ in pairs) / len(pairs)
        control_rate = sum(c["reproduced"] for _, c in pairs) / len(pairs)
        p = mcnemar(b, c)
        summary["specificity"] = {
            "radius": primary, "pairs": len(pairs), "target_only": b, "control_only": c, "mcnemar_p": p,
            "reading": ("specific to correct routes" if target_rate - control_rate >= 0.2 and p < 0.05 else
                        "generic steering" if control_rate >= target_rate else "inconclusive")}
    return summary


# Run ----------------------------------------------------------------------------

def planning_probe(model, processor, data_root, output, basis_path, layer, run_manifest, seed, fresh_split=None,
                   fresh_sizes=(5, 6), radii=(1.0, 2.0, 4.0, 8.0), steps=40, learning_rate=0.25,
                   max_candidates=64, max_new_tokens=128):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    # Without radii the headroom test is skipped (the state probe alone, e.g. on a new checkpoint).
    basis = torch.load(basis_path) if radii else None
    layers = probe_layers(model.config.num_hidden_layers)
    json_write(output / "manifest.json", run_manifest | {
        "basis": str(basis_path) if radii else None, "basis_k": basis["k"] if radii else None, "layer": layer,
        "probe_layers": [l + 1 for l in layers],
        "fresh_split": fresh_split, "fresh_sizes": list(fresh_sizes), "radii": list(radii), "steps": steps,
        "learning_rate": learning_rate, "max_candidates": max_candidates, "max_new_tokens": max_new_tokens,
        "seed": seed})
    seed_all(seed)
    model.eval()
    evaluator = ReferenceEvaluator(data_root)
    seconds = {}
    began = time.time()
    test_rows = records(data_root, "test")
    sets = {"test": ("test", list(enumerate(test_rows)), True)}
    if fresh_split:
        fresh = [(i, r) for i, r in enumerate(records(data_root, fresh_split)) if len(r["map_desc"]) in fresh_sizes]
        sets["fresh"] = (fresh_split, fresh, True)
    sets["train"] = ("train", list(enumerate(records(data_root, "train"))), False)
    features, grids = {"layers": layers}, {}
    with (output / "routes.jsonl").open("w", encoding="utf-8") as log:
        for name, (split, rows, generate) in sets.items():
            features[name] = collect(model, processor, data_root, split, rows, layers, evaluator, generate,
                                     max_new_tokens, log)
            lookup = dict(rows)
            grids[name] = [lookup[m["row_index"]]["map_desc"] for m in features[name]["maps"]]
            seconds[f"routes_{name}"] = time.time() - began
    torch.save(features, output / "features.pt")
    report = state_probe(features, grids, seed)
    (output / "steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in report.pop("steps")), encoding="utf-8")
    summary = {"state_probe": report}
    json_write(output / "summary.json", summary)
    print(json.dumps(report, indent=1), flush=True)

    if not radii:
        seconds["total"] = time.time() - began
        summary |= {"headroom": None, "seconds": seconds,
                    "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
        json_write(output / "summary.json", summary)
        return summary
    failed = [(m["row_index"], test_rows[m["row_index"]], m["route"]) for m in features["test"]["maps"]
              if not m["success"]]
    adjuster = LatentAdjuster(model, layer)
    with (output / "headroom.jsonl").open("w", encoding="utf-8") as log:
        results = headroom(model, processor, data_root, failed, basis, adjuster, evaluator, list(radii), steps,
                           learning_rate, max_candidates, max_new_tokens, log)
    adjuster.remove()
    seconds["total"] = time.time() - began
    summary |= {"headroom": summarise_headroom(results, list(radii)), "seconds": seconds,
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    json_write(output / "summary.json", summary)
    print(json.dumps(summary["headroom"], indent=1), flush=True)
    return summary


def main():
    """Recompute the state probe on CPU from a run directory (features.pt, routes.jsonl)."""
    run = Path(sys.argv[1])
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    features = torch.load(run / "features.pt")
    data_root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/vsp")
    splits = {"test": "test", "train": "train", "fresh": manifest.get("fresh_split")}
    grids = {}
    for name in ("test", "fresh", "train"):
        if name in features:
            lookup = records(data_root, splits[name])
            grids[name] = [lookup[m["row_index"]]["map_desc"] for m in features[name]["maps"]]
    report = state_probe(features, grids, manifest["seed"])
    report.pop("steps")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
