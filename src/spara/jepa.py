"""Step 2 of docs/DETOUR_LEARNING_SPEC.md (docs/JEPA_JUDGE_SPEC.md): a JEPA transition
model in LIVR's latent space, used as guided search's judge.

A state is LIVR's hidden state (frozen) at the position where the next move is chosen,
after a route prefix. The predictor maps (state, move) to the next state's embedding;
the outcome head reads the predicted embedding and says alive, dead or done. Movement
follows the rules the prompt states (a move off the grid leaves the agent in place);
what lies in each cell is what the model must judge.
"""
import json
import math
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from .data import layout_key, records
from .detour_probe import alignment, find
from .fresh_maps import label_route
from .latent_battery import mcnemar
from .planning_probe import MOVES, answer_ids, move_tokens, teacher_batch
from .route_search import FLOOR, FrozenLakeEnvironment, problems_of, probe_agreement
from .runtime import json_write, seed_all
from .search import ALIVE, DEAD, DONE, STOP, guided_search, verdict_search
from .serialization import encode
from .training import amp, to_device

START, GOAL, HOLE = 1, 2, -1
OUTCOMES = (ALIVE, DEAD, DONE)
ACTIONS = tuple(MOVES)  # U, D, L, R
LAYERS = (12, 20, 28, 35)  # candidate decoder layers (hidden_states[layer + 1])
WALKS, FATAL = 4, 0.2  # walks per map: the labelled route and random walks; chance a walk takes a hole move
ALIGNMENT = 0.99  # validity: cosine between batched and single-pass states
READY = 0.9


# Walks and labels ---------------------------------------------------------------------

def in_grid(grid, cell, move):
    r, c = cell[0] + MOVES[move][0], cell[1] + MOVES[move][1]
    return (r, c) if 0 <= r < len(grid) and 0 <= c < len(grid) else None


def cell_outcome(grid, cell):
    return DEAD if grid[cell[0]][cell[1]] == HOLE else DONE if grid[cell[0]][cell[1]] == GOAL else ALIVE


def walks(grid, rng, count=WALKS, fatal=FATAL):
    """The labelled shortest route and `count - 1` random walks over moves that stay on
    the grid. A walk steps into a hole with probability `fatal` when one is next to it
    (or when nothing else is), and ends at a hole, at the goal or after 2n moves."""
    routes = [label_route(grid)]
    start = find(grid, START)
    for _ in range(count - 1):
        cell, route = start, ""
        for _ in range(2 * len(grid)):
            moves = [m for m in ACTIONS if in_grid(grid, cell, m)]
            holes = [m for m in moves if cell_outcome(grid, in_grid(grid, cell, m)) == DEAD]
            safe = [m for m in moves if m not in holes]
            move = rng.choice(holes) if holes and (not safe or rng.random() < fatal) else rng.choice(safe)
            route += move
            cell = in_grid(grid, cell, move)
            if cell_outcome(grid, cell) != ALIVE:
                break
        routes.append(route)
    return routes


def route_cells(grid, route):
    """The cell before each move and after the last one."""
    cells = [find(grid, START)]
    for move in route:
        cells.append(in_grid(grid, cells[-1], move) or cells[-1])
    return cells


# LIVR's states --------------------------------------------------------------------------

@torch.no_grad()
def route_states(model, tokenizer, prompt, routes, layers):
    """Per route, (moves + 1, layers, d): the states that choose each move and the state
    after the last move (read where a further move would be chosen). One batched pass."""
    completions = [answer_ids(tokenizer, r + "U") for r in routes]  # the extra move is never attended to
    batch, ids, _ = teacher_batch(prompt, completions, tokenizer.pad_token_id or 0)
    length = prompt["input_ids"].shape[-1]
    with amp(model.device):
        hidden = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
    states = []
    for i, (route, completion) in enumerate(zip(routes, completions)):
        tokens = [t for _, t in move_tokens(tokenizer, completion)]
        if len(tokens) != len(route) + 1:
            raise ValueError("Move tokens do not align with the route")
        positions = torch.tensor([length + t - 1 for t in tokens], device=ids.device)
        states.append(torch.stack([hidden[layer + 1][i, positions] for layer in layers], 1).float().cpu())
    return states


def collect(model, processor, data_root, split, rows, layers, seed, log=None):
    """States and labels along each map's walks. Returns states (N, layers, d) in fp16 and
    one record per state: map, size, walk, step, cell, whether the agent is alive there,
    the next move taken, and the outcome of every move that stays on the grid."""
    rng = random.Random(seed)
    states, items = [], []
    for count, (index, row) in enumerate(rows):
        grid = row["map_desc"]
        prompt = to_device(encode(processor, data_root, split, row, training=False), model.device)
        routes = walks(grid, rng)
        for w, (route, tensor) in enumerate(zip(routes, route_states(model, processor.tokenizer, prompt, routes,
                                                                      layers))):
            cells = route_cells(grid, route)
            for k, cell in enumerate(cells):
                alive = cell_outcome(grid, cell) == ALIVE
                items.append({"split": split, "row_index": index, "size": len(grid), "walk": w, "k": k,
                              "cell": list(cell), "at": cell_outcome(grid, cell), "alive": alive, "route": route,
                              "next": route[k] if k < len(route) else None,
                              "outcomes": {m: cell_outcome(grid, in_grid(grid, cell, m))
                                           for m in ACTIONS if in_grid(grid, cell, m)} if alive else {}})
            states.append(tensor.half())
        if log and count % 200 == 0:
            print(f"states {split} {count + 1}/{len(rows)}", flush=True)
    if not states:
        return torch.zeros(0, len(layers), model.config.hidden_size, dtype=torch.float16), items
    return torch.cat(states), items


def pairs(items):
    """(state, move, outcome class) for every judged move, and (state, move, next state)
    for every move taken."""
    judged = [(i, ACTIONS.index(m), OUTCOMES.index(o)) for i, it in enumerate(items) for m, o in it["outcomes"].items()]
    taken = [(i, ACTIONS.index(it["next"]), i + 1) for i, it in enumerate(items) if it["next"]]
    return torch.tensor(judged, dtype=torch.long).reshape(-1, 3), torch.tensor(taken, dtype=torch.long).reshape(-1, 3)


# Models --------------------------------------------------------------------------------

class Jepa(torch.nn.Module):
    """Predictor of the next state's embedding from (state, move), and an outcome head on
    the predicted embedding. Embeddings are the frozen states, layer-normalised."""

    def __init__(self, d, width=1024, action_dim=64):
        super().__init__()
        self.d = d
        self.norm = torch.nn.LayerNorm(d, elementwise_affine=False)
        self.action = torch.nn.Embedding(len(ACTIONS), action_dim)
        self.predictor = torch.nn.Sequential(torch.nn.Linear(d + action_dim, width), torch.nn.GELU(),
                                             torch.nn.Linear(width, width), torch.nn.GELU(), torch.nn.Linear(width, d))
        self.head = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.GELU(), torch.nn.Linear(256, len(OUTCOMES)))

    def predict(self, z, a):
        z = self.norm(z)
        return z + self.predictor(torch.cat([z, self.action(a)], -1))

    def judge(self, z, a):
        """Outcome logits (alive, dead, done) of move `a` from state `z`."""
        return self.head(self.norm(self.predict(z, a)))


class Probe(torch.nn.Module):
    """The same capacity with no next-state prediction: outcome logits straight from
    (state, move). The comparison that says whether predicting the next state helps."""

    def __init__(self, d, width=1024, action_dim=64):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d, elementwise_affine=False)
        self.action = torch.nn.Embedding(len(ACTIONS), action_dim)
        self.net = torch.nn.Sequential(torch.nn.Linear(d + action_dim, width), torch.nn.GELU(),
                                       torch.nn.Linear(width, width), torch.nn.GELU(),
                                       torch.nn.Linear(width, len(OUTCOMES)))

    def judge(self, z, a):
        return self.net(torch.cat([self.norm(z), self.action(a)], -1))


class Oracle(torch.nn.Module):
    """Outcome of the move just taken, from LIVR's true state after it: what the frozen
    states hold once the move is written. An upper reference, not a judge (it needs the
    pass the judge is meant to save)."""

    def __init__(self, d):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d, elementwise_affine=False)
        self.net = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.GELU(), torch.nn.Linear(256, len(OUTCOMES)))

    def forward(self, z):
        return self.net(self.norm(z))


def batches(n, size, generator):
    order = torch.randperm(n, generator=generator)
    return [order[i:i + size] for i in range(0, n, size)]


def train(kind, x, judged, taken, labels_next, valid, settings, device, seed):
    """Train a jepa, probe or oracle on states `x` (N, d) with early stopping on the
    validation loss; `valid` holds the validation set's (x, judged, taken, labels_next)."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    d = x.shape[1]
    model = {"jepa": Jepa, "probe": Probe, "oracle": Oracle}[kind](d).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    x, vx = x.to(device), valid[0].to(device)

    def loss_of(model, x, judged, taken, labels_next, rows=None, steps=None):
        parts = {}
        t = (taken if steps is None else taken[steps]).to(device)
        if kind == "oracle":
            y = (labels_next if steps is None else labels_next[steps]).to(device)
            parts["outcome"] = torch.nn.functional.cross_entropy(model(x[t[:, 2]]), y)
            return parts
        j = (judged if rows is None else judged[rows]).to(device)
        parts["outcome"] = torch.nn.functional.cross_entropy(model.judge(x[j[:, 0]], j[:, 1]), j[:, 2])
        if kind == "jepa":
            predicted = model.predict(x[t[:, 0]], t[:, 1])
            parts["prediction"] = (1 - torch.nn.functional.cosine_similarity(predicted, model.norm(x[t[:, 2]]), -1)).mean()
        return parts

    def validate():
        model.eval()
        with torch.no_grad():
            parts = loss_of(model, vx, valid[1], valid[2], valid[3])
        model.train()
        return {k: float(v) for k, v in parts.items()}

    best, best_state, stale, history = math.inf, None, 0, []
    for epoch in range(settings["epochs"]):
        if kind == "oracle":
            for steps in batches(len(taken), settings["batch"], generator):
                loss = sum(loss_of(model, x, judged, taken, labels_next, steps=steps).values())
                optimiser.zero_grad(); loss.backward(); optimiser.step()
        else:
            step_batches = batches(len(taken), settings["batch"], generator)
            for b, rows in enumerate(batches(len(judged), settings["batch"], generator)):
                loss = sum(loss_of(model, x, judged, taken, labels_next, rows=rows,
                                   steps=step_batches[b % len(step_batches)]).values())
                optimiser.zero_grad(); loss.backward(); optimiser.step()
        val = validate()
        history.append({"epoch": epoch, **val})
        total = sum(val.values())
        if total < best - 1e-4:
            best, stale = total, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= settings["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def next_labels(items, taken):
    """For the oracle: the outcome class of each taken move, where it led."""
    return torch.tensor([OUTCOMES.index(items[j]["at"]) for _, _, j in taken.tolist()], dtype=torch.long)


# Judging and its error rates -------------------------------------------------------------

def rates(predicted, truth):
    """Misses (dead judged alive or done), false alarms (alive or done judged dead), and
    done errors, from outcome class indices."""
    dead, pred_dead = truth == OUTCOMES.index(DEAD), predicted == OUTCOMES.index(DEAD)
    done, pred_done = truth == OUTCOMES.index(DONE), predicted == OUTCOMES.index(DONE)
    count = lambda mask: int(mask.sum())
    return {"pairs": len(truth), "dead": count(dead), "not_dead": count(~dead),
            "miss_rate": count(dead & ~pred_dead) / max(1, count(dead)),
            "false_alarm_rate": count(~dead & pred_dead) / max(1, count(~dead)),
            "done_missed": count(done & ~pred_done), "done_false": count(~done & pred_done), "done": count(done),
            "accuracy": count(predicted == truth) / max(1, len(truth))}


@torch.no_grad()
def judge_all(model, x, judged, device):
    return model.judge(x[judged[:, 0]].to(device), judged[:, 1].to(device)).argmax(-1).cpu()


@torch.no_grad()
def imagination(model, x, items, device, horizons=(1, 2, 3)):
    """Judge the move taken j steps ahead from a state imagined by rolling the predictor
    along the walk's own moves, against judging it from LIVR's true state there."""
    out = {}
    for j in horizons:
        imagined, true_state, truth = [], [], []
        for i, it in enumerate(items):
            if not it["alive"]:
                continue
            ahead = i + j
            if ahead >= len(items) or items[ahead]["walk"] != it["walk"] or items[ahead]["row_index"] != it["row_index"] \
                    or items[ahead]["split"] != it["split"] or not items[ahead]["alive"] or not items[ahead]["next"] \
                    or not all(items[i + s]["alive"] for s in range(j)):
                continue
            z = x[i:i + 1].to(device)
            for s in range(j):
                z = model.predict(z, torch.tensor([ACTIONS.index(items[i + s]["next"])], device=device))
            move = items[ahead]["next"]
            a = torch.tensor([ACTIONS.index(move)], device=device)
            imagined.append(int(model.judge(z, a).argmax()))
            true_state.append(int(model.judge(x[ahead:ahead + 1].to(device), a).argmax()))
            truth.append(OUTCOMES.index(items[ahead]["outcomes"][move]))
        truth = torch.tensor(truth, dtype=torch.long)
        out[str(j)] = {"imagined": rates(torch.tensor(imagined, dtype=torch.long), truth),
                       "true_state": rates(torch.tensor(true_state, dtype=torch.long), truth)}
    return out


# Search with LIVR as policy and the JEPA as judge ---------------------------------------

@torch.no_grad()
def decide(model, tokenizer, prompt, prefix, layer):
    """One teacher-forced pass after `prefix`: LIVR's raw move and ending probabilities (as
    move_probabilities computes them) and its state where the next move is chosen."""
    context, at, split, tokens, end_token, continue_token = alignment(tokenizer, prefix)
    length = prompt["input_ids"].shape[-1]
    ids = torch.cat([prompt["input_ids"], context[None].to(prompt["input_ids"].device)], -1)
    with amp(model.device):
        out = model(**prompt | {"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                    output_hidden_states=True, use_cache=False)
    logits = out.logits[0].float()
    move_dist = torch.softmax(logits[length + at - 1], -1)
    if split == at:
        carry, end = 1.0, float(move_dist[end_token])
    else:
        choice = torch.softmax(logits[length + split - 1], -1)
        carry, end = float(choice[continue_token]), float(choice[end_token])
    probs = {m: carry * float(move_dist[t]) for m, t in tokens.items()} | {"end": end}
    return probs, out.hidden_states[layer + 1][0, length + at - 1].float().cpu()


class Decisions:
    """LIVR's pass for each (problem, prefix), cached: shared by the policy and the judge,
    so each distinct prefix costs one forward pass."""

    def __init__(self, model, processor, data_root, layer):
        self.model, self.processor, self.data_root, self.layer = model, processor, data_root, layer
        self.prompts, self.cache, self.forwards = {}, {}, 0

    def get(self, problem, prefix):
        key = (problem["id"], prefix)
        if key not in self.cache:
            if problem["id"] not in self.prompts:
                batch = encode(self.processor, self.data_root, problem["split"], problem["row"], training=False)
                self.prompts[problem["id"]] = to_device(batch, self.model.device)
            self.forwards += 1
            self.cache[key] = decide(self.model, self.processor.tokenizer, self.prompts[problem["id"]], prefix,
                                     self.layer)
        return self.cache[key]

    def probabilities(self, problem, actions):  # the interface probe_agreement uses
        return self.get(problem, "".join(actions))[0]


class LivrPolicy:
    """LIVR's raw log-probabilities of the candidate moves, and of ending (STOP)."""

    def __init__(self, decisions):
        self.decisions = decisions

    def __call__(self, requests):
        replies = []
        for problem, actions, candidates in requests:
            probs = self.decisions.get(problem, "".join(actions))[0]
            replies.append({c: math.log(max(probs["end" if c == STOP else c], FLOOR)) for c in candidates})
        return replies

    def probabilities(self, problem, actions):
        return self.decisions.probabilities(problem, actions)


class JepaFrozenLake(FrozenLakeEnvironment):
    """FrozenLake judged by the JEPA from LIVR's state: movement follows the rules, the
    outcome of a move onto the grid is the outcome head's argmax. The verifier is the real
    one. A state is (cell, prefix); repeats are judged by the cell."""

    def __init__(self, evaluator, decisions, jepa, device):
        super().__init__(evaluator)
        self.decisions, self.jepa, self.device, self.judgments = decisions, jepa, device, {}

    def start(self, problem):
        return find(problem["row"]["map_desc"], START), ""

    def key(self, state):
        return state[0]

    @torch.no_grad()
    def step(self, problem, state, action):
        cell, prefix = state
        target = in_grid(problem["row"]["map_desc"], cell, action)
        if target is None:
            return (cell, prefix + action), ALIVE
        key = (problem["id"], prefix, action)
        if key not in self.judgments:
            z = self.decisions.get(problem, prefix)[1][None].to(self.device)
            self.judgments[key] = OUTCOMES[int(self.jepa.judge(z, torch.tensor([ACTIONS.index(action)],
                                                                                device=self.device)).argmax())]
        return (target, prefix + action), self.judgments[key]


def in_search_rates(env, problems, results):
    """The judge's errors on the moves the search actually judged: misses and false alarms,
    and arrivals missed or invented (a missed arrival loses the route, since the search
    walks past the goal and never submits it)."""
    count = Counter()
    for problem, result in zip(problems, results):
        grid = problem["row"]["map_desc"]
        for record in result["transitions"]:
            cell = route_cells(grid, record["actions"])[-1]
            for move, outcome in record["outcomes"].items():
                target = in_grid(grid, cell, move)
                if target is None or outcome in ("repeat", "depth"):
                    continue
                truth = cell_outcome(grid, target)
                count["dead"] += truth == DEAD
                count["missed"] += truth == DEAD and outcome != DEAD
                count["not_dead"] += truth != DEAD
                count["alarmed"] += truth != DEAD and outcome == DEAD
                count["done"] += truth == DONE
                count["done_missed"] += truth == DONE and outcome != DONE
                count["done_false"] += truth != DONE and outcome == DONE
    return {"dead": count["dead"], "miss_rate": count["missed"] / max(1, count["dead"]),
            "not_dead": count["not_dead"], "false_alarm_rate": count["alarmed"] / max(1, count["not_dead"]),
            "done": count["done"], "done_missed": count["done_missed"], "done_false": count["done_false"]}


def search_row(result):
    return {k: result[k] for k in ("solved", "stop", "policy_calls", "verifier_calls", "rejected")} | {
        "route": result["solved"] and "".join(result["actions"])}


# The run ----------------------------------------------------------------------------------

def disjoint(data_root, settings):
    """Validity: the JEPA's training maps share no layout with any map it is scored on, or
    with the released splits."""
    train = {layout_key(r) for r in records(data_root, settings["train_split"])}
    others = ["train", "test", settings["heldout_split"], settings["fresh_split"]]
    shared = {split: len(train & {layout_key(r) for r in records(data_root, split)}) for split in others}
    return {"shared_layouts": shared, "valid": not any(shared.values())}


def by_group(predicted, judged, items, key):
    groups = {}
    for g in sorted({key(items[i]) for i in judged[:, 0].tolist()}):
        rows = torch.tensor([key(items[i]) == g for i in judged[:, 0].tolist()])
        groups[str(g)] = rates(predicted[rows], judged[rows, 2])
    return groups


def run_jepa(model, processor, evaluator, data_root, output, settings, run_manifest, probe_points):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began, device = time.time(), model.device
    model.eval()
    seed_all(settings["seed"])
    checks = {"disjoint": disjoint(data_root, settings)}
    json_write(output / "manifest.json", run_manifest | {"settings": settings, "checks": checks})
    layers = settings["layers"]
    every = settings["validation_every"]
    rows = list(enumerate(records(data_root, settings["train_split"])))
    fit_rows = [r for r in rows if r[0] % every != every - 1]
    val_rows = [r for r in rows if r[0] % every == every - 1]
    problems = problems_of(data_root, "test") + problems_of(data_root, settings["fresh_split"])

    timing = {}
    started = time.time()
    x_fit, it_fit = collect(model, processor, data_root, settings["train_split"], fit_rows, layers, settings["seed"], log=True)
    x_val, it_val = collect(model, processor, data_root, settings["train_split"], val_rows, layers, settings["seed"] + 1)
    x_eval, it_eval = [], []
    for split in ("test", settings["fresh_split"]):
        chosen = [(p["row_index"], p["row"]) for p in problems if p["split"] == split]
        x, it = collect(model, processor, data_root, split, chosen, layers, settings["seed"] + 2)
        x_eval.append(x); it_eval += [i | {"population": "search"} for i in it]
    x, it = collect(model, processor, data_root, settings["heldout_split"],
                    list(enumerate(records(data_root, settings["heldout_split"]))), layers, settings["seed"] + 3)
    x_eval.append(x); it_eval += [i | {"population": "heldout"} for i in it]
    x_eval = torch.cat(x_eval)
    timing["states"] = time.time() - started
    torch.save({"layers": layers, "fit": x_fit, "val": x_val, "eval": x_eval}, output / "states.pt")
    for name, items in (("fit", it_fit), ("val", it_val), ("eval", it_eval)):
        (output / f"items-{name}.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))

    j_fit, t_fit = pairs(it_fit)
    j_val, t_val = pairs(it_val)
    j_eval, t_eval = pairs(it_eval)
    n_fit, n_val = next_labels(it_fit, t_fit), next_labels(it_val, t_val)

    # Layer: the lowest validation outcome loss of the JEPA, on validation maps only.
    started = time.time()
    selection, trained = {}, {}
    for li, layer in enumerate(layers):
        jepa, history = train("jepa", x_fit[:, li].float(), j_fit, t_fit, n_fit,
                              (x_val[:, li].float(), j_val, t_val, n_val), settings, device, settings["seed"])
        best = min(history, key=lambda h: h["outcome"] + h["prediction"])
        selection[str(layer)] = {"validation": best, "epochs": len(history)}
        trained[layer] = (jepa, history)
    layer = min(layers, key=lambda l: selection[str(l)]["validation"]["outcome"])
    li = layers.index(layer)
    jepa, history = trained[layer]
    probe, probe_history = train("probe", x_fit[:, li].float(), j_fit, t_fit, n_fit,
                                 (x_val[:, li].float(), j_val, t_val, n_val), settings, device, settings["seed"])
    oracle, oracle_history = train("oracle", x_fit[:, li].float(), j_fit, t_fit, n_fit,
                                   (x_val[:, li].float(), j_val, t_val, n_val), settings, device, settings["seed"])
    timing["training"] = time.time() - started
    torch.save({"layer": layer, "d": x_fit.shape[-1], "jepa": jepa.state_dict(), "probe": probe.state_dict(),
                "oracle": oracle.state_dict()}, output / "jepa.pt")

    # Judge accuracy on unseen maps.
    xe = x_eval[:, li].float()
    judges = {"jepa": judge_all(jepa, xe, j_eval, device), "probe": judge_all(probe, xe, j_eval, device)}
    with torch.no_grad():
        oracle_pred = oracle(xe[t_eval[:, 2]].to(device)).argmax(-1).cpu()
    accuracy = {name: {"all": rates(pred, j_eval[:, 2]),
                       "by_population": by_group(pred, j_eval, it_eval, lambda i: i["population"]),
                       "by_size": by_group(pred, j_eval, it_eval, lambda i: i["size"])}
                for name, pred in judges.items()}
    accuracy["oracle_taken_moves"] = rates(oracle_pred, next_labels(it_eval, t_eval))
    accuracy["jepa_taken_moves"] = rates(judge_all(jepa, xe, torch.stack([t_eval[:, 0], t_eval[:, 1],
                                                                          next_labels(it_eval, t_eval)], 1), device),
                                         next_labels(it_eval, t_eval))
    imagined = imagination(jepa, xe, it_eval, device)

    # Validity: batched states equal single-pass states, and the policy matches step 0.
    rng = random.Random(settings["seed"])
    by_id = {p["id"]: p for p in problems}
    sample = rng.sample([i for i, it in enumerate(it_eval) if it["population"] == "search" and it["alive"]], 20)
    cosines = []
    for i in sample:
        it = it_eval[i]
        single = decide(model, processor.tokenizer,
                        to_device(encode(processor, data_root, it["split"], by_id[f"{it['split']}/{it['row_index']}"]["row"],
                                         training=False), device), it["route"][:it["k"]], layer)[1]
        cosines.append(float(torch.nn.functional.cosine_similarity(single, xe[i], 0)))
    probe_diffs = probe_agreement(Decisions(model, processor, data_root, layer), problems, probe_points)
    checks |= {"state_alignment_min_cosine": min(cosines), "probe_points": len(probe_diffs),
               "probe_max_difference": max(probe_diffs) if probe_diffs else None}

    interim = {"layer": layer, "layer_selection": selection, "checks": checks, "judge_accuracy": accuracy,
               "imagination": imagined, "timing": timing, "seconds": time.time() - began}
    json_write(output / "summary-before-search.json", interim)  # survives a failure in the search phase

    # Search on the unseen detour maps, LIVR as the policy in every arm.
    started = time.time()
    arms, forwards = {}, {}
    for arm in ("exact", "jepa", "verdict_only"):
        arm_started = time.time()
        decisions = Decisions(model, processor, data_root, layer)
        policy = LivrPolicy(decisions)
        if arm == "exact":
            results = guided_search(FrozenLakeEnvironment(evaluator), policy, problems, settings["budget"],
                                    settings["max_depth"], max_expansions=settings["max_expansions"])
        elif arm == "jepa":
            env = JepaFrozenLake(evaluator, decisions, jepa, device)
            results = guided_search(env, policy, problems, settings["budget"], settings["max_depth"],
                                    veto=settings["veto"], reopen=settings["reopen"],
                                    max_expansions=settings["max_expansions"])
            arms["jepa_in_search_rates"] = in_search_rates(env, problems, results)
        else:
            results = verdict_search(FrozenLakeEnvironment(evaluator), policy, problems, settings["budget"],
                                     settings["max_depth"], log=False)
        arms[arm] = results
        forwards[arm] = decisions.forwards
        timing[f"search_{arm}"] = time.time() - arm_started
        # Each arm is kept as it finishes, so a run cut short still has the arms it completed.
        with (output / f"searches-{arm}.jsonl").open("w", encoding="utf-8") as log:
            for p, r in zip(problems, results):
                log.write(json.dumps({"id": p["id"], "grid_size": len(p["row"]["map_desc"])} | search_row(r)) + "\n")
        json_write(output / f"search-{arm}.json", {"solved": sum(bool(r["solved"]) for r in results),
                                                   "stops": dict(Counter(r["stop"] for r in results)),
                                                   "forwards": decisions.forwards, "seconds": timing[f"search_{arm}"]}
                   | ({"in_search_rates": arms["jepa_in_search_rates"]} if arm == "jepa" else {}))
    timing["search"] = time.time() - started
    with (output / "searches.jsonl").open("w", encoding="utf-8") as log:
        for i, p in enumerate(problems):
            log.write(json.dumps({"id": p["id"], "grid_size": len(p["row"]["map_desc"]),
                                  **{arm: search_row(arms[arm][i]) for arm in ("exact", "jepa", "verdict_only")}}) + "\n")
    for i, p in enumerate(problems):
        for arm in ("exact", "jepa", "verdict_only"):
            r = arms[arm][i]
            if r["solved"] and not evaluator.score(p["row"], r["answer"])["success"]:
                raise ValueError(f"An accepted answer fails the evaluator: {arm} {p['id']}")
    solved = {arm: [bool(r["solved"]) for r in arms[arm]] for arm in ("exact", "jepa", "verdict_only")}
    only_jepa = sum(a and not b for a, b in zip(solved["jepa"], solved["verdict_only"]))
    only_verdict = sum(b and not a for a, b in zip(solved["jepa"], solved["verdict_only"]))
    primary = {"maps": len(problems), "jepa": sum(solved["jepa"]), "verdict_only": sum(solved["verdict_only"]),
               "only_jepa": only_jepa, "only_verdict_only": only_verdict, "p": mcnemar(only_jepa, only_verdict)}
    valid = (checks["disjoint"]["valid"] and checks["state_alignment_min_cosine"] >= ALIGNMENT
             and bool(probe_diffs) and checks["probe_max_difference"] <= settings["probe_tolerance"]
             and sum(solved["exact"]) >= READY * len(problems))
    checks |= {"exact_arm_solved": sum(solved["exact"]), "valid": valid}
    withheld = "withheld: the measurement failed a validity check"
    reading = withheld if not valid else (
        "the learned judge makes search work without the simulator" if primary["p"] < 0.05 and only_jepa > only_verdict
        else "the learned judge hurts search" if primary["p"] < 0.05 else "no detectable difference")
    ready = withheld if not valid else ("ready" if sum(solved["jepa"]) >= READY * len(problems) else "not ready")
    calls = lambda arm: {"median": statistics.median(r["policy_calls"] if r["solved"] else settings["budget"] + 1
                                                     for r in arms[arm]),
                         "forwards": forwards[arm]}
    summary = {
        "layer": layer, "layer_selection": selection, "checks": checks,
        "primary": primary | {"reading": reading}, "ready": ready,
        "search": {arm: {"solved": sum(solved[arm]), "stops": dict(Counter(r["stop"] for r in arms[arm])),
                         **calls(arm)} for arm in ("exact", "jepa", "verdict_only")}
        | {"jepa_in_search_rates": arms["jepa_in_search_rates"], "setting": {"veto": settings["veto"],
                                                                             "reopen": settings["reopen"]}},
        "judge_accuracy": accuracy, "imagination": imagined,
        "training": {"jepa": history, "probe": probe_history, "oracle": oracle_history},
        "data": {"fit_maps": len(fit_rows), "validation_maps": len(val_rows), "fit_states": len(it_fit),
                 "fit_judged": len(j_fit), "fit_taken": len(t_fit), "eval_states": len(it_eval),
                 "eval_judged": len(j_eval)},
        "seconds": time.time() - began, "timing": timing,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    json_write(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("layer", "checks", "primary", "ready", "search")}, indent=1), flush=True)
    return summary
