"""FrozenLake as a search environment, LIVR as its policy, and the step 1a search pilot
(docs/GUIDED_SEARCH_SPEC.md).

A problem is a dict: "id", "split", "row_index" and "row" (a record in the released format).
"""
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from .data import records
from .detour_probe import find, is_detour, move_probabilities
from .planning_probe import MOVES, WORDS, distances
from .runtime import json_write, seed_all
from .search import ALIVE, DEAD, DONE, STOP, fixed_draw, guided_search, uniform_policy
from .serialization import encode
from .training import to_device

START, GOAL, HOLE = 1, 2, -1
BUDGET = 64  # policy calls per map
READY, GUIDANCE, PROBE_TOLERANCE = 0.9, 0.5, 0.01  # reading and validity rules
FLOOR = 1e-30  # probabilities that underflow to zero still rank last
RAW = {"greedy": {}, "sampled": {}}  # move_probabilities settings: the raw softmax only


class FrozenLakeEnvironment:
    """The reference evaluator's rules, one move at a time: a hole ends the route, the goal
    finishes it, and a move off the grid leaves the agent where it is."""

    def __init__(self, evaluator):
        self.evaluator = evaluator

    def start(self, problem):
        return find(problem["row"]["map_desc"], START)

    def actions(self, problem, state):
        return list(MOVES)

    def step(self, problem, state, action):
        return move(problem["row"]["map_desc"], state, action)

    def key(self, state):
        return state

    def answer(self, actions):
        return "\\boxed{" + ", ".join(WORDS[a] for a in actions) + "}"

    def reward(self, problem, answer):
        return float(self.evaluator.score(problem["row"], answer)["success"])


def move(grid, state, action):
    """One move under the reference rules on `grid`: (next cell, outcome)."""
    r, c = state[0] + MOVES[action][0], state[1] + MOVES[action][1]
    if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
        return state, ALIVE
    value = grid[r][c]
    return (r, c), DEAD if value == HOLE else DONE if value == GOAL else ALIVE


class MisperceivedFrozenLake(FrozenLakeEnvironment):
    """A judge that misreads the map (step 1a′, per-cell noise): each hole is read as safe
    with probability `miss`, each safe cell other than the start and goal as a hole with
    probability `false_alarm`. Every move into a cell is judged by the reading. One fixed
    draw per cell and seed, so the errors at a rate include those at every smaller rate.
    The verifier still scores the true map."""

    def __init__(self, evaluator, miss, false_alarm, seed):
        super().__init__(evaluator)
        self.miss, self.false_alarm, self.seed, self.readings = miss, false_alarm, seed, {}

    def reading(self, problem):
        if problem["id"] not in self.readings:
            grid = [list(row) for row in problem["row"]["map_desc"]]
            for r, row in enumerate(grid):
                for c, value in enumerate(row):
                    rate = self.miss if value == HOLE else self.false_alarm if value == 0 else 0.0
                    if rate and fixed_draw(self.seed, problem["id"], (r, c)) < rate:
                        row[c] = 0 if value == HOLE else HOLE
            self.readings[problem["id"]] = grid
        return self.readings[problem["id"]]

    def step(self, problem, state, action):
        return move(self.reading(problem), state, action)


class LivrMovePolicy:
    """LIVR's raw move log-probabilities after a route prefix, from one teacher-forced pass
    per request. Each problem's prompt is encoded once."""

    def __init__(self, model, processor, data_root):
        self.model, self.processor, self.data_root = model, processor, data_root
        self.prompts, self.calls = {}, 0

    def prompt(self, problem):
        if problem["id"] not in self.prompts:
            batch = encode(self.processor, self.data_root, problem["split"], problem["row"], training=False)
            self.prompts[problem["id"]] = to_device(batch, self.model.device)
        return self.prompts[problem["id"]]

    def probabilities(self, problem, actions):
        self.calls += 1
        return move_probabilities(self.model, self.processor.tokenizer, self.prompt(problem), "".join(actions),
                                  RAW)["raw"]["moves"]

    def __call__(self, requests):
        replies = []
        for problem, actions, candidates in requests:
            probs = self.probabilities(problem, actions)
            replies.append({c: math.log(max(probs[c], FLOOR)) for c in candidates})
        return replies


def toward_goal_policy(requests, sharpness=3.0):
    """A model-free stand-in with LIVR's failure: it prefers moves that shorten the grid
    distance to the goal, whatever lies on the way. It knows the moving rules (off the
    grid, the agent stays) but not the holes. Ending the answer (STOP) scores the distance
    it leaves, so at the goal ending is the likeliest choice."""
    replies = []
    for problem, actions, candidates in requests:
        grid = problem["row"]["map_desc"]
        n, goal = len(grid), find(grid, GOAL)
        clip = lambda r, c: (min(max(r, 0), n - 1), min(max(c, 0), n - 1))
        r, c = find(grid, START)
        for a in actions:
            r, c = clip(r + MOVES[a][0], c + MOVES[a][1])

        def distance(m):
            rr, cc = (r, c) if m == STOP else clip(r + MOVES[m][0], c + MOVES[m][1])
            return abs(rr - goal[0]) + abs(cc - goal[1])
        replies.append({m: -sharpness * distance(m) for m in candidates})
    return replies


# Pilot -----------------------------------------------------------------------------

def problems_of(data_root, split, detour_only=True):
    return [{"id": f"{split}/{i}", "split": split, "row_index": i, "row": row}
            for i, row in enumerate(records(data_root, split)) if not detour_only or is_detour(row["map_desc"])]


def simulator_agreement(env, problems, results):
    """Every alive, dead or done outcome the searches recorded, against the reference
    evaluator's verdict on the same route: (agreeing, checked)."""
    expected = {ALIVE: "Did not reach goal", DEAD: "Fell in hole", DONE: "Reached goal"}
    agree = checked = 0
    for problem, result in zip(problems, results):
        for record in result["transitions"]:
            for action, outcome in record["outcomes"].items():
                if outcome in expected:
                    verdict = env.evaluator.score(problem["row"], env.answer(record["actions"] + [action]))["status"]
                    agree += verdict == expected[outcome]
                    checked += 1
    return agree, checked


def probe_agreement(policy, problems, probe_points):
    """The policy's move probabilities at step 0's primary points on these problems, against
    the probe's. Forced points occur only on detour maps, so every one has a problem here
    when the splits match."""
    by_id = {p["id"]: p for p in problems}
    differences = []
    for line in Path(probe_points).read_text(encoding="utf-8").splitlines():
        point = json.loads(line)
        if (point["split"] == "train" or point["kind"] != "forced" or point["taken_outcome"] == "optimal"
                or f"{point['split']}/{point['row_index']}" not in by_id):
            continue
        problem = by_id[f"{point['split']}/{point['row_index']}"]
        probs = policy.probabilities(problem, point["route"][:point["k"] - 1])
        differences.append(max(abs(probs[m] - point["p_moves"][m]) for m in MOVES))
    return differences


def route_scores(policy, problem, actions, transitions):
    """Scores of the four moves at each step of a found route: the search's own where it
    called the policy, a new call where the move was forced. Only their order is used."""
    known = {tuple(t["actions"]): t["scores"] for t in transitions if t["scores"]}
    return [known.get(tuple(actions[:j])) or policy.probabilities(problem, actions[:j]) for j in range(len(actions))]


def censored(result, budget):
    return result["policy_calls"] if result["solved"] else budget + 1


def describe(problem, guided, uniform, budget, departures):
    grid = problem["row"]["map_desc"]
    shortest = distances(grid)[find(grid, START)]
    row = {"split": problem["split"], "row_index": problem["row_index"], "grid_size": len(grid),
           "shortest": shortest, "departures": departures}
    for name, result in (("guided", guided), ("uniform", uniform)):
        row[name] = {k: result[k] for k in ("solved", "actions", "log_likelihood", "stop", "policy_calls",
                                            "expansions", "forced_expansions", "rejected")}
        row[name]["calls_censored"] = censored(result, budget)
        row[name]["excess_length"] = len(result["actions"]) - shortest if result["solved"] else None
    return row


def summarise(rows, agreement, differences, budget):
    def share(values):
        return sum(values) / len(values) if values else None

    def coverage(subset):
        return {"maps": len(subset), "solved": sum(r["guided"]["solved"] for r in subset),
                "share": share([r["guided"]["solved"] for r in subset]),
                "uniform_solved": sum(r["uniform"]["solved"] for r in subset)}

    agree, checked = agreement
    valid = checked > 0 and agree == checked and bool(differences) and max(differences) <= PROBE_TOLERANCE
    primary = coverage(rows)
    guided = statistics.median(r["guided"]["calls_censored"] for r in rows)
    uniform = statistics.median(r["uniform"]["calls_censored"] for r in rows)
    withheld = "withheld: the measurement failed a validity check"
    solved = [r for r in rows if r["guided"]["solved"]]
    deps = [r["departures"] for r in solved if r["departures"] is not None]
    return {
        "validity": {"simulator_agreement": [agree, checked], "probe_points": len(differences),
                     "probe_max_difference": max(differences) if differences else None, "valid": valid},
        "primary": primary | {"by_split": {s: coverage([r for r in rows if r["split"] == s])
                                           for s in sorted({r["split"] for r in rows})},
                              "by_size": {str(n): coverage([r for r in rows if r["grid_size"] == n])
                                          for n in sorted({r["grid_size"] for r in rows})}},
        "reading": withheld if not valid else "ready" if primary["share"] >= READY else "insufficient",
        "guidance": {"median_calls_guided": guided, "median_calls_uniform": uniform,
                     "guided_fewer": sum(r["guided"]["calls_censored"] < r["uniform"]["calls_censored"] for r in rows),
                     "uniform_fewer": sum(r["guided"]["calls_censored"] > r["uniform"]["calls_censored"] for r in rows),
                     "reading": withheld if not valid else
                     "guidance saves calls" if guided <= GUIDANCE * uniform else "no clear saving"},
        "descriptive": {
            "excess_length": dict(sorted(Counter(r["guided"]["excess_length"] for r in solved).items())),
            "departures": {"routes": len(deps), "median_moves": statistics.median(d["moves"] for d in deps) if deps else None,
                           "into_hole": sum(d["into_hole"] for d in deps),
                           "routes_with_any": sum(d["moves"] > 0 for d in deps)},
            "forced_expansions": sum(r["guided"]["forced_expansions"] for r in rows),
            "rejected": sum(r["guided"]["rejected"] + r["uniform"]["rejected"] for r in rows),
            "stops": dict(Counter(r["guided"]["stop"] for r in rows))}}


def departures_of(env, problem, actions, scores):
    """Steps of a found route where the move was not the policy's top raw choice of the
    four, and how many of those top choices would have walked into a hole."""
    moves = into_hole = 0
    state = env.start(problem)
    for action, probs in zip(actions, scores):
        top = max(probs, key=probs.get)
        if top != action:
            moves += 1
            into_hole += env.step(problem, state, top)[1] == DEAD
        state = env.step(problem, state, action)[0]
    return {"moves": moves, "into_hole": into_hole}


def search_pilot(model, processor, evaluator, data_root, output, probe_points, fresh_split, run_manifest, seed,
                 budget=BUDGET):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    problems = problems_of(data_root, "test") + (problems_of(data_root, fresh_split) if fresh_split else [])
    depth = max(len(p["row"]["map_desc"]) ** 2 for p in problems)  # one move per cell: never binds
    json_write(output / "manifest.json", run_manifest | {
        "probe_points": str(probe_points), "fresh_split": fresh_split, "maps": len(problems), "budget": budget,
        "max_depth": depth, "rules": {"ready": READY, "guidance": GUIDANCE, "probe_tolerance": PROBE_TOLERANCE},
        "seed": seed})
    seed_all(seed)
    model.eval()
    env, policy = FrozenLakeEnvironment(evaluator), LivrMovePolicy(model, processor, data_root)
    began = time.time()
    differences = probe_agreement(policy, problems, probe_points)
    validity_calls, started = policy.calls, time.time()
    guided = guided_search(env, policy, problems, budget, depth)
    search_seconds, search_calls = time.time() - started, policy.calls - validity_calls
    uniform = guided_search(env, uniform_policy, problems, budget, depth)
    rows = []
    with (output / "searches.jsonl").open("w", encoding="utf-8") as log:
        for problem, g, u in zip(problems, guided, uniform):
            departures = None
            if g["solved"]:
                scores = route_scores(policy, problem, g["actions"], g["transitions"])
                departures = departures_of(env, problem, g["actions"], scores)
            row = describe(problem, g, u, budget, departures)
            rows.append(row)
            log.write(json.dumps(row | {"transitions": {"guided": g["transitions"], "uniform": u["transitions"]}}) + "\n")
    summary = summarise(rows, simulator_agreement(env, problems * 2, guided + uniform), differences, budget)
    summary |= {"policy_calls": {"validity": validity_calls, "search": search_calls,
                                 "departure_rescoring": policy.calls - validity_calls - search_calls},
                "search_seconds": search_seconds, "seconds": time.time() - began,
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    json_write(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("validity", "primary", "reading", "guidance")}, indent=1), flush=True)
    return summary
