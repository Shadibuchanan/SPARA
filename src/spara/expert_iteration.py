"""Step 1b of docs/GUIDED_SEARCH_SPEC.md, specified in docs/EXPERT_ITERATION_SPEC.md: one
round of expert iteration whose routes come from final-verdict-only search.

`run_search` searches generated detour maps with LIVR as the policy and only the
verifier's accept or reject (no step simulator, no judge, no labels), and writes the
routes it finds. `run_train` fine-tunes LIVR on them, or, as the matched control, on the
same maps' shortest-route labels, with the detour-scaling recipe and evaluation.
`compare` applies the readings fixed in the specification.
"""
import contextlib
import json
import math
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from .data import records
from .detour_probe import alignment
from .detour_scaling import HELD_OUT, jsonl, paired, run_arm, verdict
from .jepa import decide
from .planning_probe import MOVES
from .route_search import FLOOR, FrozenLakeEnvironment, problems_of
from .runtime import json_write, seed_all
from .search import STOP, verdict_search
from .serialization import encode
from .training import amp

LABELS = ("search", "shortest")
SAMPLE = 32  # validity: cached batched answers re-read one at a time
PROMPT_KEYS = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}


def route_of(answer):
    """Move letters of an answer in the released format."""
    return "".join(word[0] for word in re.findall(r"UP|DOWN|LEFT|RIGHT", answer))


# LIVR as a batched policy ------------------------------------------------------------------

PRECISIONS = ("bf16", "tf32")


@contextlib.contextmanager
def read_precision(device, precision):
    """bf16: autocast, as every other pass (round 1's reads). tf32: the FP32 weights with
    TF32 matmuls, whose rounding is finer, for batched reads (docs/EXPERT_ITERATION_ROUND2_SPEC.md)."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}")
    if precision == "bf16":
        with amp(device):
            yield
        return
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@torch.no_grad()
def batched_decide(model, tokenizer, prompts, prefixes, precision="bf16"):
    """LIVR's raw move and ending probabilities after each prefix, as `jepa.decide` computes
    them, for rows with different prompts in one right-padded forward pass. Stage 2
    attention is causal, so padding after a row's last token does not change the row."""
    rows, reads = [], []
    for prompt, prefix in zip(prompts, prefixes):
        if set(prompt) - PROMPT_KEYS:
            raise ValueError(f"Unexpected prompt fields {sorted(set(prompt) - PROMPT_KEYS)}")
        context, at, split, tokens, end_token, continue_token = alignment(tokenizer, prefix)
        rows.append(torch.cat([prompt["input_ids"][0].cpu(), context.cpu()]))
        reads.append((prompt["input_ids"].shape[-1], at, split, tokens, end_token, continue_token))
    ids = torch.full((len(rows), max(len(r) for r in rows)), tokenizer.pad_token_id or 0, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, row in enumerate(rows):
        ids[i, :len(row)], mask[i, :len(row)] = row, 1
    device = model.device
    batch = {"input_ids": ids.to(device), "attention_mask": mask.to(device),
             "pixel_values": torch.cat([p["pixel_values"].cpu() for p in prompts]).to(device),
             "image_grid_thw": torch.cat([p["image_grid_thw"].cpu() for p in prompts]).to(device)}
    with read_precision(device, precision):
        logits = model(**batch, use_cache=False).logits
    out = []
    for i, (length, at, split, tokens, end_token, continue_token) in enumerate(reads):
        move = torch.softmax(logits[i, length + at - 1].float(), -1)
        if split == at:
            carry, end = 1.0, float(move[end_token])
        else:
            choice = torch.softmax(logits[i, length + split - 1].float(), -1)
            carry, end = float(choice[continue_token]), float(choice[end_token])
        out.append({m: carry * float(move[t]) for m, t in tokens.items()} | {"end": end})
    return out


class BatchedLivr:
    """LIVR as the search's policy: raw log-probabilities of the moves and of ending (STOP),
    cached per (problem, prefix). Uncached requests are read `batch` rows per forward pass,
    rows from different maps together."""

    def __init__(self, model, processor, data_root, batch, precision="bf16"):
        self.model, self.processor, self.data_root, self.batch = model, processor, data_root, batch
        self.precision = precision
        self.prompts, self.cache = {}, {}
        self.sequences = self.forward_passes = 0

    def prompt(self, problem):
        if problem["id"] not in self.prompts:
            self.prompts[problem["id"]] = encode(self.processor, self.data_root, problem["split"], problem["row"],
                                                 training=False)
        return self.prompts[problem["id"]]

    def many(self, queries):
        """Probabilities for [(problem, prefix)], computing the uncached ones in batches."""
        problems = {p["id"]: p for p, _ in queries}
        missing = list(dict.fromkeys((p["id"], prefix) for p, prefix in queries if (p["id"], prefix) not in self.cache))
        for start in range(0, len(missing), self.batch):
            chunk = missing[start:start + self.batch]
            probs = batched_decide(self.model, self.processor.tokenizer, [self.prompt(problems[i]) for i, _ in chunk],
                                   [prefix for _, prefix in chunk], self.precision)
            self.cache.update(zip(chunk, probs))
            self.sequences += len(chunk)
            self.forward_passes += 1
        return [self.cache[(p["id"], prefix)] for p, prefix in queries]

    def __call__(self, requests):
        probs = self.many([(problem, "".join(actions)) for problem, actions, _ in requests])
        return [{c: math.log(max(p["end" if c == STOP else c], FLOOR)) for c in candidates}
                for p, (_, _, candidates) in zip(probs, requests)]

    def probabilities(self, problem, actions):
        return self.many([(problem, "".join(actions))])[0]

    def reset(self, batch, precision=None):
        self.batch, self.cache = batch, {}
        self.precision = precision or self.precision


def point_differences(policy, problems, probe_points):
    """Validity: the policy's move probabilities at step 0's primary points (as
    `route_search.probe_agreement` selects them), all requested at once so that rows of
    different maps share forward passes."""
    by_id = {p["id"]: p for p in problems}
    points = [json.loads(line) for line in Path(probe_points).read_text(encoding="utf-8").splitlines()]
    points = [p for p in points if p["split"] != "train" and p["kind"] == "forced" and p["taken_outcome"] != "optimal"
              and f"{p['split']}/{p['row_index']}" in by_id]
    probs = policy.many([(by_id[f"{p['split']}/{p['row_index']}"], p["route"][:p["k"] - 1]) for p in points])
    return [max(abs(q[m] - p["p_moves"][m]) for m in MOVES) for q, p in zip(probs, points)]


def single_differences(policy, problems, rng, count=SAMPLE):
    """Validity: cached batched answers against one teacher-forced pass each."""
    by_id = {p["id"]: p for p in problems}
    keys = rng.sample(sorted(k for k in policy.cache if k[0] in by_id), min(count, len(policy.cache)))
    out = []
    for problem_id, prefix in keys:
        prompt = {k: v.to(policy.model.device) for k, v in policy.prompt(by_id[problem_id]).items()}
        single = decide(policy.model, policy.processor.tokenizer, prompt, prefix, 0)[0]
        batched = policy.cache[(problem_id, prefix)]
        out.append(max(abs(single[k] - batched[k]) for k in single))
    return out


# Search ------------------------------------------------------------------------------------

def search_row(problem, result):
    row = problem["row"]
    label = route_of(row["text_output"]) if row.get("text_output") else None
    route = "".join(result["actions"]) if result["solved"] else None
    return {"id": problem["id"], "row_index": problem["row_index"], "map_id": row.get("map_id"),
            "grid_size": len(row["map_desc"]), "solved": result["solved"], "stop": result["stop"],
            "policy_calls": result["policy_calls"], "verifier_calls": result["verifier_calls"],
            "rejected": result["rejected"], "route": route, "answer": result["answer"], "label_route": label,
            "equals_label": route is not None and route == label,
            "longer_than_label": route is not None and label is not None and len(route) > len(label)}


def summarise(rows, budget):
    sizes = sorted({r["grid_size"] for r in rows})
    solved = [r for r in rows if r["solved"]]
    return {"maps": len(rows), "solved": len(solved),
            "by_size": {str(n): [sum(r["solved"] for r in rows if r["grid_size"] == n),
                                 sum(r["grid_size"] == n for r in rows)] for n in sizes},
            "stops": dict(Counter(r["stop"] for r in rows)),
            "policy_calls_median": statistics.median(r["policy_calls"] if r["solved"] else budget + 1 for r in rows)
            if rows else None,
            "policy_calls_median_solved": statistics.median(r["policy_calls"] for r in solved) if solved else None,
            "equal_to_label": sum(r["equals_label"] for r in solved),
            "longer_than_label": sum(r["longer_than_label"] for r in solved),
            "rejected": sum(r["rejected"] for r in rows)}


def searched(env, policy, problems, settings):
    started, sequences, passes = time.time(), policy.sequences, policy.forward_passes
    results = verdict_search(env, policy, problems, settings["budget"], settings["max_depth"], log=False)
    for problem, result in zip(problems, results):
        if result["solved"] and not env.evaluator.score(problem["row"], result["answer"])["success"]:
            raise ValueError(f"An accepted answer fails the evaluator: {problem['id']}")
    rows = [search_row(p, r) for p, r in zip(problems, results)]
    return rows, {"sequences": policy.sequences - sequences, "forward_passes": policy.forward_passes - passes,
                  "seconds": time.time() - started}


def reproduction(rows, reference):
    """The 97 detour maps searched again, against the JEPA run's final-verdict-only arm."""
    recorded = {r["id"]: r for r in jsonl(reference)}
    same = [r for r in rows if r["id"] in recorded and bool(recorded[r["id"]]["solved"]) == r["solved"]
            and (recorded[r["id"]]["route"] or None) == r["route"]]
    return {"maps": len(rows), "recorded": sum(r["id"] in recorded for r in rows), "same_outcome_and_route": len(same),
            "solved": sum(r["solved"] for r in rows), "recorded_solved": sum(bool(r["solved"]) for r in recorded.values())}


def run_search(model, processor, evaluator, data_root, output, settings, run_manifest, probe_points, reference=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    model.eval()
    seed_all(settings["seed"])
    json_write(output / "manifest.json", run_manifest | {"settings": settings})
    env = FrozenLakeEnvironment(evaluator)
    detours = problems_of(data_root, "test") + problems_of(data_root, settings["fresh_split"])
    pool = [{"id": f"{settings['pool_split']}/{i}", "split": settings["pool_split"], "row_index": i, "row": row}
            for i, row in enumerate(records(data_root, settings["pool_split"])[:settings["maps"]])]
    if len(pool) != settings["maps"]:
        raise ValueError(f"The pool has fewer than {settings['maps']} maps")
    policy = BatchedLivr(model, processor, data_root, settings["batch"])
    rng = random.Random(settings["seed"])

    # Validity: the batched policy reads what single passes read. If it does not, fall back
    # to one row per pass, which is the JEPA run's policy.
    checks = {"batch": settings["batch"]}
    points = point_differences(policy, detours, probe_points)
    detour_rows, detour_cost = searched(env, policy, detours, settings)
    singles = single_differences(policy, detours, rng)
    checks |= {"probe_points": len(points), "probe_max_difference": max(points) if points else None,
               "single_max_difference": max(singles) if singles else None}
    agrees = bool(points) and max(points) <= settings["probe_tolerance"] and max(singles) <= settings["probe_tolerance"]
    if not agrees and settings["batch"] > 1:
        checks["batched_failed"] = {k: checks[k] for k in ("probe_max_difference", "single_max_difference")}
        policy.reset(1)
        points = point_differences(policy, detours, probe_points)
        detour_rows, detour_cost = searched(env, policy, detours, settings)
        checks |= {"batch": 1, "probe_max_difference": max(points) if points else None, "single_max_difference": None}
        agrees = bool(points) and max(points) <= settings["probe_tolerance"]
    checks["policy_agrees"] = agrees
    with (output / "detour-searches.jsonl").open("w", encoding="utf-8") as log:
        log.writelines(json.dumps(r) + "\n" for r in detour_rows)
    detour = summarise(detour_rows, settings["budget"]) | {"cost": detour_cost}
    if reference:
        detour["reproduction"] = reproduction(detour_rows, reference)
    json_write(output / "summary-before-pool.json", {"checks": checks, "detour_maps": detour})

    # The pool: the routes expert iteration trains on.
    rows, cost = searched(env, policy, pool, settings)
    with (output / "routes.jsonl").open("w", encoding="utf-8") as log:
        log.writelines(json.dumps(r) + "\n" for r in rows)
    summary = {"checks": checks | {"valid": agrees}, "detour_maps": detour,
               "pool": summarise(rows, settings["budget"]) | {"cost": cost},
               "seconds": time.time() - began,
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    json_write(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("checks", "pool")}, indent=1), flush=True)
    return summary


# Fine-tuning ---------------------------------------------------------------------------------

def training_rows(data_root, recipe, routes, labels):
    """The released training maps plus `extra_examples` drawn in turn from the pool maps the
    search solved, in pool order: with the found route as the answer ("search") or with the
    map's shortest-route label ("shortest", the matched control). Returns the rows and the
    number of distinct pool maps."""
    if labels not in LABELS:
        raise ValueError(f"labels must be one of {LABELS}")
    pool = records(data_root, recipe["pool_split"])
    solved = [r for r in routes if r["solved"]]
    if not solved:
        raise ValueError("The search solved no map")
    rows = []
    for r in solved:
        row = pool[r["row_index"]]
        if row.get("map_id") != r["map_id"]:
            raise ValueError(f"Route {r['row_index']} does not belong to this pool")
        rows.append(row | {"text_output": r["answer"]} if labels == "search" else row)
    extra = [(recipe["pool_split"], rows[i % len(rows)]) for i in range(recipe["extra_examples"])]
    return [("train", row) for row in records(data_root, "train")] + extra, len(rows)


def run_train(model, processor, evaluator, config, data_root, output, recipe, labels, routes_path, run_manifest):
    routes = jsonl(routes_path)
    data, distinct = training_rows(data_root, recipe, routes, labels)
    return run_arm(model, processor, evaluator, config, data_root, output, labels, recipe,
                   run_manifest | {"labels": labels, "routes": str(routes_path)}, data=data, distinct=distinct,
                   benchmark=labels == "search")


# Readings ------------------------------------------------------------------------------------

def compare(runs, search_run, livr_outcomes):
    """`runs`: name -> run directory with greedy.jsonl, for search, shortest, livr and n500
    (the last two from the detour scaling check). `search_run`: the route search's run.
    `livr_outcomes`: LIVR's per-map benchmark outcomes, for the harm check."""
    greedy = {name: {(r["population"], r["row_index"]): r for r in jsonl(Path(run) / "greedy.jsonl")}
              for name, run in runs.items()}
    keys = sorted(greedy["livr"])
    if any(sorted(g) != keys for g in greedy.values()):
        raise ValueError("The arms evaluated different maps")
    held = [k for k in keys if k[0] in HELD_OUT]
    success = {name: {k: bool(g[k]["success"]) for k in keys} for name, g in greedy.items()}
    sizes = {k: greedy["livr"][k]["grid_size"] for k in keys}

    def compared(a, b):
        return paired(success[a], success[b], held) | {
            "by_population": {p: paired(success[a], success[b], [k for k in held if k[0] == p]) for p in HELD_OUT},
            "by_size": {str(n): paired(success[a], success[b], [k for k in held if sizes[k] == n])
                        for n in sorted({sizes[k] for k in held})}}

    summaries = {name: json.loads((Path(runs[name]) / "summary.json").read_text()) for name in LABELS}
    search = json.loads((Path(search_run) / "summary.json").read_text())
    routes = {r["row_index"]: r for r in jsonl(Path(search_run) / "routes.jsonl")}
    valid = search["checks"]["valid"] and all(s["checks"]["valid"] for s in summaries.values())
    withheld = "withheld: the measurement failed a validity check"
    primary, secondary = compared("search", "livr"), compared("search", "shortest")
    seen = [k for k in keys if k[0] == "seen"]
    trained = [k for k in seen if routes.get(k[1], {}).get("solved")]
    livr_bench = {r["row_index"]: bool(r["success"]) for r in jsonl(livr_outcomes)}
    mine = {r["row_index"]: bool(r["success"]) for r in jsonl(Path(runs["search"]) / "evaluation/responses.jsonl")}
    detour_rows = {k[1] for k in keys if k[0] == "test_detour"}
    rows = sorted(livr_bench)
    return {
        "validity": {"search": search["checks"], "set_checks": {n: s["checks"]["valid"] for n, s in summaries.items()},
                     "valid": valid},
        "primary": primary | {"reading": withheld if not valid else verdict(
            primary, "expert iteration helps", "expert iteration harms", "no detectable difference")},
        "secondary": secondary | {"reading": withheld if not valid else verdict(
            secondary, "found routes teach better than labels", "found routes teach worse than labels",
            "no detectable difference between found routes and labels")},
        "descriptive": {
            "held_out_success": {name: sum(success[name][k] for k in held) for name in runs},
            "search_vs_n500": compared("search", "n500"), "shortest_vs_n500": compared("shortest", "n500"),
            "seen": {name: {"trained": sum(success[name][k] for k in trained), "untrained":
                            sum(success[name][k] for k in seen if k not in trained)} for name in runs}
            | {"maps": {"trained": len(trained), "untrained": len(seen) - len(trained)}},
            "benchmark_vs_livr": {"all": paired(mine, livr_bench, rows),
                                  "detour": paired(mine, livr_bench, [r for r in rows if r in detour_rows]),
                                  "no_detour": paired(mine, livr_bench, [r for r in rows if r not in detour_rows])},
            "routes": search["pool"], "training": {n: s["training"] for n, s in summaries.items()}},
        "seconds_in_jobs": {"routes": search["seconds"], **{n: s["seconds"] for n, s in summaries.items()}}}
