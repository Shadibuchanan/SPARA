"""Expert iteration round 2 (docs/EXPERT_ITERATION_ROUND2_SPEC.md): a second round from E's
weights on 7 x 7 and 8 x 8 detour maps, with and without LIVR's bottleneck in training.

- `batching_check` (CPU): batched against single policy reads, in full precision and
  under bfloat16 autocast, for the spec's precondition.
- `run_search`: E's evaluation on the new sets, then final-verdict-only search with E as
  the policy over round 1's unsolved pool maps (the frontier) and the new pool.
- `livr_greedy`: L's greedy answers on the new held-out maps (same job, after E).
- `run_arm`: E2 or EB2. Two training phases on the same shuffled rows; EB2 trains its
  first phase under the Stage 1 mask. Then the same evaluation as E's.
- `compare` (CPU): the readings fixed in the specification.
"""
import contextlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch

from .benchmark import run_benchmark
from .data import records
from .detour_scaling import ALPHA, check_sets, greedy_eval, jsonl, paired
from .expert_iteration import SAMPLE, BatchedLivr, batched_decide, searched, summarise
from .fresh_maps import is_detour, shortest_path
from .jepa import decide
from .latent_battery import MODES, latent_positions
from .route_search import FrozenLakeEnvironment
from .runtime import json_write, seed_all
from .serialization import encode
from .training import amp, to_device, train_stage

ARMS = {"e2": 2, "eb2": MODES["bottleneck"]}  # the stage each arm's first phase trains in
ROUND1 = ("heldout", "test_detour")  # round 1's held-out populations
MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)  # ridge strengths, as scripts/analyze_latent_battery.py


def problems(data_root, split, indices=None):
    rows = records(data_root, split)
    keep = range(len(rows)) if indices is None else indices
    return [{"id": f"{split}/{i}", "split": split, "row_index": i, "row": rows[i]} for i in keep]


def require_round1_checkpoint(path, config, settings):
    """E: the final weights of round 1's fine-tune on found routes (job 6961875), saved by
    `detour_scaling.run_arm` with round 1's recipe."""
    progress = json.loads((Path(path) / "complete.json").read_text())["progress"]
    if (progress.get("pilot") or progress.get("stage") != 2 or progress.get("epoch") != settings["epochs"]
            or progress.get("next_row") != 0 or progress.get("optimizer_steps") != settings["round1_steps"]):
        raise ValueError("Expected round 1's completed fine-tune (E), not another checkpoint")
    expected = config | {"epochs_per_stage": settings["epochs"], "learning_rate": settings["learning_rate"],
                         "warmup_ratio": settings["warmup_ratio"]}
    for key, value in expected.items():
        if key not in {"local_model_path", "checkpoint_every_steps"} and progress["config"].get(key) != value:
            raise ValueError(f"Checkpoint recipe mismatch: {key}")


@contextlib.contextmanager
def masked(model):
    """The Stage 1 mask at test time: the answer sees the image only through the latents."""
    stage = model.config.livr_stage
    model.config.livr_stage = MODES["bottleneck"]
    try:
        yield
    finally:
        model.config.livr_stage = stage


# Precondition (CPU) ---------------------------------------------------------------------------

BATCHING_SIZES = {3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 3}  # 16 requests from different maps
PREFIXES = ("", "R", "DR", "DDR", "RRD", "DRDR")
EQUAL = 4  # size-7 maps read together with the empty prefix: a batch without padding


def batching_check(model, processor, data_root, split, output):
    """Batched against single reads of the same 16 requests, in full precision and under
    bfloat16 autocast. Also a batch of four equal-length rows, which needs no padding."""
    rows = records(data_root, split)
    chosen = [i for size, count in BATCHING_SIZES.items()
              for i in [k for k, r in enumerate(rows) if len(r["map_desc"]) == size][:count]]
    prompts = [encode(processor, data_root, split, rows[i], training=False) for i in chosen]
    prefixes = [PREFIXES[k % len(PREFIXES)] for k in range(len(chosen))]
    equal = [k for k, i in enumerate(chosen) if len(rows[i]["map_desc"]) == 7][:EQUAL - 1]
    extra = [i for i, r in enumerate(rows) if len(r["map_desc"]) == 7 and i not in chosen][:EQUAL - len(equal)]
    equal_prompts = [prompts[k] for k in equal] + [encode(processor, data_root, split, rows[i], training=False)
                                                   for i in extra]
    tokenizer = processor.tokenizer
    model.eval()
    result = {"split": split, "rows": chosen, "prefixes": prefixes, "equal_rows": [chosen[k] for k in equal] + extra}
    for precision in ("fp32", "bf16"):
        context = (contextlib.nullcontext() if precision == "fp32"
                   else torch.autocast(model.device.type, dtype=torch.bfloat16))
        started = time.time()
        with context, torch.no_grad():
            batched = batched_decide(model, tokenizer, prompts, prefixes)
            single = [decide(model, tokenizer, p, prefix, 0)[0] for p, prefix in zip(prompts, prefixes)]
            together = batched_decide(model, tokenizer, equal_prompts, [""] * len(equal_prompts))
            alone = [decide(model, tokenizer, p, "", 0)[0] for p in equal_prompts]
        differences = [max(abs(b[k] - s[k]) for k in s) for b, s in zip(batched, single)]
        result[precision] = {
            "max_difference": max(differences), "differences": differences,
            "by_size": {str(n): max(d for d, i in zip(differences, chosen) if len(rows[i]["map_desc"]) == n)
                        for n in BATCHING_SIZES},
            "equal_length_max_difference": max(max(abs(b[k] - s[k]) for k in s) for b, s in zip(together, alone)),
            "seconds": time.time() - started}
        print(json.dumps({precision: {k: v for k, v in result[precision].items() if k != "differences"}}), flush=True)
    json_write(Path(output) / "batching-check.json", result)
    return result


# Evaluation ---------------------------------------------------------------------------------

def populations(data_root, settings):
    """(population, split, row_index, row) for every unmasked greedy evaluation. Round 1's
    held-out maps come first, so the loading check can run before the rest."""
    old = [("heldout", settings["round1_heldout_split"], i, r)
           for i, r in enumerate(records(data_root, settings["round1_heldout_split"]))]
    test = [("test_detour", "test", i, r) for i, r in enumerate(records(data_root, "test")) if is_detour(r["map_desc"])]
    new = [("heldout78", settings["heldout_split"], i, r)
           for i, r in enumerate(records(data_root, settings["heldout_split"]))]
    seen = [("seen78", settings["pool_split"], i, r)
            for i, r in enumerate(records(data_root, settings["pool_split"])[:settings["seen_maps"]])]
    return old + test + new + seen


def state_items(data_root, settings):
    return ([("pool78", settings["pool_split"], i, r) for i, r in enumerate(records(data_root, settings["pool_split"]))]
            + [("heldout78", settings["heldout_split"], i, r)
               for i, r in enumerate(records(data_root, settings["heldout_split"]))])


@torch.no_grad()
def latent_states(model, processor, data_root, items, layers, path):
    """The four latent tokens' hidden states at decoder `layers` (one-based outputs, as in the
    latent battery), from an unmasked prompt pass: [maps, layers, 4, hidden], half precision."""
    if model.config.livr_stage != 2:
        raise ValueError("Latent states are read unmasked")
    states = []
    for count, (population, split, index, row) in enumerate(items):
        prompt = to_device(encode(processor, data_root, split, row, training=False), model.device)
        positions = latent_positions(prompt["input_ids"], model.config.livr_latent_ids)
        with amp(model.device):
            hidden = model(**prompt, output_hidden_states=True, use_cache=False).hidden_states
        states.append(torch.stack([hidden[layer][0, positions].float() for layer in layers]).half().cpu())
        if count % 100 == 0:
            print(f"states {count + 1}/{len(items)}", flush=True)
    torch.save({"layers": list(layers), "states": torch.stack(states),
                "items": [{"population": p, "split": s, "row_index": i, "map_desc": r["map_desc"]}
                          for p, s, i, r in items]}, path)


def evaluate(model, processor, evaluator, data_root, output, settings, loading=None):
    """Unmasked greedy on every population, masked greedy on the new held-out maps, and the
    latent states. `loading(greedy)`, if given, checks round 1's populations first and stops
    the evaluation when it fails."""
    model.eval()
    model.config.livr_stage = 2
    items = populations(data_root, settings)
    first = [item for item in items if item[0] in ROUND1]
    with (output / "greedy.jsonl").open("w", encoding="utf-8") as log:
        greedy = greedy_eval(model, processor, evaluator, data_root, first, log, settings["max_new_tokens"])
        check = loading(greedy) if loading else None
        if check is not None and not check["valid"]:
            return {"loading": check}
        greedy += greedy_eval(model, processor, evaluator, data_root, [i for i in items if i[0] not in ROUND1], log,
                              settings["max_new_tokens"])
    held = [item for item in items if item[0] == "heldout78"]
    with masked(model), (output / "masked.jsonl").open("w", encoding="utf-8") as log:
        bottleneck = greedy_eval(model, processor, evaluator, data_root, held, log, settings["max_new_tokens"])
    latent_states(model, processor, data_root, state_items(data_root, settings), settings["state_layers"],
                  output / "states.pt")
    counts = {p: [sum(r["success"] for r in greedy if r["population"] == p), sum(r["population"] == p for r in greedy)]
              for p in dict.fromkeys(r["population"] for r in greedy)}
    return {"loading": check, "greedy": counts, "masked": [sum(r["success"] for r in bottleneck), len(bottleneck)]}


def loading_check(round1_greedy, agreement):
    """Validity check 4: E's outcomes on round 1's held-out maps against round 1's."""
    recorded = {(r["population"], r["row_index"]): bool(r["success"]) for r in jsonl(round1_greedy)
                if r["population"] in ROUND1}

    def check(greedy):
        same = sum(recorded.get((r["population"], r["row_index"])) == bool(r["success"]) for r in greedy)
        return {"same_outcome": same, "maps": len(recorded), "agreement": agreement,
                "valid": len(greedy) == len(recorded) and same >= agreement}
    return check


# Search --------------------------------------------------------------------------------------

def step_zero_differences(policy, chosen):
    """Validity: batched first-move reads against one pass each, computed in this job."""
    batched = policy.many([(p, "") for p in chosen])
    out = []
    for problem, probs in zip(chosen, batched):
        prompt = {k: v.to(policy.model.device) for k, v in policy.prompt(problem).items()}
        single = decide(policy.model, policy.processor.tokenizer, prompt, "", 0)[0]
        out.append(max(abs(single[k] - probs[k]) for k in single))
    return out


def cached_differences(policy, chosen, rng, count=SAMPLE):
    """Validity: cached batched reads for `chosen` problems against one pass each."""
    ids = {p["id"] for p in chosen}
    keys = sorted(k for k in policy.cache if k[0] in ids)
    by_id = {p["id"]: p for p in chosen}
    out = []
    for problem_id, prefix in rng.sample(keys, min(count, len(keys))):
        prompt = {k: v.to(policy.model.device) for k, v in policy.prompt(by_id[problem_id]).items()}
        single = decide(policy.model, policy.processor.tokenizer, prompt, prefix, 0)[0]
        out.append(max(abs(single[k] - policy.cache[(problem_id, prefix)][k]) for k in single))
    return out


def run_search(model, processor, evaluator, data_root, output, settings, run_manifest, round1_routes,
               round1_greedy):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    model.eval()
    model.requires_grad_(False)
    seed_all(settings["seed"])
    json_write(output / "manifest.json", run_manifest | {"settings": settings, "round1_routes": str(round1_routes),
                                                         "round1_greedy": str(round1_greedy)})
    sets = check_sets(evaluator, data_root, settings)
    evaluation = evaluate(model, processor, evaluator, data_root, output, settings,
                          loading_check(round1_greedy, settings["loading_agreement"]))
    if not evaluation["loading"]["valid"] or not sets["valid"]:
        json_write(output / "summary.json", {"checks": {"sets": sets, "loading": evaluation["loading"], "valid": False}})
        raise ValueError("A validity check failed before the search; see summary.json")

    env = FrozenLakeEnvironment(evaluator)
    unsolved = sorted(r["row_index"] for r in jsonl(round1_routes) if not r["solved"])
    frontier = problems(data_root, settings["round1_pool_split"], unsolved)
    pool = problems(data_root, settings["pool_split"])
    if len(pool) != settings["maps"]:
        raise ValueError(f"The pool has {len(pool)} maps, not {settings['maps']}")
    policy = BatchedLivr(model, processor, data_root, settings["batch"])
    rng = random.Random(settings["seed"])
    chosen = rng.sample(pool, settings["check_maps"])

    # Validity check 1, as in round 1, with single passes computed here: before the search,
    # and on cached reads after the frontier search. One request per pass if either fails.
    checks = {"batch": settings["batch"]}
    first = step_zero_differences(policy, chosen)
    frontier_rows, frontier_cost = searched(env, policy, frontier, settings)
    after = cached_differences(policy, frontier, rng)
    checks |= {"step_zero_max_difference": max(first), "single_max_difference": max(after)}
    agrees = max(first) <= settings["probe_tolerance"] and max(after) <= settings["probe_tolerance"]
    if not agrees and settings["batch"] > 1:
        checks["batched_failed"] = {k: checks[k] for k in ("step_zero_max_difference", "single_max_difference")}
        policy.reset(1)
        first = step_zero_differences(policy, chosen)
        frontier_rows, frontier_cost = searched(env, policy, frontier, settings)
        checks |= {"batch": 1, "step_zero_max_difference": max(first), "single_max_difference": None}
        agrees = max(first) <= settings["probe_tolerance"]
    checks["policy_agrees"] = agrees
    with (output / "frontier.jsonl").open("w", encoding="utf-8") as log:
        log.writelines(json.dumps(r) + "\n" for r in frontier_rows)
    front = summarise(frontier_rows, settings["budget"]) | {"cost": frontier_cost}
    json_write(output / "summary-before-pool.json", {"checks": checks, "frontier": front, "evaluation": evaluation})

    rows, cost = searched(env, policy, pool, settings)
    with (output / "routes.jsonl").open("w", encoding="utf-8") as log:
        log.writelines(json.dumps(r) + "\n" for r in rows)
    summary = {"checks": checks | {"sets": sets, "loading": evaluation["loading"], "valid": agrees},
               "evaluation": evaluation, "frontier": front, "pool": summarise(rows, settings["budget"]) | {"cost": cost},
               "seconds": time.time() - began,
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    json_write(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("checks", "frontier", "pool")}, indent=1), flush=True)
    return summary


def livr_greedy(model, processor, evaluator, data_root, output, settings):
    """L's unmasked greedy answers on the new held-out maps, written beside E's."""
    output = Path(output)
    began = time.time()
    model.eval()
    model.config.livr_stage = 2
    seed_all(settings["seed"])
    held = [item for item in populations(data_root, settings) if item[0] == "heldout78"]
    with (output / "livr-greedy.jsonl").open("w", encoding="utf-8") as log:
        greedy = greedy_eval(model, processor, evaluator, data_root, held, log, settings["max_new_tokens"])
    summary = {"heldout78": [sum(r["success"] for r in greedy), len(greedy)], "seconds": time.time() - began}
    json_write(output / "livr-summary.json", summary)
    return summary


# Fine-tuning ---------------------------------------------------------------------------------

def found(pool, route):
    row = pool[route["row_index"]]
    if row.get("map_id") != route["map_id"]:
        raise ValueError(f"Route {route['row_index']} does not belong to this pool")
    return row | {"text_output": route["answer"]}


def training_rows(data_root, settings, round1_routes, search_run):
    """The released training maps; `extra_new` examples drawn in turn from the solved new pool
    maps; and `extra_old` from every solved round 1 pool map (round 1's routes and the
    frontier maps E's search solved), each in pool order."""
    new = [r for r in jsonl(Path(search_run) / "routes.jsonl") if r["solved"]]
    if len(new) < settings["variety_floor"]:
        raise ValueError(f"The search solved {len(new)} new maps, below the variety floor of "
                         f"{settings['variety_floor']}; the fine-tunes are held for a decision")
    old = {r["row_index"]: r for r in jsonl(round1_routes) if r["solved"]}
    frontier = {r["row_index"]: r for r in jsonl(Path(search_run) / "frontier.jsonl") if r["solved"]}
    if old.keys() & frontier.keys():
        raise ValueError("A frontier map was already solved in round 1")
    new_pool, old_pool = records(data_root, settings["pool_split"]), records(data_root, settings["round1_pool_split"])
    new_rows = [found(new_pool, r) for r in new]
    old_rows = [found(old_pool, (old | frontier)[i]) for i in sorted(old | frontier)]
    extra = ([(settings["pool_split"], new_rows[i % len(new_rows)]) for i in range(settings["extra_new"])]
             + [(settings["round1_pool_split"], old_rows[i % len(old_rows)]) for i in range(settings["extra_old"])])
    data = [("train", row) for row in records(data_root, "train")] + extra
    return data, {"new_maps": len(new_rows), "old_maps": len(old_rows), "frontier_maps": len(frontier),
                  "rows": len(data)}


def phases(data, settings, accumulation):
    """Shuffle, then split: the first `phase_one_rows` rows, and the rest."""
    cut = settings["phase_one_rows"]
    if cut % accumulation or not 0 < cut < len(data):
        raise ValueError("The first phase must be a whole number of optimizer steps, shorter than the data")
    order = list(range(len(data)))
    random.Random(settings["seed"]).shuffle(order)
    return [data[i] for i in order[:cut]], [data[i] for i in order[cut:]]


def loss_summary(path):
    steps = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return {"optimizer_steps": len(steps), "first_loss": steps[0]["loss"],
            "mean_loss": sum(s["loss"] for s in steps) / len(steps),
            "last_quarter_loss": sum(s["loss"] for s in steps[-max(1, len(steps) // 4):]) / max(1, len(steps) // 4)}


def run_arm(model, processor, evaluator, config, data_root, output, settings, arm, round1_routes, search_run,
            run_manifest):
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    checks = check_sets(evaluator, data_root, settings)
    data, counts = training_rows(data_root, settings, round1_routes, search_run)
    one, two = phases(data, settings, config["gradient_accumulation"])
    json_write(output / "manifest.json", run_manifest | {"arm": arm, "settings": settings, "checks": checks,
                                                         "counts": counts, "round1_routes": str(round1_routes),
                                                         "search_run": str(search_run)})
    tuned = config | {"epochs_per_stage": settings["epochs"], "learning_rate": settings["learning_rate"],
                      "warmup_ratio": settings["warmup_ratio"]}
    started = time.time()
    train_stage(model, processor, data_root, output / "phase1", tuned, ARMS[arm], final_weights_only=True, data=one,
                save_final=False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    middle = time.time()
    train_stage(model, processor, data_root, output / "phase2", tuned, 2, final_weights_only=True, data=two)
    training = {"phase1": loss_summary(output / "phase1/steps.jsonl") | {"seconds": middle - started,
                                                                          "stage": ARMS[arm]},
                "phase2": loss_summary(output / "phase2/steps.jsonl") | {"seconds": time.time() - middle, "stage": 2}}
    seed_all(config["seed"])
    evaluation = evaluate(model, processor, evaluator, data_root, output, settings)
    run_benchmark(model, processor, config, data_root, output / "evaluation",
                  run_manifest | {"checkpoint": f"in-memory {output / 'phase2/final'}", "arm": arm})
    summary = {"arm": arm, "checks": checks, "counts": counts, "training": training, "evaluation": evaluation,
               "seconds": time.time() - began,
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    json_write(output / "summary.json", summary)
    print(json.dumps(summary, indent=1), flush=True)
    return summary


# Readings ------------------------------------------------------------------------------------

def ridge(train_x, train_y, test_x):
    """Ridge on standardised features via a thin SVD, strength by leave-one-out error: the
    latent battery's probe (scripts/analyze_latent_battery.py)."""
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
            best = (loo, lam, alpha)
    coef = vh.T @ ((s / (s2 + best[1]))[:, None] * uy)
    return z @ coef + y_mean, best[2]


def balanced(predicted, truth, weights):
    """Per-cell balanced accuracy, averaged over the cells where both classes occur, for each
    row of `weights` (how often each held-out map is counted). predicted, truth: [maps, cells]."""
    p, t = predicted.double(), truth.double()
    positives, negatives = weights @ t, weights @ (1 - t)
    hit = (weights @ (p * t)) / positives.clamp(min=1e-9)
    reject = (weights @ ((1 - p) * (1 - t))) / negatives.clamp(min=1e-9)
    valid = (positives > 0) & (negatives > 0)
    return ((hit + reject) / 2 * valid).sum(-1) / valid.sum(-1).clamp(min=1)


def grid_of(desc):
    n = len(desc)
    return n, [(r, c) for r in range(n) for c in range(n)]


def probe_predictions(states, layer):
    """Per grid size: hole predictions and truth on the held-out maps, from a probe fitted on
    the pool maps; and goal and start cell accuracy."""
    index = states["layers"].index(layer)
    items, x = states["items"], states["states"][:, index].flatten(1)
    out = {}
    for n in sorted({len(i["map_desc"]) for i in items}):
        fit = [k for k, i in enumerate(items) if i["population"] == "pool78" and len(i["map_desc"]) == n]
        test = [k for k, i in enumerate(items) if i["population"] == "heldout78" and len(i["map_desc"]) == n]
        if not fit or not test:
            continue
        _, cells = grid_of(items[fit[0]]["map_desc"])
        target = lambda ks, value: torch.tensor([[float(items[k]["map_desc"][r][c] == value) for r, c in cells]
                                                 for k in ks])
        holes, alpha = ridge(x[fit], target(fit, -1), x[test])
        result = {"predicted": holes > 0.5, "truth": target(test, -1) > 0.5, "alpha": alpha,
                  "rows": [items[k]["row_index"] for k in test]}
        for name, value in (("goal", 2), ("start", 1)):
            scores, _ = ridge(x[fit], target(fit, value), x[test])
            result[f"{name}_accuracy"] = float((scores.argmax(-1) == target(test, value).argmax(-1)).double().mean())
        out[n] = result
    return out


def probe_score(predictions):
    by_size = {n: float(balanced(p["predicted"], p["truth"], torch.ones(1, len(p["rows"]), dtype=torch.double))[0])
               for n, p in predictions.items()}
    return {"hole_balanced_accuracy": sum(by_size.values()) / len(by_size), "by_size": by_size,
            "goal_accuracy": {n: p["goal_accuracy"] for n, p in predictions.items()},
            "start_accuracy": {n: p["start_accuracy"] for n, p in predictions.items()}}


def bootstrap_difference(first, second, resamples, seed):
    """Paired bootstrap of the hole score's difference, first minus second: held-out maps
    resampled within each size, the probes' predictions fixed."""
    generator = torch.Generator().manual_seed(seed)
    totals = torch.zeros(resamples, dtype=torch.double)
    for n in first:
        if first[n]["rows"] != second[n]["rows"]:
            raise ValueError("The two arms' probes scored different maps")
        maps = len(first[n]["rows"])
        draws = torch.randint(maps, (resamples, maps), generator=generator)
        weights = torch.zeros(resamples, maps, dtype=torch.double).scatter_add_(
            1, draws, torch.ones(resamples, maps, dtype=torch.double))
        totals += (balanced(first[n]["predicted"], first[n]["truth"], weights)
                   - balanced(second[n]["predicted"], second[n]["truth"], weights))
    totals /= len(first)
    ordered = totals.sort().values
    return {"low": float(ordered[int(0.025 * resamples)]), "high": float(ordered[int(0.975 * resamples) - 1])}


def outcome(desc, route):
    """The reference rules: into a hole, the goal by a shortest or longer route, passes the
    goal and moves on, or stops short."""
    n = len(desc)
    r, c = next((r, c) for r in range(n) for c in range(n) if desc[r][c] == 1)
    passed = False
    for letter in route or "":
        dr, dc = MOVES[letter]
        if 0 <= r + dr < n and 0 <= c + dc < n:
            r, c = r + dr, c + dc
            if desc[r][c] == -1:
                return "hole"
            passed = passed or desc[r][c] == 2
    if desc[r][c] == 2:
        return "shortest" if len(route) == shortest_path(desc) else "longer"
    return "passes the goal" if passed else "stops short"


def wall_bumps(desc, route):
    n = len(desc)
    r, c = next((r, c) for r in range(n) for c in range(n) if desc[r][c] == 1)
    bumps = 0
    for letter in route:
        dr, dc = MOVES[letter]
        if 0 <= r + dr < n and 0 <= c + dc < n:
            r, c = r + dr, c + dc
        else:
            bumps += 1
    return bumps


def reading(result, took, better, worse, neither, valid):
    if not valid:
        return "withheld: the measurement failed a validity check"
    if took is False:
        return "the bottleneck did not take; no reading on perception"
    if result["p"] < ALPHA:
        return better if result["only_first"] > result["only_second"] else worse
    return neither


def compare(data_root, search_run, arms, livr_outcomes, e_outcomes, settings):
    """`search_run`: the search job's output (E's evaluation and L's greedy answers). `arms`:
    {"e2": run, "eb2": run}. `livr_outcomes`, `e_outcomes`: per-map benchmark outcomes of
    LIVR and of E, for the harm checks. The map layouts come from `data_root`."""
    search_run = Path(search_run)
    runs = {"e": search_run, **{name: Path(run) for name, run in arms.items()}}
    greedy = {name: {(r["population"], r["row_index"]): r for r in jsonl(run / "greedy.jsonl")}
              for name, run in runs.items()}
    keys = sorted(greedy["e"])
    if any(sorted(g) != keys for g in greedy.values()):
        raise ValueError("The arms evaluated different maps")
    success = {name: {k: bool(g[k]["success"]) for k in keys} for name, g in greedy.items()}
    held = [k for k in keys if k[0] == "heldout78"]
    old = [k for k in keys if k[0] in ROUND1]
    livr = {("heldout78", r["row_index"]): bool(r["success"]) for r in jsonl(search_run / "livr-greedy.jsonl")}
    if sorted(livr) != held:
        raise ValueError("L evaluated different held-out maps")
    sizes = {k: greedy["e"][k]["grid_size"] for k in keys}
    tables = {split: records(data_root, split) for split in {r["split"] for r in greedy["e"].values()}}
    desc = {k: tables[greedy["e"][k]["split"]][k[1]]["map_desc"] for k in keys}

    def compared(a, b, among=held, table=success):
        return paired(table[a], table[b], among) | {
            "by_size": {str(n): paired(table[a], table[b], [k for k in among if sizes[k] == n])
                        for n in sorted({sizes[k] for k in among})}}

    search = json.loads((search_run / "summary.json").read_text())
    summaries = {name: json.loads((Path(run) / "summary.json").read_text()) for name, run in arms.items()}
    valid = search["checks"]["valid"] and all(s["checks"]["valid"] for s in summaries.values())

    # Manipulation check: the hole probe at the gate layer, EB2 against E2.
    states = {name: torch.load(run / "states.pt") for name, run in runs.items()}
    probes = {layer: {name: probe_predictions(s, layer) for name, s in states.items()}
              for layer in settings["state_layers"]}
    scores = {layer: {name: probe_score(p) for name, p in by_arm.items()} for layer, by_arm in probes.items()}
    layer = settings["probe_layer"]
    difference = scores[layer]["eb2"]["hole_balanced_accuracy"] - scores[layer]["e2"]["hole_balanced_accuracy"]
    interval = bootstrap_difference(probes[layer]["eb2"], probes[layer]["e2"], settings["bootstrap"], settings["seed"])
    took = difference >= settings["took_margin"] and interval["low"] > 0

    primary, secondary = compared("eb2", "e2"), compared("e2", "e")
    masked_runs = {name: {("heldout78", r["row_index"]): bool(r["success"]) for r in jsonl(run / "masked.jsonl")}
                   for name, run in runs.items()}
    pool = records(data_root, settings["pool_split"])
    routes = {r["row_index"]: r for r in jsonl(search_run / "routes.jsonl")}
    solved = [r for r in routes.values() if r["solved"]]
    seen = [k for k in keys if k[0] == "seen78"]
    trained = [k for k in seen if routes.get(k[1], {}).get("solved")]
    livr_bench = {r["row_index"]: bool(r["success"]) for r in jsonl(livr_outcomes)}
    e_bench = {r["row_index"]: bool(r["success"]) for r in jsonl(e_outcomes)}
    test_detours = {k[1] for k in keys if k[0] == "test_detour"}
    rows = sorted(livr_bench)

    def harm(mine, other):
        return {"all": paired(mine, other, rows),
                "detour": paired(mine, other, [r for r in rows if r in test_detours]),
                "no_detour": paired(mine, other, [r for r in rows if r not in test_detours])}

    benchmark = {}
    for name, run in arms.items():
        mine = {r["row_index"]: bool(r["success"]) for r in jsonl(Path(run) / "evaluation/responses.jsonl")}
        benchmark[name] = {"vs_livr": harm(mine, livr_bench), "vs_e": harm(mine, e_bench)}
    return {
        "validity": {"search": search["checks"], "set_checks": {n: s["checks"]["valid"] for n, s in summaries.items()},
                     "valid": valid},
        "manipulation": {"layer": layer, "e2": scores[layer]["e2"], "eb2": scores[layer]["eb2"],
                         "difference": difference, "interval_95": interval, "margin": settings["took_margin"],
                         "took": took},
        "primary": primary | {"reading": reading(primary, took, "the bottleneck adds to expert iteration",
                                                 "the bottleneck harms expert iteration", "no detectable difference",
                                                 valid)},
        "secondary": secondary | {"reading": reading(secondary, None, "round 2 helps on larger maps",
                                                     "round 2 harms on larger maps", "no detectable difference",
                                                     valid)},
        "descriptive": {
            "held_out_success": {name: sum(success[name][k] for k in held) for name in runs}
            | {"l": sum(livr.values())},
            "masked": {name: {"unmasked": sum(success[name][k] for k in held), "masked": sum(m.values())}
                       for name, m in masked_runs.items()},
            "masked_eb2_vs_e2": compared("eb2", "e2", table=masked_runs),
            "probes": {str(n): by_arm for n, by_arm in scores.items()},
            "frontier": search["frontier"], "pool": search["pool"],
            "found_routes": {"maps": len(solved), "equal_to_label": sum(r["equals_label"] for r in solved),
                             "longer_than_label": sum(r["longer_than_label"] for r in solved),
                             "wall_bumps": sum(wall_bumps(pool[r["row_index"]]["map_desc"], r["route"]) > 0
                                               for r in solved)},
            "e_vs_l": paired(success["e"], livr, held),
            "round1_heldout_vs_e": {name: compared(name, "e", among=old) for name in arms},
            "benchmark": benchmark,
            "failures": {name: dict(Counter(outcome(desc[k], greedy[name][k]["route"]) for k in held))
                         for name in runs} | {"l": dict(Counter(
                             outcome(desc[k], r["route"]) for k, r in
                             ((("heldout78", r["row_index"]), r) for r in jsonl(search_run / "livr-greedy.jsonl"))))},
            "seen": {name: {"trained": sum(success[name][k] for k in trained),
                            "untrained": sum(success[name][k] for k in seen if k not in trained)} for name in runs}
            | {"maps": {"trained": len(trained), "untrained": len(seen) - len(trained)}},
            "training": {name: s["training"] for name, s in summaries.items()}},
        "seconds_in_jobs": {"search": search["seconds"], **{n: s["seconds"] for n, s in summaries.items()}}}
