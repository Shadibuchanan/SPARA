"""Detour scaling check (docs/DETOUR_SCALING_SPEC.md): fine-tune LIVR's final Stage 2
weights on the released training maps plus 2,000 generated detour examples drawn from n
distinct layouts, then compare greedy success on held-out detour maps across n.

One arm per job (`run_arm`); `compare` reads every arm's output and applies the rules
fixed in the specification.
"""
import json
import statistics
import time
from pathlib import Path

import torch

from .benchmark import run_benchmark
from .data import layout_key, records
from .explore_probe import GREEDY, score
from .fresh_maps import is_detour
from .latent_battery import mcnemar
from .runtime import json_write, seed_all
from .serialization import encode
from .training import amp, to_device, train_stage

AGREEMENT, TOOK, ALPHA = 38, 0.9, 0.05  # validity: reproduced test routes, seen-map success; test level
HELD_OUT = ("heldout", "test_detour")


def training_data(data_root, recipe, distinct):
    """The released training maps plus `extra_examples` drawn in turn from the first
    `distinct` pool maps, so each appears equally often (to within one)."""
    pool = records(data_root, recipe["pool_split"])
    if not 0 < distinct <= len(pool):
        raise ValueError(f"The pool has {len(pool)} maps, not {distinct}")
    extra = [(recipe["pool_split"], pool[i % distinct]) for i in range(recipe["extra_examples"])]
    return [("train", row) for row in records(data_root, "train")] + extra


def check_sets(evaluator, data_root, recipe):
    """Validity checks 1 and 2: labels accepted by the reference evaluator, and no layout
    of the held-out set or the pool shared with each other, the other fresh sets or the
    released splits. (The released splits share 7 layouts between themselves, none of them
    a test detour map; that is a property of the benchmark, not checked here.)"""
    generated = (recipe["heldout_split"], recipe["pool_split"])
    labelled = [row for split in generated for row in records(data_root, split)]
    accepted = sum(bool(evaluator.score(row, row["text_output"])["success"]) for row in labelled)
    sets = {split: {layout_key(r) for r in records(data_root, split)}
            for split in [*generated, *recipe["other_fresh"], "train", "test"]}
    names = list(sets)
    shared = {f"{a}|{b}": len(sets[a] & sets[b]) for i, a in enumerate(names) for b in names[i + 1:]
              if a in generated}
    return {"labels_accepted": [accepted, len(labelled)], "shared_layouts": shared,
            "valid": accepted == len(labelled) and not any(shared.values())}


def populations(data_root, recipe):
    """(population, split, row_index, row) for every greedy evaluation."""
    test = [("test_detour", "test", i, r) for i, r in enumerate(records(data_root, "test")) if is_detour(r["map_desc"])]
    heldout = [("heldout", recipe["heldout_split"], i, r) for i, r in enumerate(records(data_root, recipe["heldout_split"]))]
    seen = [("seen", recipe["pool_split"], i, r)
            for i, r in enumerate(records(data_root, recipe["pool_split"])[:recipe["seen_maps"]])]
    return heldout + test + seen


@torch.no_grad()
def greedy_eval(model, processor, evaluator, data_root, items, log, max_new_tokens):
    model.eval()
    results = []
    for count, (population, split, index, row) in enumerate(items):
        prompt = to_device(encode(processor, data_root, split, row, training=False), model.device)
        with amp(model.device):
            generated = model.generate(**prompt, **GREEDY, max_new_tokens=max_new_tokens, use_cache=True)
        (route, success), = score(evaluator, processor, row, generated)
        record = {"population": population, "split": split, "row_index": index, "grid_size": len(row["map_desc"]),
                  "route": route, "success": success}
        results.append(record)
        log.write(json.dumps(record) + "\n"); log.flush()
        if count % 50 == 0:
            print(f"greedy {count + 1}/{len(items)}", flush=True)
    return results


def run_arm(model, processor, evaluator, config, data_root, output, arm, recipe, run_manifest, data=None,
            distinct=None, benchmark=True):
    """One arm. `data` (with its count of `distinct` pool maps) replaces the recipe's arm,
    for expert iteration (docs/EXPERT_ITERATION_SPEC.md); `benchmark` False skips the
    reference benchmark."""
    if data is None:
        distinct = recipe["arms"][arm]
        data = training_data(data_root, recipe, distinct) if distinct else None
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    checks = check_sets(evaluator, data_root, recipe)
    json_write(output / "manifest.json", run_manifest | {"arm": arm, "distinct": distinct, "recipe": recipe,
                                                         "checks": checks})
    training = None
    if distinct:
        tuned = config | {"epochs_per_stage": recipe["epochs"], "learning_rate": recipe["learning_rate"],
                          "warmup_ratio": recipe["warmup_ratio"]}
        started = time.time()
        train_stage(model, processor, data_root, output / "stage2", tuned, 2, final_weights_only=True, data=data)
        steps = [json.loads(line) for line in (output / "stage2/steps.jsonl").read_text().splitlines()]
        training = {"seconds": time.time() - started, "optimizer_steps": len(steps),
                    "mean_loss": statistics.mean(s["loss"] for s in steps),
                    "last_quarter_loss": statistics.mean(s["loss"] for s in steps[-max(1, len(steps) // 4):])}
    model.eval()
    seed_all(config["seed"])
    with (output / "greedy.jsonl").open("w", encoding="utf-8") as log:
        greedy = greedy_eval(model, processor, evaluator, data_root, populations(data_root, recipe), log,
                             recipe["max_new_tokens"])
    if distinct and benchmark:
        run_benchmark(model, processor, config, data_root, output / "evaluation",
                      run_manifest | {"checkpoint": f"in-memory {output / 'stage2/final'}", "arm": arm})
    summary = {"arm": arm, "distinct": distinct, "checks": checks, "training": training,
               "greedy": {p: [sum(r["success"] for r in greedy if r["population"] == p),
                              sum(r["population"] == p for r in greedy)] for p in (*HELD_OUT, "seen")},
               "seconds": time.time() - began,
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    json_write(output / "summary.json", summary)
    print(json.dumps(summary, indent=1), flush=True)
    return summary


# Comparison across arms ---------------------------------------------------------------

def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def paired(a, b, keys):
    """McNemar on success, a against b, over `keys`."""
    only_a = sum(a[k] and not b[k] for k in keys)
    only_b = sum(b[k] and not a[k] for k in keys)
    return {"maps": len(keys), "first": sum(a[k] for k in keys), "second": sum(b[k] for k in keys),
            "only_first": only_a, "only_second": only_b, "p": mcnemar(only_a, only_b)}


def verdict(result, better, worse, neither):
    if result["p"] < ALPHA:
        return better if result["only_first"] > result["only_second"] else worse
    return neither


def compare(runs, planning_routes, livr_outcomes):
    """`runs`: arm -> run directory for livr, n98, n500 and n2000. `planning_routes`: the
    planning probe's routes.jsonl; `livr_outcomes`: LIVR's per-map benchmark outcomes."""
    greedy = {arm: {(r["population"], r["row_index"]): r for r in jsonl(Path(run) / "greedy.jsonl")}
              for arm, run in runs.items()}
    summaries = {arm: json.loads((Path(run) / "summary.json").read_text()) for arm, run in runs.items()}
    keys = sorted(greedy["livr"])
    if any(sorted(g) != keys for g in greedy.values()):
        raise ValueError("The arms evaluated different maps")
    held = [k for k in keys if k[0] in HELD_OUT]
    success = {arm: {k: g[k]["success"] for k in keys} for arm, g in greedy.items()}
    sizes = {k: greedy["livr"][k]["grid_size"] for k in keys}

    def compared(a, b):
        return paired(success[a], success[b], held) | {
            "by_population": {p: paired(success[a], success[b], [k for k in held if k[0] == p]) for p in HELD_OUT},
            "by_size": {str(n): paired(success[a], success[b], [k for k in held if sizes[k] == n])
                        for n in sorted({sizes[k] for k in held})}}

    recorded = {r["row_index"]: r["route"] for r in jsonl(planning_routes) if r["split"] == "test"}
    test_keys = [k for k in keys if k[0] == "test_detour"]
    reproduced = sum(greedy["livr"][k]["route"] == recorded[k[1]] for k in test_keys)
    seen = {arm: sum(success[arm][k] for k in keys if k[0] == "seen") / sum(k[0] == "seen" for k in keys)
            for arm in runs}
    checks_valid = all(s["checks"]["valid"] for s in summaries.values())
    valid = checks_valid and reproduced >= AGREEMENT and seen["n98"] >= TOOK
    withheld = "withheld: the measurement failed a validity check"
    primary, secondary = compared("n2000", "n98"), compared("n98", "livr")
    livr_bench = {r["row_index"]: bool(r["success"]) for r in jsonl(livr_outcomes)}
    detour_rows = {k[1] for k in test_keys}
    benchmark = {}
    for arm in ("n98", "n500", "n2000"):
        mine = {r["row_index"]: bool(r["success"]) for r in jsonl(Path(runs[arm]) / "evaluation/responses.jsonl")}
        rows = sorted(livr_bench)
        benchmark[arm] = {"all": paired(mine, livr_bench, rows),
                          "detour": paired(mine, livr_bench, [r for r in rows if r in detour_rows]),
                          "no_detour": paired(mine, livr_bench, [r for r in rows if r not in detour_rows])}
    return {
        "validity": {"set_checks": checks_valid, "reproduced_test_routes": [reproduced, len(test_keys)],
                     "seen_success": seen, "valid": valid},
        "primary": primary | {"reading": withheld if not valid else verdict(
            primary, "variety helps", "variety hurts", "no detectable effect of variety")},
        "secondary": secondary | {"reading": withheld if not valid else verdict(
            secondary, "detour fine-tuning helps", "detour fine-tuning harms", "no detectable difference")},
        "descriptive": {"held_out_success": {arm: sum(success[arm][k] for k in held) for arm in runs},
                        "n500_vs_n98": compared("n500", "n98"), "n2000_vs_n500": compared("n2000", "n500"),
                        "benchmark_vs_livr": benchmark,
                        "training": {arm: summaries[arm]["training"] for arm in runs if summaries[arm]["training"]}},
        "gpu_seconds_in_jobs": {arm: summaries[arm]["seconds"] for arm in runs}}
