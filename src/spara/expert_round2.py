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

Round 3 (docs/EXPERT_ITERATION_ROUND3_SPEC.md) runs the same stages one round on, with
configs/expert_round3.json: B (EB2's recipe rerun with its weights saved) is the policy,
round 2's unsolved pool maps are the frontier, and EB3 trains from B on the routes of every
round. `round3` (CPU) applies its readings.

The hole check (docs/HOLE_CHECK_SPEC.md): S3 is EB3 with every training answer led by a line
per move naming the cell and the holes around it. `hole_check` (CPU) applies its readings.

The hole check on the exact map (docs/HOLE_CHECK_TEXT_SPEC.md): S3T is S3 and O3 is EB3,
each with the map also given as text in every prompt. `text_check` (CPU) applies its readings.
- `signatures` and `benchmark_outcomes`: no-detour failure signatures on the benchmark
  (scripts/analyze_nodetour_failures.py).
"""
import contextlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
from mathruler.grader import extract_boxed_content

from .benchmark import run_benchmark
from .data import records
from .detour_scaling import ALPHA, check_sets, greedy_eval, jsonl, paired
from .expert_iteration import SAMPLE, BatchedLivr, batched_decide, searched, summarise
from .fresh_maps import find, is_detour, shortest_path
from .explore_probe import GREEDY
from .hole_check import (forced_answer, line_reading, probe_lines, read, reading_rows, reading_score,
                         with_hole_check)
from .jepa import decide
from .latent_battery import MODES, latent_positions
from .latent_rl import moves
from .route_search import FrozenLakeEnvironment
from .runtime import json_write, seed_all
from .serialization import encode, latent_kwargs
from .training import amp, to_device, train_stage

ARMS = {"e2": 2, "eb2": MODES["bottleneck"], "o2": 2, "t2": 2,  # the stage each arm's first phase trains in
        "eb3": MODES["bottleneck"], "s3": MODES["bottleneck"], "o3": MODES["bottleneck"],
        "s3t": MODES["bottleneck"], "s3r": MODES["bottleneck"], "g3": MODES["bottleneck"],
        "g3r": MODES["bottleneck"]}
# The perception ceiling: every prompt also states the map as text; T2 blanks the image, so
# the text is its only map. The value: whether the image is blanked.
TEXT_ARMS = {"o2": False, "t2": True, "o3": False, "s3t": False}
HOLE_CHECK_ARMS = {"s3", "s3t", "s3r", "g3", "g3r"}  # every training answer leads with the hole check
GLIMPSE_ARMS = {"g3", "g3r"}  # four latents after every line's cell, each seeing one neighbouring cell
READING_ARMS = {"s3r", "g3r"}  # a phase of reading rows first
CELLS = {0: "F", -1: "H", 1: "S", 2: "G"}
ROUND1 = ("heldout", "test_detour")  # round 1's held-out populations
MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)  # ridge strengths, as scripts/analyze_latent_battery.py


def problems(data_root, split, indices=None):
    rows = records(data_root, split)
    keep = range(len(rows)) if indices is None else indices
    return [{"id": f"{split}/{i}", "split": split, "row_index": i, "row": rows[i]} for i in keep]


def require_base_checkpoint(path, config, settings):
    """The round's starting weights, saved with the fine-tuning recipe: E, round 1's fine-tune
    on found routes (375 steps), for round 2; B, the second phase of EB2's recipe (188 steps),
    for round 3. `base_steps` names the count."""
    progress = json.loads((Path(path) / "complete.json").read_text())["progress"]
    if (progress.get("pilot") or progress.get("stage") != 2 or progress.get("epoch") != settings["epochs"]
            or progress.get("next_row") != 0 or progress.get("optimizer_steps") != settings["base_steps"]):
        raise ValueError("Expected the round's completed base fine-tune, not another checkpoint")
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


def with_map_text(row, blank=False):
    """The ceiling arms' prompt (docs/PERCEPTION_CEILING_SPEC.md): the released prompt, with
    the map also written out as text after the image; `blank` replaces the image with a
    white one of the same size."""
    grid = "\n".join(" ".join(CELLS[cell] for cell in line) for line in row["map_desc"])
    before, after = row["text_input"].split("<image>")
    return row | {"text_input": f"{before}<image>\nThe same map as text, one row per line from top to bottom "
                                f"(S start, G goal, H hole, F safe):\n{grid}\n{after}",
                  "blank_image": blank}


def success_counts(greedy):
    return {p: [sum(r["success"] for r in greedy if r["population"] == p), sum(r["population"] == p for r in greedy)]
            for p in dict.fromkeys(r["population"] for r in greedy)}


def text_evaluate(model, processor, evaluator, data_root, output, settings, name, blank, among=None):
    """Unmasked greedy with the map also given as text, on every population or on `among`."""
    model.eval()
    model.config.livr_stage = 2
    items = [(p, s, i, with_map_text(r, blank)) for p, s, i, r in populations(data_root, settings)
             if among is None or p in among]
    with (output / f"{name}.jsonl").open("w", encoding="utf-8") as log:
        return success_counts(greedy_eval(model, processor, evaluator, data_root, items, log, settings["max_new_tokens"]))


def reading_eval(model, processor, evaluator, data_root, output, name, items, max_new_tokens):
    """The probe of S3R (docs/HOLE_CHECK_SPEC.md): greedy answers to one reading row per held-out map."""
    model.config.livr_stage = 2
    with (output / f"{name}.jsonl").open("w", encoding="utf-8") as log:
        answers = greedy_eval(model, processor, evaluator, data_root, items, log, max_new_tokens)
    counts = sum((reading_score(item[3], a["answer"]) for item, a in zip(items, answers)), Counter())
    return dict(counts) | {"capped": sum(a["new_tokens"] >= max_new_tokens for a in answers)}


@torch.no_grad()
def oracle_check(model, processor, evaluator, data_root, output, settings):
    """The oracle check (docs/HOLE_CHECK_SPEC.md): S3's greedy answers on the held-out maps, written line by
    line with nothing forced (which must reproduce its recorded answers), with the hole lists forced to the
    truth, and with the cells forced as well. A run cut short resumes: each pass keeps the maps it has written."""
    output, began = Path(output), time.time()
    model.eval()
    model.config.livr_stage = 2
    tokenizer, cap = processor.tokenizer, settings["max_new_tokens"]
    held = [item for item in populations(data_root, settings) if item[0] == "heldout78"]
    summary = {}
    for name, force in (("free", None), ("holes", "holes"), ("cell", "cell")):
        path = output / f"oracle-{name}.jsonl"
        done = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        solved = sum(record["success"] for record in done)
        with path.open("a", encoding="utf-8") as log:
            for population, split, index, row in held[len(done):]:
                prompt = to_device(encode(processor, data_root, split, row, training=False, **latent_kwargs(model)),
                                   model.device)

                def write(text, stop):
                    ids = prompt["input_ids"]
                    if text:  # the answer so far, tokenized apart from the prompt as in training
                        more = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
                        ids = torch.cat([ids, more.to(ids.device)], dim=-1)
                    room = cap - (ids.shape[-1] - prompt["input_ids"].shape[-1])
                    if room <= 0:
                        return ""
                    with amp(model.device):
                        out = model.generate(**(prompt | {"input_ids": ids, "attention_mask": torch.ones_like(ids)}),
                                             **GREEDY, max_new_tokens=room, use_cache=True, stop_strings=[stop],
                                             tokenizer=tokenizer)
                    new = tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)
                    return new[:new.index(stop) + len(stop)] if stop in new else new
                answer = forced_answer(row["map_desc"], write, force)
                scored = evaluator.score(row, answer)
                record = {"population": population, "split": split, "row_index": index,
                          "grid_size": len(row["map_desc"]), "route": moves(scored["extracted_answer"]),
                          "success": bool(scored["success"]), "answer": answer,
                          "new_tokens": len(tokenizer(answer, add_special_tokens=False)["input_ids"])}
                solved += record["success"]
                log.write(json.dumps(record) + "\n")
                log.flush()
        summary[name] = [solved, len(held)]
        print(json.dumps({name: summary[name], "seconds": time.time() - began}), flush=True)
    json_write(output / "oracle-summary.json", summary | {"seconds": time.time() - began})
    return summary


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
        prompt = to_device(encode(processor, data_root, split, row, training=False, **latent_kwargs(model)),
                           model.device)
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
    return {"loading": check, "greedy": success_counts(greedy),
            "masked": [sum(r["success"] for r in bottleneck), len(bottleneck)]}


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

def single_read(policy, problem, prefix):
    """One request in its own pass, at the policy's read precision (a one-row batch has no
    padding; round 1 found it equal to `jepa.decide` within 5e-7)."""
    return batched_decide(policy.model, policy.processor.tokenizer, [policy.prompt(problem)], [prefix],
                          policy.precision)[0]


def step_zero_differences(policy, chosen):
    """Validity: batched first-move reads against one pass each, computed in this job."""
    batched = policy.many([(p, "") for p in chosen])
    out = []
    for problem, probs in zip(chosen, batched):
        single = single_read(policy, problem, "")
        out.append(max(abs(single[k] - probs[k]) for k in single))
    return out


def cached_differences(policy, chosen, rng, count=SAMPLE):
    """Validity: cached batched reads for `chosen` problems against one pass each."""
    ids = {p["id"] for p in chosen}
    keys = sorted(k for k in policy.cache if k[0] in ids)
    by_id = {p["id"]: p for p in chosen}
    out = []
    for problem_id, prefix in rng.sample(keys, min(count, len(keys))):
        single = single_read(policy, by_id[problem_id], prefix)
        out.append(max(abs(single[k] - policy.cache[(problem_id, prefix)][k]) for k in single))
    return out


def run_search(model, processor, evaluator, data_root, output, settings, run_manifest, frontier_routes,
               reference_greedy):
    """`frontier_routes`: the previous round's search of `frontier_split`, whose unsolved maps
    are searched first. `reference_greedy`: the policy's recorded answers on round 1's
    held-out maps, for the loading check."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    model.eval()
    model.requires_grad_(False)
    seed_all(settings["seed"])
    json_write(output / "manifest.json", run_manifest | {"settings": settings, "frontier_routes": str(frontier_routes),
                                                         "reference_greedy": str(reference_greedy)})
    sets = check_sets(evaluator, data_root, settings)
    evaluation = evaluate(model, processor, evaluator, data_root, output, settings,
                          loading_check(reference_greedy, settings["loading_agreement"]))
    if not evaluation["loading"]["valid"] or not sets["valid"]:
        json_write(output / "summary.json", {"checks": {"sets": sets, "loading": evaluation["loading"], "valid": False}})
        raise ValueError("A validity check failed before the search; see summary.json")

    env = FrozenLakeEnvironment(evaluator)
    unsolved = sorted(r["row_index"] for r in jsonl(frontier_routes) if not r["solved"])
    frontier = problems(data_root, settings["frontier_split"], unsolved)
    pool = problems(data_root, settings["pool_split"])
    if len(pool) != settings["maps"]:
        raise ValueError(f"The pool has {len(pool)} maps, not {settings['maps']}")
    policy = BatchedLivr(model, processor, data_root, settings["batch"], settings["read_precision"])
    rng = random.Random(settings["seed"])
    chosen = rng.sample(pool, settings["check_maps"])

    # Validity check 1, as in round 1, with single passes at the same precision computed here:
    # before the search, and on cached reads after the frontier search. If either fails, round
    # 1's reads: one request per pass under bfloat16 autocast.
    checks = {"batch": settings["batch"], "read_precision": settings["read_precision"]}
    first = step_zero_differences(policy, chosen)
    frontier_rows, frontier_cost = searched(env, policy, frontier, settings)
    after = cached_differences(policy, frontier, rng)
    checks |= {"step_zero_max_difference": max(first), "single_max_difference": max(after)}
    agrees = max(first) <= settings["probe_tolerance"] and max(after) <= settings["probe_tolerance"]
    if not agrees and settings["batch"] > 1:
        checks["batched_failed"] = {k: checks[k] for k in ("step_zero_max_difference", "single_max_difference")}
        policy.reset(1, "bf16")
        first = step_zero_differences(policy, chosen)
        frontier_rows, frontier_cost = searched(env, policy, frontier, settings)
        checks |= {"batch": 1, "read_precision": "bf16", "step_zero_max_difference": max(first),
                   "single_max_difference": None}
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


def training_rows(data_root, settings, round1_routes, search_runs):
    """The released training maps; `extra_new` examples drawn in turn from every solved 7 x 7
    and 8 x 8 map; and `extra_old` from every solved map of round 1's pool (3 x 3 to 6 x 6).

    The routes come from round 1's search and from each later search in `search_runs`, in
    order: its frontier (`frontier_split`, the earlier rounds' unsolved maps) and its pool,
    as the run's manifest names them. Maps are taken pool by pool, each in pool order. The
    variety floor applies to the last search's pool."""
    sources = [(settings["round1_pool_split"], Path(round1_routes))]
    for run in map(Path, search_runs):
        used = json.loads((run / "manifest.json").read_text())["settings"]
        sources += [(used.get("frontier_split", used["round1_pool_split"]), run / "frontier.jsonl"),
                    (used["pool_split"], run / "routes.jsonl")]
    latest = sum(r["solved"] for r in jsonl(sources[-1][1]))
    if latest < settings["variety_floor"]:
        raise ValueError(f"The search solved {latest} new maps, below the variety floor of "
                         f"{settings['variety_floor']}; the fine-tunes are held for a decision")
    solved, by_source = {}, {}
    for split, path in sources:
        found_here = [route for route in jsonl(path) if route["solved"]]
        for route in found_here:
            if (split, route["row_index"]) in solved:
                raise ValueError(f"Map {route['row_index']} of {split} was already solved by an earlier search")
            solved[split, route["row_index"]] = route
        by_source[f"{path.parent.name}/{path.name}"] = len(found_here)
    pools = {split: records(data_root, split) for split, _ in sources}
    rows = [(split, found(pools[split], solved[split, i])) for split in pools
            for i in sorted(i for s, i in solved if s == split)]
    old_rows = [r for r in rows if r[0] == settings["round1_pool_split"]]
    new_rows = [r for r in rows if r[0] != settings["round1_pool_split"]]
    extra = ([new_rows[i % len(new_rows)] for i in range(settings["extra_new"])]
             + [old_rows[i % len(old_rows)] for i in range(settings["extra_old"])])
    data = [("train", row) for row in records(data_root, "train")] + extra
    return data, {"new_maps": len(new_rows), "old_maps": len(old_rows), "rows": len(data), "solved": by_source}


def replayed(data):
    """Every training answer's lines replay to its route with the map's hole sets; a row that
    does not stops the arm before training."""
    wrong = []
    for split, row in data:
        route = moves(extract_boxed_content(row["text_output"]))
        lines = read(row["map_desc"], row["text_output"], route)
        if not (lines["follows"] and lines["fatal"] is None and all(l["cell"] and l["holes"] for l in lines["lines"])):
            wrong.append((split, row.get("map_id")))
    if wrong:
        raise ValueError(f"{len(wrong)} hole-check answers do not replay, first {wrong[:3]}")
    return {"rows": len(data), "replay": len(data)}


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


def run_arm(model, processor, evaluator, config, data_root, output, settings, arm, round1_routes, search_runs,
            run_manifest):
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    checks = check_sets(evaluator, data_root, settings)
    data, counts = training_rows(data_root, settings, round1_routes, search_runs)
    text = arm in TEXT_ARMS
    if text:
        data = [(split, with_map_text(row, TEXT_ARMS[arm])) for split, row in data]
    if arm in HOLE_CHECK_ARMS:
        data = [(split, with_hole_check(row)) for split, row in data]
        checks = checks | {"hole_check_rows": replayed(data)}
    if arm in GLIMPSE_ARMS:  # a glimpse of each neighbouring cell after every line's cell
        from .glimpse import install, with_glimpses
        install(model, processor)
        data = [(split, with_glimpses(row)) for split, row in data]
    one, two = phases(data, settings, config["gradient_accumulation"])
    json_write(output / "manifest.json", run_manifest | {"arm": arm, "settings": settings, "checks": checks,
                                                         "counts": counts, "round1_routes": str(round1_routes),
                                                         "search_runs": [str(r) for r in search_runs]})
    if text:  # E, untrained on text maps, given them: the zero-shot read
        zero_shot = text_evaluate(model, processor, evaluator, data_root, output, settings, "zero-shot",
                                  TEXT_ARMS[arm], among={"heldout78"})
        seed_all(config["seed"])
    tuned = config | {"epochs_per_stage": settings["epochs"], "learning_rate": settings["learning_rate"],
                      "warmup_ratio": settings["warmup_ratio"]}
    def finish(training, evaluation):
        summary = {"arm": arm, "checks": checks, "counts": counts, "training": training, "evaluation": evaluation,
                   "seconds": time.time() - began,
                   "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
        json_write(output / "summary.json", summary)
        print(json.dumps(summary, indent=1), flush=True)
        return summary
    if arm in READING_ARMS:  # a reading phase on the pool maps first; then S3's phases, unchanged
        rng = random.Random(settings["seed"])
        zero = [(settings["pool_split"], r) for row in records(data_root, settings["pool_split"])
                for r in reading_rows(row, rng)]
        rng.shuffle(zero)
        probe = [("reading78", settings["heldout_split"], i, reading_rows(row, rng)[0])
                 for i, row in enumerate(records(data_root, settings["heldout_split"]))]
        if arm in GLIMPSE_ARMS:
            zero = [(split, with_glimpses(row)) for split, row in zero]
        began_reading = time.time()
        train_stage(model, processor, data_root, output / "phase0", tuned, 2, final_weights_only=True, data=zero,
                    save_final=False)
        reading_phase = loss_summary(output / "phase0/steps.jsonl") | {"seconds": time.time() - began_reading,
                                                                         "stage": 2, "rows": len(zero)}
        probes = {"after_phase0": reading_eval(model, processor, evaluator, data_root, output, "reading-phase0",
                                               probe, settings["max_new_tokens"])}
        if arm in GLIMPSE_ARMS:  # the ablation: the probe again with the written latents seeing no image token
            model.config.livr_glimpse["blind"] = True
            probes["after_phase0_blind"] = reading_eval(model, processor, evaluator, data_root, output,
                                                        "reading-phase0-blind", probe, settings["max_new_tokens"])
            del model.config.livr_glimpse["blind"]
        if settings.get("reading_only"):  # the reading test alone: no route phases
            return finish({"phase0": reading_phase}, {"reading": probes})
        seed_all(config["seed"])
    started = time.time()
    train_stage(model, processor, data_root, output / "phase1", tuned, ARMS[arm], final_weights_only=True, data=one,
                save_final=False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    middle = time.time()
    train_stage(model, processor, data_root, output / "phase2", tuned, 2, final_weights_only=True, data=two,
                save_final=settings.get("save_arm_weights", True))
    training = {"phase1": loss_summary(output / "phase1/steps.jsonl") | {"seconds": middle - started,
                                                                          "stage": ARMS[arm]},
                "phase2": loss_summary(output / "phase2/steps.jsonl") | {"seconds": time.time() - middle, "stage": 2}}
    seed_all(config["seed"])
    if text:  # a ceiling, not a deployable model: no masked read, states or benchmark
        evaluation = {"zero_shot": zero_shot,
                      "greedy": text_evaluate(model, processor, evaluator, data_root, output, settings, "greedy",
                                              TEXT_ARMS[arm])}
    else:
        evaluation = evaluate(model, processor, evaluator, data_root, output, settings)
        run_benchmark(model, processor, config, data_root, output / "evaluation",
                      run_manifest | {"checkpoint": f"in-memory {output / 'phase2/final'}", "arm": arm})
    if arm in GLIMPSE_ARMS:  # the ablation: the same answers with the written latents seeing no image token
        model.config.livr_glimpse["blind"] = True
        held = [item for item in populations(data_root, settings) if item[0] == "heldout78"]
        with (output / "blind.jsonl").open("w", encoding="utf-8") as log:
            blind = greedy_eval(model, processor, evaluator, data_root, held, log, settings["max_new_tokens"])
        del model.config.livr_glimpse["blind"]
        evaluation["blind"] = [sum(r["success"] for r in blind), len(blind)]
    if arm in READING_ARMS:
        training["phase0"] = reading_phase
        evaluation["reading"] = probes | {"after_phase2": reading_eval(
            model, processor, evaluator, data_root, output, "reading-phase2", probe, settings["max_new_tokens"])}
    return finish(training, evaluation)


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
                  "rows": [items[k]["row_index"] for k in test], "desc": [items[k]["map_desc"] for k in test]}
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


def walk(desc, route):
    """The cells a route visits before it ends or falls, and the hole it falls into (or None)."""
    n = len(desc)
    r, c = next((r, c) for r in range(n) for c in range(n) if desc[r][c] == 1)
    visited = [(r, c)]
    for letter in route or "":
        dr, dc = MOVES[letter]
        if 0 <= r + dr < n and 0 <= c + dc < n:
            r, c = r + dr, c + dc
            if desc[r][c] == -1:
                return visited, (r, c)
            visited.append((r, c))
    return visited, None


def hole_walks(predictions, routes):
    """Is a hole the model walks into one its latents read as safe? On each held-out map the
    probe scored and the arm answered (`routes`, keyed by held-out row), the probe's reading of
    the fatal hole, against its reading of the holes next to the cells the route visited and
    did not enter."""
    counts = {"fatal": [0, 0], "avoided": [0, 0]}
    maps = 0
    for n, p in predictions.items():
        for i, (row, desc) in enumerate(zip(p["rows"], p["desc"])):
            if row not in routes:
                continue
            maps += 1
            visited, fatal = walk(desc, routes[row])
            near = {(r + dr, c + dc) for r, c in visited for dr, dc in MOVES.values()
                    if 0 <= r + dr < n and 0 <= c + dc < n and desc[r + dr][c + dc] == -1} - {fatal}
            for kind, cells in (("fatal", [fatal] if fatal else []), ("avoided", near)):
                for r, c in cells:
                    counts[kind][0] += bool(p["predicted"][i, r * n + c])
                    counts[kind][1] += 1
    return {"maps": maps} | {kind: {"read_as_hole": read, "holes": total, "share": read / total if total else None}
                             for kind, (read, total) in counts.items()}


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
            "hole_walks": {name: hole_walks(probes[layer][name], {k[1]: greedy[name][k]["route"] for k in held})
                           for name in runs},
            "seen": {name: {"trained": sum(success[name][k] for k in trained),
                            "untrained": sum(success[name][k] for k in seen if k not in trained)} for name in runs}
            | {"maps": {"trained": len(trained), "untrained": len(seen) - len(trained)}},
            "training": {name: s["training"] for name, s in summaries.items()}},
        "seconds_in_jobs": {"search": search["seconds"], **{n: s["seconds"] for n, s in summaries.items()}}}


def ceiling(data_root, search_run, e2_run, o2_run, settings, t2_run=None):
    """The perception ceiling's readings (docs/PERCEPTION_CEILING_SPEC.md): O2, trained as E2
    but with the map also given as text, against E2 on the new held-out maps; and, given
    `t2_run`, T2 (the text map with the image blanked) against E2."""
    runs = {"e": Path(search_run), "e2": Path(e2_run), "o2": Path(o2_run)} | ({"t2": Path(t2_run)} if t2_run else {})
    greedy = {name: {(r["population"], r["row_index"]): r for r in jsonl(run / "greedy.jsonl")}
              for name, run in runs.items()}
    for name in TEXT_ARMS.keys() & runs.keys():
        greedy[f"{name}_zero_shot"] = {(r["population"], r["row_index"]): r for r in jsonl(runs[name] / "zero-shot.jsonl")}
    held = sorted(k for k in greedy["e"] if k[0] == "heldout78")
    old = sorted(k for k in greedy["e"] if k[0] in ROUND1)
    trained = [name for name in runs if name != "e"]
    rows = {name: json.loads((runs[name] / "summary.json").read_text())["counts"] for name in trained}
    valid = (all(held == sorted(k for k in g if k[0] == "heldout78") for g in greedy.values())
             and all(sorted(k for k in greedy[n] if k[0] in ROUND1) == old for n in trained)
             and all(rows[n] == rows["e2"] for n in trained))
    success = {name: {k: bool(r["success"]) for k, r in g.items()} for name, g in greedy.items()}
    sizes = {k: greedy["e"][k]["grid_size"] for k in held}
    desc = {("heldout78", i): r["map_desc"] for i, r in enumerate(records(data_root, settings["heldout_split"]))}

    def compared(a, b, among):
        result = paired(success[a], success[b], among)
        return result | {"gain": result["first"] - result["second"], "by_size": {
            str(n): paired(success[a], success[b], [k for k in among if sizes.get(k) == n])
            for n in sorted({sizes[k] for k in among if k in sizes})}}

    primary = compared("o2", "e2", held)
    if primary["p"] >= ALPHA:
        reading = "no detectable difference: choosing moves limits"
    elif primary["gain"] < 0:
        reading = "the text map harms"
    elif primary["gain"] >= settings["ceiling_margin"]:
        reading = "perfect perception helps a lot"
    else:
        reading = "perfect perception helps a little"
    result = {
        "validity": {"rows": rows, "valid": valid},
        "primary": primary | {"margin": settings["ceiling_margin"], "reading": reading},
        "descriptive": {
            "held_out_success": {name: sum(success[name][k] for k in held) for name in greedy},
            "o2_zero_shot_vs_e": compared("o2_zero_shot", "e", held),
            "round1_heldout_vs_e2": {n: paired(success[n], success["e2"], old) for n in trained if n != "e2"},
            "failures": {name: dict(Counter(outcome(desc[k], greedy[name][k]["route"]) for k in held))
                         for name in greedy},
        },
    }
    if t2_run:
        text_only = compared("t2", "e2", held)
        if text_only["p"] >= ALPHA:
            text_reading = "the policy plans from the text map as well as from the image"
        elif text_only["gain"] < 0:
            text_reading = "the policy plans worse from the text map: the ceiling is inconclusive"
        else:
            text_reading = "the policy plans better from the text map"
        result["text_only"] = text_only | {"reading": text_reading}
        result["descriptive"]["t2_vs_o2"] = compared("t2", "o2", held)
    return result


def round3(data_root, search_run, base_run, eb3_run, round2, livr_outcomes, settings):
    """Round 3's readings (docs/EXPERT_ITERATION_ROUND3_SPEC.md). `search_run`: B's search,
    with B's answers on every population; `base_run`: the job that trained B (EB2's recipe,
    rerun with its weights saved); `eb3_run`: round 3's arm. `round2`: {"e": round 2's search,
    "eb2": round 2's EB2}, compared on the held-out maps only. `livr_outcomes`: LIVR's
    per-map benchmark outcomes."""
    runs = {"b": Path(search_run), "eb3": Path(eb3_run)} | {name: Path(run) for name, run in round2.items()}
    greedy = {name: {(r["population"], r["row_index"]): r for r in jsonl(run / "greedy.jsonl")}
              for name, run in runs.items()}
    held = sorted(k for k in greedy["b"] if k[0] == "heldout78")
    old = sorted(k for k in greedy["b"] if k[0] in ROUND1)
    seen = sorted(k for k in greedy["b"] if k[0] == "seen78")
    success = {name: {k: bool(r["success"]) for k, r in g.items()} for name, g in greedy.items()}
    sizes = {k: greedy["b"][k]["grid_size"] for k in held}
    search = json.loads((Path(search_run) / "summary.json").read_text())
    base, eb3, eb2 = (json.loads((Path(run) / "summary.json").read_text()) for run in (base_run, eb3_run, round2["eb2"]))
    same_rows = all(base["counts"][k] == eb2["counts"][k] for k in ("new_maps", "old_maps", "rows"))
    valid = (search["checks"]["valid"] and base["checks"]["valid"] and eb3["checks"]["valid"] and same_rows
             and sorted(greedy["eb3"]) == sorted(greedy["b"])
             and all(held == sorted(k for k in g if k[0] == "heldout78") for g in greedy.values()))

    def compared(a, b, among=held):
        result = paired(success[a], success[b], among)
        if among is held:
            result["by_size"] = {str(n): paired(success[a], success[b], [k for k in held if sizes[k] == n])
                                 for n in sorted(set(sizes.values()))}
        return result

    primary = compared("eb3", "b")
    primary["reading"] = reading(primary, None, "round 3 helps", "round 3 harms", "no detectable difference", valid)
    base_greedy = {(r["population"], r["row_index"]): bool(r["success"]) for r in jsonl(Path(base_run) / "greedy.jsonl")}
    routes = {r["row_index"]: r for r in jsonl(Path(search_run) / "routes.jsonl")}
    trained = [k for k in seen if routes[k[1]]["solved"]]
    livr_bench = {r["row_index"]: bool(r["success"]) for r in jsonl(livr_outcomes)}
    test_detours = {k[1] for k in old if k[0] == "test_detour"}
    rows = sorted(livr_bench)
    bench = {name: {r["row_index"]: bool(r["success"]) for r in jsonl(Path(run) / "evaluation/responses.jsonl")}
             for name, run in (("b", base_run), ("eb3", eb3_run), ("eb2", round2["eb2"]))}

    def harm(mine, other):
        return {"all": paired(mine, other, rows),
                "detour": paired(mine, other, [r for r in rows if r in test_detours]),
                "no_detour": paired(mine, other, [r for r in rows if r not in test_detours])}
    round2_search = json.loads((Path(round2["e"]) / "summary.json").read_text())
    desc = {("heldout78", i): r["map_desc"] for i, r in enumerate(records(data_root, settings["heldout_split"]))}
    return {
        "validity": {"search": search["checks"], "base": base["checks"]["valid"], "eb3": eb3["checks"]["valid"],
                     "base_rows_equal_eb2": same_rows, "valid": valid},
        "primary": primary,
        "descriptive": {
            "held_out_success": {name: sum(success[name][k] for k in held) for name in runs},
            "eb3_vs_e": compared("eb3", "e"),
            "b_vs_eb2": compared("b", "eb2"),
            "b_rerun_same_outcome": sum(base_greedy[k] == success["eb2"][k] for k in held),
            "search": {"frontier": search["frontier"], "pool": search["pool"],
                       "round2_pool_by_e": round2_search["pool"]},
            "round1_heldout_eb3_vs_b": compared("eb3", "b", among=old),
            "benchmark": {"eb3_vs_b": harm(bench["eb3"], bench["b"]), "eb3_vs_livr": harm(bench["eb3"], livr_bench),
                          "b_vs_livr": harm(bench["b"], livr_bench), "b_vs_eb2": harm(bench["b"], bench["eb2"])},
            "failures": {name: dict(Counter(outcome(desc[k], greedy[name][k]["route"]) for k in held))
                         for name in runs},
            "seen": {name: {"trained": sum(success[name][k] for k in trained),
                            "untrained": sum(success[name][k] for k in seen if k not in trained)}
                     for name in ("b", "eb3")} | {"maps": {"trained": len(trained),
                                                          "untrained": len(seen) - len(trained)}},
            "training": {"b": base["training"], "eb3": eb3["training"], "eb2": eb2["training"]},
            "counts": {"b": base["counts"], "eb3": eb3["counts"]}},
        "seconds_in_jobs": {"base": base["seconds"], "search": search["seconds"], "eb3": eb3["seconds"]}}


NEXT_AFTER_NULL = {"missed": "run S3 with the map also given as text",
                   "listed and taken": "train on failures as well as successes",
                   "lost": "report and hold", "no line": "report and hold"}


def hole_check(data_root, s3_run, eb3_run, search_run, livr_outcomes, settings):
    """The hole check's readings (docs/HOLE_CHECK_SPEC.md). `s3_run` and `eb3_run` train from
    B on the same rows; `search_run` holds B's answers; `livr_outcomes`: LIVR's per-map
    benchmark outcomes. `settings`: configs/hole_check.json."""
    runs = {"s3": Path(s3_run), "eb3": Path(eb3_run), "b": Path(search_run)}
    greedy = {name: {(r["population"], r["row_index"]): r for r in jsonl(run / "greedy.jsonl")}
              for name, run in runs.items()}
    held = sorted(k for k in greedy["s3"] if k[0] == "heldout78")
    old = sorted(k for k in greedy["s3"] if k[0] in ROUND1)
    success = {name: {k: bool(r["success"]) for k, r in g.items()} for name, g in greedy.items()}
    sizes = {k: greedy["s3"][k]["grid_size"] for k in held}
    s3, eb3 = (json.loads((runs[name] / "summary.json").read_text()) for name in ("s3", "eb3"))
    capped = sum(greedy["s3"][k]["new_tokens"] >= settings["max_new_tokens"] for k in held)
    rows = s3["checks"].get("hole_check_rows", {})
    validity = {"s3": s3["checks"]["valid"], "eb3": eb3["checks"]["valid"],
                "rows_replay": rows.get("replay") == rows.get("rows") == s3["counts"]["rows"],
                "same_rows": all(s3["counts"][k] == eb3["counts"][k] for k in ("new_maps", "old_maps", "rows")),
                "same_maps": sorted(greedy["s3"]) == sorted(greedy["eb3"])
                and all(held == sorted(k for k in g if k[0] == "heldout78") for g in greedy.values()),
                "capped": capped, "cap_ok": capped <= 0.02 * len(held)}
    valid = all(v for k, v in validity.items() if k != "capped")

    def compared(a, b, among=held, table=success):
        result = paired(table[a], table[b], among)
        if among is held:
            result["by_size"] = {str(n): paired(table[a], table[b], [k for k in held if sizes[k] == n])
                                 for n in sorted(set(sizes.values()))}
        return result

    primary = compared("s3", "eb3")
    primary["reading"] = reading(primary, None, "writing the check helps", "writing the check harms",
                                 "no detectable difference", valid)
    desc = {("heldout78", i): r["map_desc"] for i, r in enumerate(records(data_root, settings["heldout_split"]))}
    lines = {k: read(desc[k], greedy["s3"][k]["answer"], greedy["s3"][k]["route"]) for k in held}
    fatal = Counter(r["fatal"] for r in lines.values() if r["fatal"])
    per_line = [line for r in lines.values() for line in r["lines"]]
    right_cell = [line for line in per_line if line["cell"]]
    if primary["reading"] == "no detectable difference" and fatal:
        largest = max(NEXT_AFTER_NULL, key=lambda kind: fatal[kind])  # ties go to the earlier class
        primary["next"] = NEXT_AFTER_NULL[largest]
    masked_runs = {name: {(r["population"], r["row_index"]): bool(r["success"]) for r in jsonl(runs[name] / "masked.jsonl")}
                   for name in ("s3", "eb3")}
    livr_bench = {r["row_index"]: bool(r["success"]) for r in jsonl(livr_outcomes)}
    bench_rows = sorted(livr_bench)
    test_detours = {k[1] for k in old if k[0] == "test_detour"}
    bench = {name: {r["row_index"]: bool(r["success"]) for r in jsonl(runs[name] / "evaluation/responses.jsonl")}
             for name in ("s3", "eb3")}

    def harm(mine, other):
        return {"all": paired(mine, other, bench_rows),
                "detour": paired(mine, other, [r for r in bench_rows if r in test_detours]),
                "no_detour": paired(mine, other, [r for r in bench_rows if r not in test_detours])}
    tokens = sorted(greedy["s3"][k]["new_tokens"] for k in held)
    return {
        "validity": validity | {"valid": valid},
        "primary": primary,
        "descriptive": {
            "held_out_success": {name: sum(success[name][k] for k in held) for name in runs},
            "fatal_moves": dict(fatal),
            "format": {"follows": sum(r["follows"] for r in lines.values()), "answers": len(lines)},
            "lines": {"lines": len(per_line), "cell_right": len(right_cell),
                      "holes_exact_given_cell": sum(line["holes"] for line in right_cell),
                      "missed": sum(line["missed"] for line in right_cell),
                      "invented": sum(line["invented"] for line in right_cell)},
            "s3_vs_b": compared("s3", "b"),
            "masked_s3_vs_eb3": compared("s3", "eb3", table=masked_runs),
            "round1_heldout_s3_vs_eb3": compared("s3", "eb3", among=old),
            "failures": {name: dict(Counter(outcome(desc[k], greedy[name][k]["route"]) for k in held)) for name in runs},
            "benchmark": {"s3_vs_eb3": harm(bench["s3"], bench["eb3"]), "s3_vs_livr": harm(bench["s3"], livr_bench)},
            "new_tokens": {"median": tokens[len(tokens) // 2], "max": tokens[-1]},
            "training": {"s3": s3["training"], "eb3": eb3["training"]}},
        "seconds_in_job": s3["seconds"]}


READS = 0.9  # the lines read the text map: the start named, and the hole directions listed, at this share
AGREE = 0.95  # two jobs' zero-shot reads of the same base on the same prompts agree at this share
TEXT_NEXT = {
    "helps": "image reading is the limit: train the model to transcribe the map from the image",
    "read right": "the check does not help even when read right: drop the form and keep EB3's",
    "read wrong": "the model cannot place itself or its neighbours from an exact text map: report and hold"}


def text_check(data_root, s3t_run, o3_run, s3_run, eb3_run, settings):
    """The readings of the hole check on the exact map (docs/HOLE_CHECK_TEXT_SPEC.md). `s3t_run`
    and `o3_run` train from B on EB3's rows with the map also given as text, with and without
    the hole check; `s3_run` and `eb3_run` are the same pair without the text map.
    `settings`: configs/hole_check_text.json."""
    runs = {"s3t": Path(s3t_run), "o3": Path(o3_run), "s3": Path(s3_run), "eb3": Path(eb3_run)}
    greedy = {name: {(r["population"], r["row_index"]): r for r in jsonl(run / "greedy.jsonl")}
              for name, run in runs.items()}
    held = sorted(k for k in greedy["s3t"] if k[0] == "heldout78")
    old = sorted(k for k in greedy["s3t"] if k[0] in ROUND1)
    success = {name: {k: bool(r["success"]) for k, r in g.items()} for name, g in greedy.items()}
    sizes = {k: greedy["s3t"][k]["grid_size"] for k in held}
    summaries = {name: json.loads((run / "summary.json").read_text()) for name, run in runs.items()}
    zero = {name: {r["row_index"]: bool(r["success"]) for r in jsonl(runs[name] / "zero-shot.jsonl")}
            for name in ("s3t", "o3")}
    agree = sum(zero["s3t"][k] == zero["o3"].get(k) for k in zero["s3t"])
    capped = sum(greedy["s3t"][k]["new_tokens"] >= settings["max_new_tokens"] for k in held)
    rows = summaries["s3t"]["checks"].get("hole_check_rows", {})
    validity = {"s3t": summaries["s3t"]["checks"]["valid"], "o3": summaries["o3"]["checks"]["valid"],
                "rows_replay": rows.get("replay") == rows.get("rows") == summaries["s3t"]["counts"]["rows"],
                "same_rows": all(summaries[name]["counts"][k] == summaries["eb3"]["counts"][k]
                                 for name in ("s3t", "o3") for k in ("new_maps", "old_maps", "rows")),
                "same_maps": all(sorted(g) == sorted(greedy["s3t"]) for g in greedy.values()),
                "zero_shot_agree": [agree, len(zero["s3t"])],
                "same_base": len(zero["s3t"]) == len(zero["o3"]) == len(held) and agree >= AGREE * len(held),
                "capped": capped, "cap_ok": capped <= 0.02 * len(held)}
    valid = all(v for k, v in validity.items() if k not in ("capped", "zero_shot_agree"))

    def compared(a, b, among=held):
        result = paired(success[a], success[b], among)
        if among is held:
            result["by_size"] = {str(n): paired(success[a], success[b], [k for k in held if sizes[k] == n])
                                 for n in sorted(set(sizes.values()))}
        return result

    desc = {("heldout78", i): r["map_desc"] for i, r in enumerate(records(data_root, settings["heldout_split"]))}
    lines = {name: line_reading((desc[k], greedy[name][k]["answer"], greedy[name][k]["route"]) for k in held)
             for name in ("s3t", "s3")}
    mine = lines["s3t"]
    shares = {"start_right": mine["first_line_offset"].get("0,0", 0) / len(held),
              "hole_directions_listed": (mine["lines"].get("hole_directions_listed", 0) / mine["lines"]["hole_directions"]
                                         if mine["lines"].get("hole_directions") else 0.0)}
    reads = all(share >= READS for share in shares.values())
    primary = compared("s3t", "o3")
    primary["reading"] = reading(primary, None, "the check helps when the map is exact",
                                 "the check harms even when the map is exact", "no detectable difference", valid)
    if valid:
        helps = primary["reading"] == "the check helps when the map is exact"
        primary["next"] = TEXT_NEXT["helps" if helps else "read right" if reads else "read wrong"]
    checked = {k: read(desc[k], greedy["s3t"][k]["answer"], greedy["s3t"][k]["route"]) for k in held}
    tokens = sorted(greedy["s3t"][k]["new_tokens"] for k in held)
    return {
        "validity": validity | {"valid": valid},
        "primary": primary,
        "lines_read_the_text": shares | {"threshold": READS, "reads": reads},
        "descriptive": {
            "held_out_success": {name: sum(success[name][k] for k in held) for name in runs},
            "s3t_vs_s3": compared("s3t", "s3"), "o3_vs_eb3": compared("o3", "eb3"),
            "s3t_vs_eb3": compared("s3t", "eb3"),
            "round1_heldout_s3t_vs_o3": compared("s3t", "o3", among=old),
            "fatal_moves": dict(Counter(r["fatal"] for r in checked.values() if r["fatal"])),
            "format": {"follows": sum(r["follows"] for r in checked.values()), "answers": len(checked)},
            "line_reading": lines,
            "failures": {name: dict(Counter(outcome(desc[k], greedy[name][k]["route"]) for k in held)) for name in runs},
            "zero_shot": {name: sum(zero[name].values()) for name in zero},
            "new_tokens": {"median": tokens[len(tokens) // 2], "max": tokens[-1]},
            "training": {name: summaries[name]["training"] for name in runs}},
        "seconds_in_jobs": {name: summaries[name]["seconds"] for name in ("s3t", "o3")}}
# No-detour failure signatures (scripts/analyze_nodetour_failures.py) ------------------------

def letters(answer):
    """The reference parser's view of an answer: every U, D, R or L character."""
    return "".join(ch for ch in (answer or "").upper() if ch in "UDRL")


def away_moves(desc, route):
    """Moves that increase the grid distance to the goal, up to the end of the route or the
    move into a hole. A move off the grid leaves the player in place and is not counted."""
    n = len(desc)
    (r, c), (gr, gc) = find(desc, 1), find(desc, 2)
    away = 0
    for letter in route or "":
        dr, dc = MOVES[letter]
        if not (0 <= r + dr < n and 0 <= c + dc < n):
            continue
        away += abs(r + dr - gr) + abs(c + dc - gc) > abs(r - gr) + abs(c - gc)
        r, c = r + dr, c + dc
        if desc[r][c] == -1:
            break
    return away


def signatures(descs, routes):
    """No-detour failure signatures over the maps in `descs` (key -> map_desc), from `routes`
    (key -> move letters): solved, into a hole, stops short or passes the goal, and failed
    routes with a move away from the goal."""
    kinds = {k: outcome(desc, routes[k]) for k, desc in descs.items()}
    failed = [k for k, kind in kinds.items() if kind not in ("shortest", "longer")]
    return {"maps": len(descs), "solved": len(descs) - len(failed), "failed": len(failed),
            "hole": sum(kinds[k] == "hole" for k in failed),
            "short_or_past": sum(kinds[k] in ("stops short", "passes the goal") for k in failed),
            "failed_with_away_move": sum(away_moves(descs[k], routes[k]) > 0 for k in failed)}


def benchmark_outcomes(path):
    """Per test row: (success, move letters). `path` is a run directory (its
    evaluation/responses.jsonl, or an exported benchmark-outcomes.jsonl) or a JSONL file."""
    path = Path(path)
    if path.is_dir():
        responses = path / "evaluation/responses.jsonl"
        path = responses if responses.exists() else path / "benchmark-outcomes.jsonl"
    return {r["row_index"]: (bool(r["success"]), letters(r.get("extracted_answer"))) for r in jsonl(path)}


READING_NEXT = {
    "helps": "the check becomes the policy's form: run the next round of expert iteration with it",
    "answers read": "the check does not help even when read right: drop the form and keep EB3's",
    "probe reads": "the reading is learned and not used in the answers: train the reading rows with the routes",
    "neither": "labels for every cell do not teach the holes next to a cell: report and hold"}


def reading_check(data_root, s3r_run, s3_run, eb3_run, search_run, livr_outcomes, settings):
    """S3R's readings (docs/HOLE_CHECK_SPEC.md): the hole check's readout with S3R in S3's place, so its
    "s3" keys are S3R; then the table's split on whether the probe and the answers read. `s3_run` is S3."""
    result = hole_check(data_root, s3r_run, eb3_run, search_run, livr_outcomes, settings)
    probes = json.loads((Path(s3r_run) / "summary.json").read_text())["evaluation"]["reading"]
    maps = records(data_root, settings["heldout_split"])
    answers = {name: {r["row_index"]: r for r in jsonl(Path(run) / "greedy.jsonl") if r["population"] == "heldout78"}
               for name, run in (("s3r", s3r_run), ("s3", s3_run))}
    lines = {name: line_reading((maps[i]["map_desc"], r["answer"], r["route"]) for i, r in sorted(found.items()))
             for name, found in answers.items()}

    def shares(start_right, count, total):
        return {"start_right": start_right / total,
                "hole_directions_listed": count.get("hole_directions_listed", 0) / max(1, count.get("hole_directions", 0))}
    probe = {when: shares(c.get("start_right", 0), c, c["answers"]) for when, c in probes.items()}
    by_line = {when: dict(sum((probe_lines(maps[r["row_index"]]["map_desc"], r["answer"])
                               for r in jsonl(Path(s3r_run) / f"reading-{when}.jsonl")), Counter()))
               for when in ("phase0", "phase2")}
    mine = lines["s3r"]
    in_answers = shares(mine["first_line_offset"].get("0,0", 0), mine["lines"], mine["answers"])
    probe_reads = all(share >= READS for share in probe["after_phase0"].values())
    answers_read = all(share >= READS for share in in_answers.values())
    validity = result["validity"]
    validity["same_maps_as_s3"] = sorted(answers["s3"]) == sorted(answers["s3r"])
    validity["probe_cap_ok"] = all(c["capped"] <= 0.02 * c["answers"] for c in probes.values())
    validity["valid"] = valid = validity["valid"] and validity["same_maps_as_s3"] and validity["probe_cap_ok"]
    primary = result["primary"]
    primary.pop("next", None)  # S3's rule for a null result is not S3R's
    primary["reading"] = reading(primary, None, "the check helps once the reading is trained", "the check harms",
                                 "no detectable difference", valid)
    if valid:
        helps = primary["reading"] == "the check helps once the reading is trained"
        primary["next"] = READING_NEXT["helps" if helps else "answers read" if answers_read
                                       else "probe reads" if probe_reads else "neither"]
    success = {name: {i: bool(r["success"]) for i, r in found.items()} for name, found in answers.items()}
    return result | {"reading": {"threshold": READS, "probe_reads": probe_reads, "answers_read": answers_read,
                                 "probe": probe, "answers": in_answers, "probe_counts": probes, "probe_by_line": by_line,
                                 "lines": lines,
                                 "s3r_vs_s3": paired(success["s3r"], success["s3"], sorted(success["s3r"]))}}


GLIMPSE_NEXT = {
    "helps": "glimpse latents work: let RL adjust them, and run a round of expert iteration with this form",
    "reads": "the reading is right and the routes are not: the limit is choosing the route over several moves",
    "neither": "the glimpses did not teach the lines in this much training: compare with the blind pass and hold"}


def by_map(run, name="greedy.jsonl"):
    return {r["row_index"]: r for r in jsonl(Path(run) / name) if r["population"] == "heldout78"}


def glimpse_check(data_root, g3_run, s3_run, eb3_run, search_run, livr_outcomes, settings):
    """G3's readings (docs/HOLE_CHECK_SPEC.md): the hole check's readout with G3 in S3's place, so its "s3"
    keys are G3; then the table's split on whether the lines naming the right cell list its holes."""
    result = hole_check(data_root, g3_run, eb3_run, search_run, livr_outcomes, settings)
    maps = records(data_root, settings["heldout_split"])
    answers = {"g3": by_map(g3_run), "blind": by_map(g3_run, "blind.jsonl"), "s3": by_map(s3_run)}
    lines = {name: line_reading((maps[i]["map_desc"], r["answer"], r["route"]) for i, r in sorted(found.items()))
             for name, found in answers.items()}
    listed = {name: {"all": read["lines"].get("hole_directions_listed", 0) / max(1, read["lines"].get("hole_directions", 0))}
              | {when: part["listed"] / max(1, part["holes"]) for when, part in read["by_line"].items()}
              for name, read in lines.items()}
    whole = sum(r["latents"] == 4 * r["answer"].count(" hole ") > 0 for r in answers["g3"].values())
    validity = result["validity"]
    validity["same_maps_as_s3"] = sorted(answers["s3"]) == sorted(answers["g3"]) == sorted(answers["blind"])
    validity["four_latents_a_line"] = [whole, len(answers["g3"])]
    validity["valid"] = valid = (validity["valid"] and validity["same_maps_as_s3"]
                                 and whole >= 0.98 * len(answers["g3"]))
    primary = result["primary"]
    primary.pop("next", None)  # S3's rule for a null result is not G3's
    primary["reading"] = reading(primary, None, "glimpse latents help", "glimpse latents harm",
                                 "no detectable difference", valid)
    reads = listed["g3"]["all"] >= READS
    if valid:
        primary["next"] = GLIMPSE_NEXT["helps" if primary["reading"] == "glimpse latents help"
                                       else "reads" if reads else "neither"]
    success = {name: {i: bool(r["success"]) for i, r in found.items()} for name, found in answers.items()}
    order = sorted(success["g3"])
    return result | {"glimpse": {"threshold": READS, "lines_read": reads, "hole_directions_listed": listed,
                                 "lines": lines, "g3_vs_s3": paired(success["g3"], success["s3"], order),
                                 "g3_vs_blind": paired(success["g3"], success["blind"], order)}}


GLIMPSE_READING_NEXT = {
    "reads": "glimpse latents read under labels for every cell, so G3 lacked supervision: run the reading phase "
             "and then S3's phases with glimpses, against EB3",
    "reads in part": "glimpse latents read part of the map under labels for every cell: report by direction and "
                     "against the blind probe, and hold",
    "does not read": "glimpse latents do not read even under labels for every cell: the form is the limit; "
                     "report and hold"}


def glimpse_reading(data_root, g3r_run, s3r_run, settings):
    """G3R's readings (docs/HOLE_CHECK_SPEC.md): after a reading phase with glimpses, the share of hole
    directions the probe lists on lines for cells that only their coordinates name, with the blind probe
    and S3R's probe beside it."""
    maps = records(data_root, settings["heldout_split"])
    answers = list(jsonl(Path(g3r_run) / "reading-phase0.jsonl"))

    def lines(run, name):
        return dict(sum((probe_lines(maps[r["row_index"]]["map_desc"], r["answer"])
                         for r in jsonl(Path(run) / name)), Counter()))

    def shares(count):
        listed = lambda key: count.get(f"{key}listed", 0) / max(1, count.get(f"{key}hole_directions", 0))
        return {"start": listed("start_"), "named": listed("named_"), "by_direction": {
            d: count.get(f"named_listed_{d}", 0) / max(1, count.get(f"named_listed_{d}", 0)
                                                        + count.get(f"named_missed_{d}", 0)) for d in MOVES}}
    counts = {"g3r": lines(g3r_run, "reading-phase0.jsonl"), "blind": lines(g3r_run, "reading-phase0-blind.jsonl"),
              "s3r": lines(s3r_run, "reading-phase0.jsonl")}
    whole = sum(r["latents"] == 4 * r["answer"].count(" hole ") > 0 for r in answers)
    capped = sum(r["new_tokens"] >= settings["max_new_tokens"] for r in answers)
    valid = len(answers) == len(maps) and capped <= 0.02 * len(answers) and whole >= 0.98 * len(answers)
    named = shares(counts["g3r"])["named"]
    reading = "reads" if named >= READS else "reads in part" if named >= 0.5 else "does not read"
    return {"validity": {"probe_answers": [len(answers), len(maps)], "capped": capped,
                         "four_latents_a_line": [whole, len(answers)], "valid": valid},
            "primary": {"threshold": READS, "named_cells_listed": named,
                        "reading": reading if valid else "withheld: the measurement failed a validity check",
                        **({"next": GLIMPSE_READING_NEXT[reading]} if valid else {})},
            "descriptive": {"shares": {name: shares(count) for name, count in counts.items()}, "counts": counts,
                            "summary": json.loads((Path(g3r_run) / "summary.json").read_text())}}


ORACLE_NEXT = {
    True: "a right reading makes safe routes: build per-move reading latents; the forced count bounds them",
    False: "the policy does not turn right readings into routes: the limit is choosing the route over several moves"}


def oracle_read(data_root, oracle_run, s3_run, eb3_run, settings):
    """The oracle check's readings (docs/HOLE_CHECK_SPEC.md): S3 written line by line with nothing forced,
    the hole lists forced, and the cells forced as well, against EB3 and S3's recorded answers."""
    maps = records(data_root, settings["heldout_split"])
    runs = {name: by_map(oracle_run, f"oracle-{name}.jsonl") for name in ("free", "holes", "cell")}
    runs |= {"s3": by_map(s3_run), "eb3": by_map(eb3_run)}
    order = sorted(runs["eb3"])
    success = {name: {i: bool(r["success"]) for i, r in found.items()} for name, found in runs.items()}
    sizes = sorted({len(maps[i]["map_desc"]) for i in order})
    lines = {name: {i: read(maps[i]["map_desc"], runs[name][i]["answer"], runs[name][i]["route"]) for i in order}
             for name in ("free", "holes", "cell")}
    same = sum(runs["free"][i]["route"] == runs["s3"][i]["route"] for i in order)
    capped = {name: sum(runs[name][i]["new_tokens"] >= settings["max_new_tokens"] for i in order) for name in lines}
    right = sum(all(line["cell"] and line["holes"] for line in lines["cell"][i]["lines"]) for i in order)
    validity = {"same_maps": all(sorted(found) == order for found in runs.values()),
                "free_routes_as_s3": [same, len(order)], "capped": capped, "forced_lines_right": [right, len(order)]}
    valid = (validity["same_maps"] and same >= 0.95 * len(order) and right == len(order)
             and all(count <= 0.02 * len(order) for count in capped.values()))

    def compared(a, b):
        return paired(success[a], success[b], order) | {"by_size": {
            str(n): paired(success[a], success[b], [i for i in order if len(maps[i]["map_desc"]) == n]) for n in sizes}}
    primary = compared("cell", "eb3")
    primary["reading"] = reading(primary, None, "a right reading makes safe routes", "a right reading harms",
                                 "no detectable difference", valid)
    if valid:
        primary["next"] = ORACLE_NEXT[primary["reading"] == "a right reading makes safe routes"]
    return {"validity": validity | {"valid": valid}, "primary": primary,
            "descriptive": {
                "solved": {name: sum(found.values()) for name, found in success.items()},
                "holes_vs_eb3": compared("holes", "eb3"), "holes_vs_free": compared("holes", "free"),
                "cell_vs_free": compared("cell", "free"), "cell_vs_holes": compared("cell", "holes"),
                "failures": {name: dict(Counter(outcome(maps[i]["map_desc"], runs[name][i]["route"]) for i in order))
                             for name in runs},
                "fatal_moves": {name: dict(Counter(r["fatal"] for r in found.values() if r["fatal"]))
                                for name, found in lines.items()},
                "moves": {name: sorted(len(runs[name][i]["route"]) for i in order)[len(order) // 2] for name in runs}}}
