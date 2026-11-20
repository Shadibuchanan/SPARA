"""Latent distillation (docs/LATENT_DISTILLATION_SPEC.md): search LIVR's layer-24 latents per map
with the verifier, keep the pushes that work, and train the model to produce those latent states
(arm D) against the same found routes without the latent target (arm S).

- `run_search` (GPU): per pool map, LIVR's greedy answer, then greedy answers under pushes of
  increasing size in C″'s confirmed basis until one succeeds; every found push is replayed alone.
  Writes search.jsonl, targets.pt (LIVR's latent state z and the push delta per found map) and
  summary.json.
- `run_train` (GPU): one arm and seed: expert iteration's rows from the found routes, stage-2
  fine-tuning (CPU AdamW optionally), D's latent loss with its gradient set to a fixed share of
  the route gradient every step, then greedy answers on the held-out sets and the reference
  benchmark.
- `compare` (CPU): the readings fixed in the specification.
"""
import contextlib
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from .benchmark import run_benchmark
from .data import records
from .detour_scaling import check_sets, jsonl, paired
from .expert_iteration import training_rows
from .explore_probe import GREEDY
from .fair_rl import load_noise, sign_test, ALPHA
from .fresh_maps import answer_text, is_detour
from .latent_diagnostics import draw, generate_rows
from .latent_rl import LatentAdjuster
from .model import latent_queries
from .runtime import json_write, seed_all
from .serialization import encode
from .training import amp, to_device, train_stage

ARMS = ("S", "D")
PRIMARY = ("heldout", "test_detour")  # round 1's 240 held-out detour maps
TRANSFER = ("heldout78",)


def prompt_of(processor, data_root, split, row, device):
    return to_device(encode(processor, data_root, split, row, training=False), device)


def basis_check(family):
    """The basis's directions are orthonormal per slot, so the projection in `LatentTarget` reads
    the push's coordinates exactly."""
    directions = family["directions"].double()
    gram = torch.einsum("sdk,sdj->skj", directions, directions)
    return float((gram - torch.eye(gram.shape[-1], dtype=gram.dtype)).abs().max())


# Search ------------------------------------------------------------------------------------------

@torch.no_grad()
def capture_state(model, adjuster, prompt):
    """LIVR's latent state z (K, d), float32, at the adjuster's layer, from one prefill."""
    with adjuster.capture() as state, amp(model.device):
        model(**prompt, use_cache=False)
    if len(state["priors"]) != 1:
        raise RuntimeError(f"Expected one latent capture, saw {len(state['priors'])}")
    return state["priors"][0][0].float()


@torch.no_grad()
def run_search(model, processor, evaluator, data_root, output, settings, basis_path, run_manifest, smoke=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    search, noise = settings["search"], settings["noise"]
    family = load_noise({"noise": noise}, basis_path)  # checks the sha256
    orthonormal = basis_check(family)
    maps = settings["smoke_maps"] if smoke else settings["maps"]
    split = settings["pool_split"]
    pool = records(data_root, split)[:maps]
    if len(pool) != maps:
        raise ValueError(f"The pool has fewer than {maps} maps")
    json_write(output / "manifest.json", run_manifest | {"settings": settings, "basis": str(basis_path),
                                                         "smoke": smoke})
    model.eval()
    model.config.livr_stage = 2
    model.config.use_cache = True
    adjuster = LatentAdjuster(model, noise["layer"])
    generator = torch.Generator(device=model.device).manual_seed(search["seed"])
    generation = GREEDY | {"max_new_tokens": settings["max_new_tokens"]}
    plain = lambda a, b: contextlib.nullcontext()
    rows, saved = [], {"row_index": [], "z": [], "delta": []}
    schedule = list(zip(search["scales"], search["smoke_draws"] if smoke else search["draws"]))
    with (output / "search.jsonl").open("w", encoding="utf-8") as log:
        for index, row in enumerate(pool):
            prompt = prompt_of(processor, data_root, split, row, model.device)
            z = capture_state(model, adjuster, prompt)
            (route, success), = generate_rows(model, processor, evaluator, row, prompt, generation, 1, plain, 1)
            record = {"row_index": index, "map_id": row.get("map_id"), "grid_size": len(row["map_desc"]),
                      "greedy_route": route, "greedy_success": success, "tries": 1, "scale": None, "eps": None,
                      "route": route if success else None}
            delta = torch.zeros_like(z) if success else None
            if not success:
                for scale, draws in schedule:
                    for start in range(0, draws, search["batch"]):
                        count = min(search["batch"], draws - start)
                        offsets, eps = draw(family, scale, count, generator, model.device)
                        cells = generate_rows(model, processor, evaluator, row, prompt, generation, count,
                                              lambda a, b, o=offsets: adjuster.offset(o[a:b]), count)
                        hit = next((i for i, (_, s) in enumerate(cells) if s), None)
                        if hit is None:
                            record["tries"] += count
                            continue
                        record["tries"] += hit + 1
                        record |= {"scale": scale, "eps": eps[hit].cpu().tolist(), "route": cells[hit][0]}
                        delta = offsets[hit].float()
                        break
                    if delta is not None:
                        break
            record["solved"] = delta is not None
            if record["solved"]:
                # Validity: the push alone (batch of one) reproduces the route.
                if record["scale"] is None:
                    record["replay_route"] = route  # the greedy pass already ran alone
                else:
                    with adjuster.offset(delta[None]):
                        (replayed, _), = generate_rows(model, processor, evaluator, row, prompt, generation, 1, plain, 1)
                    record["replay_route"] = replayed
                record["reproduced"] = record["replay_route"] == record["route"]
                record["answer"] = answer_text(record["route"])
                record["answer_accepted"] = bool(evaluator.score(row, record["answer"])["success"])
                record["push_ratio"] = (delta.norm(dim=-1) / z.norm(dim=-1)).cpu().tolist()
                if record["reproduced"] and record["answer_accepted"]:
                    saved["row_index"].append(index)
                    saved["z"].append(z.cpu())
                    saved["delta"].append(delta.cpu())
            rows.append(record)
            log.write(json.dumps(record) + "\n"); log.flush()
            if index % 25 == 0:
                print(f"search {index + 1}/{len(pool)}: {sum(r['solved'] for r in rows)} found, "
                      f"{time.time() - began:.0f} s", flush=True)
    adjuster.remove()
    torch.save({"row_index": saved["row_index"], "z": torch.stack(saved["z"]) if saved["z"] else None,
                "delta": torch.stack(saved["delta"]) if saved["delta"] else None, "layer": noise["layer"]},
               output / "targets.pt")
    summary = search_summary(rows, settings) | {"basis_orthonormal_error": orthonormal,
                                                 "seconds": time.time() - began, "smoke": smoke,
                                                 "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30
                                                 if torch.cuda.is_available() else None}
    json_write(output / "summary.json", summary)
    print(json.dumps(summary, indent=1), flush=True)
    return summary


def search_summary(rows, settings):
    found = [r for r in rows if r["solved"]]
    pushed = [r for r in found if r["scale"] is not None]
    reproduced = [r for r in pushed if r["reproduced"]]
    kept = [r for r in found if r["reproduced"] and r["answer_accepted"]]
    share = len(reproduced) / len(pushed) if pushed else 1.0
    ratios = [x for r in pushed for x in r["push_ratio"]]
    checks = {"basis_sha256": settings["noise"]["sha256"], "reproduced_share": share,
              "answers_rejected": sum(not r["answer_accepted"] for r in found),
              "valid": share >= settings["search"]["min_reproduced"] and bool(kept)}
    return {"maps": len(rows), "greedy_solved": sum(r["greedy_success"] for r in rows), "found": len(found),
            "found_by_scale": {str(s): sum(r["scale"] == s for r in found) for s in [None, *settings["search"]["scales"]]},
            "pushed": len(pushed), "reproduced": len(reproduced), "kept": len(kept),
            "kept_pushed": sum(r["scale"] is not None for r in kept),
            "tries_per_pushed_map": statistics.mean(r["tries"] for r in pushed) if pushed else None,
            "push_ratio_by_slot_median": [statistics.median(r["push_ratio"][s] for r in pushed)
                                          for s in range(len(pushed[0]["push_ratio"]))] if pushed else None,
            "push_ratio_median": statistics.median(ratios) if ratios else None,
            "checks": checks}


def kept_routes(search_run):
    """The found maps that reproduced, as expert_iteration.training_rows reads routes."""
    return [{"solved": True, "row_index": r["row_index"], "map_id": r["map_id"], "answer": r["answer"],
             "pushed": r["scale"] is not None}
            for r in jsonl(Path(search_run) / "search.jsonl")
            if r["solved"] and r["reproduced"] and r["answer_accepted"]]


def load_targets(search_run, pool, family):
    """{map_id: {"z", "delta", "directions", "energy"}}, keyed by map so that the latent loss finds
    its target from the row itself, not from its position in the training list."""
    saved = torch.load(Path(search_run) / "targets.pt", weights_only=False)
    ids = [pool[i]["map_id"] for i in saved["row_index"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Found maps share a map_id")
    return {map_id: {"z": saved["z"][n], "delta": saved["delta"][n], "row_index": i}
            for n, (map_id, i) in enumerate(zip(ids, saved["row_index"]))}


# D's latent loss ---------------------------------------------------------------------------------

class LatentTarget:
    """train_stage's auxiliary objective for arm D. On a row from a map in `targets` (amendment 2:
    the maps found by a push, z from `training_reference`): `huber` of P_s (h_s - a*_s) over the
    four latent slots, where h is the model's own output of decoder layer `layer` at the latent
    positions (out.hidden_states[layer]), a* = z + delta the state that produced the success, P_s
    the projection on slot s's basis directions (the push's subspace), and E_s the expected squared
    size of a scale-1 push in slot s. On released rows, and on rows of other maps, it is 0."""

    released_only = False
    head = None
    loss_key = "latent_loss_all"
    metric_keys = ("latent_loss_all", "latent_loss", "push_fraction")

    def __init__(self, data, pool_split, targets, family, layer, latent_ids, gradient_ratio):
        self.data, self.pool_split, self.targets, self.layer = data, pool_split, targets, layer
        # train_stage rescales this term's gradient to `gradient_ratio` times the route gradient's
        # norm every step (amendment 3).
        self.latent_ids, self.gradient_ratio = latent_ids, gradient_ratio
        self.directions = family["directions"].float()  # (K, d, k), orthonormal per slot
        self.energy = family["weights"].double().square().sum(-1).float()  # (K,)

    @property
    def settings(self):
        return {"layer": self.layer, "gradient_ratio": self.gradient_ratio, "targets": len(self.targets)}

    def prepare(self, tokenizer, rows):
        return {"rows_with_target": sum(self.target_of(split, row) is not None for split, row in self.data),
                "rows": len(self.data)}

    def target_of(self, split, row):
        return self.targets.get(row.get("map_id")) if split == self.pool_split else None

    def row_loss(self, hidden_states, split, row, batch):
        target = self.target_of(split, row)
        if target is None:
            return hidden_states[self.layer].new_zeros(()).float(), {"latent_loss": None, "push_fraction": None}
        positions = latent_queries(batch["input_ids"][:1], self.latent_ids)[0].nonzero().squeeze(-1)
        if len(positions) != len(self.latent_ids):
            raise ValueError(f"Expected {len(self.latent_ids)} latent positions, found {len(positions)}")
        h = hidden_states[self.layer][0, positions].float()
        device = h.device
        z, delta = target["z"].to(device), target["delta"].to(device)
        directions, energy = self.directions.to(device), self.energy.to(device)
        error = torch.einsum("sdk,sd->sk", directions, h - (z + delta))
        loss = huber(error, energy)
        size = delta.square().sum(-1)
        fraction = None
        if bool((size > 0).all()):
            fraction = float((((h.detach() - z) * delta).sum(-1) / size).mean())
        return loss, {"latent_loss": loss.item(), "push_fraction": fraction}

    def loss(self, hidden_states, index, batch):
        split, row = self.data[index]
        return self.row_loss(hidden_states, split, row, batch)


def huber(error, energy):
    """D's training loss: per slot, e = ||error_s|| / sqrt(E_s) (in scale-1 push units), then
    e^2 / 2 below 1 and e - 1/2 above, averaged over slots. Small errors (numerical mismatch) get
    small gradients; large ones (drift) a gradient of fixed size (2026-11-19 amendment 2)."""
    e = (error.square().sum(-1) / energy + 1e-12).sqrt()
    return torch.where(e < 1, 0.5 * e.square(), e - 0.5).mean()


@torch.no_grad()
def training_reference(model, processor, data_root, pool_split, pool, targets, routes, layer, config):
    """D's targets (amendment 2): pushed maps only, with z recomputed by the training forward pass
    (training encoding with the found answer, training mode, bf16 autocast) on LIVR before any
    update, so that the starting error is the push itself rather than inference/training
    numerical mismatch. Returns the targets and the mismatch to the inference z, in push units."""
    training_mode(model, config)
    answers = {r["map_id"]: r["answer"] for r in routes}
    kept = {}
    mismatch = []
    for map_id, target in targets.items():
        if not bool((target["delta"].square().sum(-1) > 0).all()):
            continue
        row = pool[target["row_index"]] | {"text_output": answers[map_id]}
        batch = to_device(encode(processor, data_root, pool_split, row), model.device)
        with amp(model.device):
            out = model(**batch, use_cache=False, output_hidden_states=True)
        positions = latent_queries(batch["input_ids"][:1], model.config.livr_latent_ids)[0].nonzero().squeeze(-1)
        z = out.hidden_states[layer][0, positions].float().cpu()
        del out
        mismatch.append(float(((z - target["z"]).norm(dim=-1) / target["delta"].norm(dim=-1)).mean()))
        kept[map_id] = target | {"z": z}
    return kept, {"maps": len(kept), "mean_mismatch_over_push_norm": statistics.mean(mismatch)
                         if mismatch else None}


def distance(error, energy):
    """Mean over slots of ||error_s|| / sqrt(E_s); `error` (K, k) in the basis coordinates."""
    return (error.square().sum(-1) / energy + 1e-12).sqrt().mean()


def training_mode(model, config):
    model.config.livr_stage = 2
    model.train()
    model.visual.eval()
    model.config.use_cache = False
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


@torch.no_grad()
def target_fit(model, processor, data_root, pool_split, pool, targets, family, layer):
    """How close the model's own latent states are to the targets, from one prefill per found map:
    the relative error of `LatentTarget` and the fraction of the push reproduced, separately for
    pushed maps and for maps LIVR solved unpushed (whose target is LIVR's own state)."""
    model.eval()
    model.config.livr_stage = 2
    adjuster = LatentAdjuster(model, layer)
    directions = family["directions"].float().to(model.device)
    energy = family["weights"].double().square().sum(-1).float().to(model.device)
    out = {"pushed": {"loss": [], "push_fraction": [], "drift": []}, "unpushed": {"loss": [], "drift": []}}
    for map_id, target in targets.items():
        row = pool[target["row_index"]]
        h = capture_state(model, adjuster, prompt_of(processor, data_root, pool_split, row, model.device))
        z, delta = target["z"].to(h.device), target["delta"].to(h.device)
        error = torch.einsum("sdk,sd->sk", directions, h - (z + delta))
        size = delta.square().sum(-1)
        kind = "pushed" if bool((size > 0).all()) else "unpushed"
        out[kind]["loss"].append(float(distance(error, energy)))
        out[kind]["drift"].append(float(((h - z).norm(dim=-1) / z.norm(dim=-1)).mean()))
        if kind == "pushed":
            out[kind]["push_fraction"].append(float((((h - z) * delta).sum(-1) / size).mean()))
    adjuster.remove()
    return {kind: {"maps": len(values["loss"])} | {k: statistics.mean(v) if v else None for k, v in values.items()}
            for kind, values in out.items()}


# Training and evaluation -------------------------------------------------------------------------

def rows_digest(data):
    text = json.dumps([[split, row.get("map_id"), row["text_output"]] for split, row in data])
    return hashlib.sha256(text.encode()).hexdigest()


def populations(data_root, settings):
    """(population, split, row_index, row): round 1's 240 held-out detour maps, the 400 held-out
    7 x 7 and 8 x 8 maps, and the first `seen_maps` pool maps."""
    pool = settings["pool_split"]
    return ([("heldout", settings["heldout_split"], i, r) for i, r in enumerate(records(data_root, settings["heldout_split"]))]
            + [("test_detour", "test", i, r) for i, r in enumerate(records(data_root, "test")) if is_detour(r["map_desc"])]
            + [("heldout78", settings["transfer_split"], i, r)
               for i, r in enumerate(records(data_root, settings["transfer_split"]))]
            + [("seen", pool, i, r) for i, r in enumerate(records(data_root, pool)[:settings["seen_maps"]])])


@torch.no_grad()
def greedy_eval(model, processor, evaluator, data_root, items, log, max_new_tokens):
    """detour_scaling.greedy_eval, also recording the decoded answer for the CPU rescore."""
    model.eval()
    model.config.use_cache = True
    results = []
    for count, (population, split, index, row) in enumerate(items):
        prompt = prompt_of(processor, data_root, split, row, model.device)
        with amp(model.device):
            generated = model.generate(**prompt, **GREEDY, max_new_tokens=max_new_tokens, use_cache=True)
        text = processor.tokenizer.decode(generated[0], skip_special_tokens=False)
        result = evaluator.score(row, text)
        record = {"population": population, "split": split, "row_index": index, "grid_size": len(row["map_desc"]),
                  "route": "".join(ch for ch in result["extracted_answer"].upper() if ch in "UDRL"),
                  "success": bool(result["success"]), "text": text}
        results.append(record)
        log.write(json.dumps(record) + "\n"); log.flush()
        if count % 100 == 0:
            print(f"greedy {count + 1}/{len(items)}", flush=True)
    return results


def run_train(model, processor, evaluator, config, data_root, output, settings, arm, seed, search_run, basis_path,
              run_manifest, offload=False, smoke=False, accept_search=False):
    """`accept_search`: train on a search that failed only its replay bar (the 2026-11-19 amendment
    in the specification); recorded in the manifest and summary, and `compare` reads it."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    search = json.loads((Path(search_run) / "summary.json").read_text())
    if search.get("smoke") and not smoke:
        raise ValueError(f"The search {search_run} is a smoke search")
    if not search["checks"]["valid"] and not (accept_search and search["kept"] > 0):
        raise ValueError(f"The search {search_run} failed its validity checks")
    noise = settings["noise"]
    family = load_noise({"noise": noise}, basis_path)
    pool_split = settings["pool_split"]
    pool = records(data_root, pool_split)
    routes = kept_routes(search_run)
    data, distinct = training_rows(data_root, settings, routes, "search")
    digest = rows_digest(data)
    targets = load_targets(search_run, pool, family)
    if smoke:  # two optimizer steps: released rows and found rows, both kinds of found map
        found_rows = [d for d in data if d[0] == pool_split]
        data = data[:8] + found_rows[:8]
    config = config | {"seed": seed, "epochs_per_stage": settings["epochs"], "learning_rate": settings["learning_rate"],
                       "warmup_ratio": settings["warmup_ratio"]}
    checks = check_sets(evaluator, data_root, settings)
    manifest = run_manifest | {"arm": arm, "seed": seed, "search_run": str(search_run), "basis": str(basis_path),
                               "rows_sha256": digest, "rows": len(data), "distinct": distinct,
                               "offload_optimizer": offload, "smoke": smoke, "settings": settings, "checks": checks,
                               "search_accepted_below_replay_bar": not search["checks"]["valid"]}
    json_write(output / "manifest.json", manifest)
    seed_all(seed)
    before = target_fit(model, processor, data_root, pool_split, pool, targets, family, noise["layer"])
    print(json.dumps({"target_fit_before": before}), flush=True)
    objective, latent = None, None
    if arm == "D":
        reference, mismatch = training_reference(model, processor, data_root, pool_split, pool, targets, routes,
                                                 noise["layer"], config)
        latent = {"gradient_ratio": settings["latent"]["ratio"], "training_reference": mismatch}
        print(json.dumps({"latent": latent}), flush=True)
        objective = LatentTarget(data, pool_split, reference, family, noise["layer"], model.config.livr_latent_ids,
                                 latent["gradient_ratio"])
    started = time.time()
    train_stage(model, processor, data_root, output / "stage2", config, 2, final_weights_only=True, data=data,
                grounding=objective, save_final=not smoke, offload_optimizer=offload)
    steps = jsonl(output / "stage2/steps.jsonl")
    training = {"seconds": time.time() - started, "optimizer_steps": len(steps),
                "seconds_per_step": statistics.mean(s["seconds"] for s in steps),
                "mean_loss": statistics.mean(s["loss"] for s in steps),
                "last_quarter_loss": statistics.mean(s["loss"] for s in steps[-max(1, len(steps) // 4):]),
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    if arm == "D":
        losses = [s["latent_loss"] for s in steps if s.get("latent_loss") is not None]
        training["latent_loss_first_last_quarter"] = [statistics.mean(losses[:max(1, len(losses) // 4)]),
                                                      statistics.mean(losses[-max(1, len(losses) // 4):])] if losses else None
    after = target_fit(model, processor, data_root, pool_split, pool, targets, family, noise["layer"])
    seed_all(seed)
    items = populations(data_root, settings)
    if smoke:
        items = [i for p in ("heldout", "test_detour", "heldout78", "seen") for i in [x for x in items if x[0] == p][:2]]
    started = time.time()
    with (output / "greedy.jsonl").open("w", encoding="utf-8") as log:
        greedy = greedy_eval(model, processor, evaluator, data_root, items, log, settings["max_new_tokens"])
    greedy_seconds = time.time() - started
    if not smoke:
        run_benchmark(model, processor, config, data_root, output / "evaluation",
                      run_manifest | {"checkpoint": f"in-memory {output / 'stage2/final'}", "arm": arm, "seed": seed})
    summary = {"arm": arm, "seed": seed, "rows_sha256": digest, "rows": len(data),
               "search_accepted_below_replay_bar": not search["checks"]["valid"],
               "expected_steps": -(-len(data) // config["gradient_accumulation"]) * settings["epochs"],
               "checks": checks, "latent": latent,
               "target_fit": {"before": before, "after": after}, "training": training,
               "greedy": {p: [sum(r["success"] for r in greedy if r["population"] == p),
                              sum(r["population"] == p for r in greedy)] for p in (*PRIMARY, *TRANSFER, "seen")},
               "greedy_seconds": greedy_seconds, "seconds": time.time() - began, "smoke": smoke,
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    json_write(output / "summary.json", summary)
    print(json.dumps(summary, indent=1), flush=True)
    return summary


# Readings (CPU) ----------------------------------------------------------------------------------

def greedy_of(run):
    return {(r["population"], r["row_index"]): bool(r["success"]) for r in jsonl(Path(run) / "greedy.jsonl")}


def rescore(evaluator, data_root, run):
    """Every greedy and benchmark answer against the reference evaluator, on the CPU."""
    cache, checked, disagree = {}, 0, 0
    for r in jsonl(Path(run) / "greedy.jsonl"):
        if r["split"] not in cache:
            cache[r["split"]] = records(data_root, r["split"])
        checked += 1
        disagree += bool(evaluator.score(cache[r["split"]][r["row_index"]], r["text"])["success"]) != r["success"]
    test = records(data_root, "test")
    for r in jsonl(Path(run) / "evaluation/responses.jsonl"):
        checked += 1
        disagree += bool(evaluator.score(test[r["row_index"]], r["raw_response"])["success"]) != bool(r["success"])
    return {"answers": checked, "disagreements": disagree, "valid": checked > 0 and disagree == 0}


def verdict(test, valid, helps, hurts, neither):
    if not valid:
        return "withheld: the measurement failed a validity check"
    each = [a - b for a, b in zip(test["first"], test["second"])]
    if test["p"] < ALPHA and test["maps_first_ahead"] > test["maps_second_ahead"] and all(e > 0 for e in each):
        return helps
    if test["p"] < ALPHA and test["maps_first_ahead"] < test["maps_second_ahead"] and all(e < 0 for e in each):
        return hurts
    return neither


def latent_record(summary):
    """D's latent settings from a run summary: `latent`, or, in the runs of 2026-11-19 (jobs 7170868),
    the `calibration` record whose `ratio` is the per-step gradient ratio."""
    if summary.get("latent"):
        return summary["latent"]
    return {"gradient_ratio": summary["calibration"]["ratio"],
            "training_reference": summary["calibration"]["training_reference"]}


def compare(runs, search_run, livr_greedy, livr_benchmark, data_root, evaluator, settings):
    """`runs`: {"S42", "S43", "D42", "D43"} -> run directory. `livr_greedy`: LIVR's greedy answers on
    the same populations (the fair pilot's greedy-livr.jsonl); `livr_benchmark`: LIVR's benchmark
    responses.jsonl."""
    seeds = settings["seeds"]
    names = [f"{arm}{seed}" for arm in ARMS for seed in seeds]
    summaries = {n: json.loads((Path(runs[n]) / "summary.json").read_text()) for n in names}
    greedy = {n: greedy_of(runs[n]) for n in names}
    livr = {(r["population"], r["row_index"]): bool(r["success"]) for r in jsonl(livr_greedy)}
    keys = sorted(k for k in greedy[names[0]] if k[0] != "seen")
    if any(sorted(k for k in g if k[0] != "seen") != keys for g in greedy.values()):
        raise ValueError("The runs evaluated different maps")
    primary_keys = [k for k in keys if k[0] in PRIMARY]
    transfer_keys = [k for k in keys if k[0] in TRANSFER]
    search = json.loads((Path(search_run) / "summary.json").read_text())
    rescored = {n: rescore(evaluator, data_root, runs[n]) for n in names}
    validity = {
        "search": search["checks"],
        "rows_match_per_seed": all(summaries[f"S{s}"]["rows_sha256"] == summaries[f"D{s}"]["rows_sha256"] for s in seeds),
        "steps": {n: s["training"]["optimizer_steps"] for n, s in summaries.items()},
        "rescore": rescored,
        "latent": {f"D{s}": latent_record(summaries[f"D{s}"]) for s in seeds},
        "set_checks": all(s["checks"]["valid"] for s in summaries.values())}
    others = (validity["rows_match_per_seed"] and validity["set_checks"]
             and all(s["training"]["optimizer_steps"] == s["expected_steps"] for s in summaries.values())
             and all(r["valid"] for r in rescored.values())
             and all(r["gradient_ratio"] > 0 and r["training_reference"]["maps"] > 0
                     for r in validity["latent"].values())
             and not any(s.get("smoke") for s in summaries.values()))
    valid = search["checks"]["valid"] and others
    # The 2026-11-19 amendment (made after the search, before training): accept a search whose only
    # failure is the replay bar. Reported beside the pre-fixed reading, never in its place.
    amended = others and search["kept"] > 0
    validity |= {"valid": valid, "valid_if_replay_amendment_accepted": amended}
    d, s_ = [greedy[f"D{s}"] for s in seeds], [greedy[f"S{s}"] for s in seeds]
    primary = sign_test(d, s_, primary_keys)
    transfer = sign_test(d, s_, transfer_keys)
    livr_bench = {r["row_index"]: bool(r["success"]) for r in jsonl(livr_benchmark)}
    test = records(data_root, "test")
    detour_rows = {i for i, r in enumerate(test) if is_detour(r["map_desc"])}
    rows = sorted(livr_bench)
    bench = {n: {r["row_index"]: bool(r["success"]) for r in jsonl(Path(runs[n]) / "evaluation/responses.jsonl")}
             for n in names}
    benchmark = {n: {"all": paired(bench[n], livr_bench, rows),
                     "detour": paired(bench[n], livr_bench, [r for r in rows if r in detour_rows]),
                     "no_detour": paired(bench[n], livr_bench, [r for r in rows if r not in detour_rows])}
                 for n in names}
    bench_keys = [("bench", r) for r in rows]
    bench_d = [{("bench", r): bench[f"D{s}"][r] for r in rows} for s in seeds]
    bench_s = [{("bench", r): bench[f"S{s}"][r] for r in rows} for s in seeds]
    readings = ("the latent target helps", "the latent target hurts", "no detectable difference")
    return {
        "validity": validity,
        "primary": primary | {"reading": verdict(primary, valid, *readings),
                              "reading_if_replay_amendment_accepted": verdict(primary, amended, *readings)},
        "secondary": {
            "transfer": transfer | {"reading": verdict(transfer, valid, *readings),
                                    "reading_if_replay_amendment_accepted": verdict(transfer, amended, *readings)},
            "benchmark_vs_livr": benchmark,
            "benchmark_d_vs_s": sign_test(bench_d, bench_s, bench_keys),
            "vs_livr_on_240": {n: paired(greedy[n], livr, primary_keys) for n in names}},
        "diagnostics": {
            "target_fit": {n: summaries[n]["target_fit"] for n in names},
            "search": {k: v for k, v in search.items() if k != "checks"},
            "training": {n: summaries[n]["training"] for n in names},
            "greedy": {n: summaries[n]["greedy"] for n in names}},
        "seconds_in_jobs": {"search": search["seconds"], **{n: s["seconds"] for n, s in summaries.items()}}}
