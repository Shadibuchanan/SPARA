"""B′ and C′ under fair conditions (docs/STRUCTURED_EXPLORATION_SPEC.md, the amendment of
2026-11-17): text-only GRPO (B′) and GRPO with a learned structured latent adjustment (C′),
from the LIVR rerun, on unseen detour maps, with the model's full sampling distribution.

- `run_pilot`: one job on LIVR before the runs. The pool's set checks; the basis of C′'s
  noise on the pool maps; the sampler check; the share of groups with mixed rewards under
  B′'s settings, which sets the maps per step; C′'s scale by C's rule; L's greedy answers on
  every evaluation population. The rules are fixed in the specification.
- `train_arm` and `evaluate_arm`: one arm and seed. C′ first checks that its fresh heads
  leave LIVR's routes unchanged.
- `compare` (CPU): the readings fixed in the specification.
"""
import contextlib
import json
import math
import time
from pathlib import Path

import torch

from .benchmark import run_benchmark
from .data import layout_key, records
from .detour_scaling import greedy_eval, jsonl, paired
from .explore_probe import GREEDY, latent_states, noise_probe, principal_basis, structured_offsets, summarise_noise
from .fresh_maps import is_detour
from .latent_battery import mcnemar
from .latent_rl import LatentAdjuster, add_latent_heads, sigma_bounds
from .runtime import json_write, seed_all
from .serialization import encode
from .text_rl import repeat_prompt, rollout, stream_batch, train_text_rl
from .training import amp, to_device

ALPHA = 0.05
ARMS = ("b", "c")
PRIMARY = ("heldout", "test_detour")  # round 1's 240 held-out maps


# Populations ---------------------------------------------------------------------------------

def pilot_indices(data_root, settings):
    """The first `pilot_maps` maps of the training stream of seed `pilot_seed`: the maps the
    pilot measures, and the policy-moved check's maps."""
    rows = records(data_root, settings["pool_split"])
    return stream_batch(len(rows), settings["pilot_maps"], settings["pilot_seed"], 0)[0]


def populations(data_root, settings):
    """(population, split, row_index, row) for every greedy evaluation: round 1's 240 held-out
    maps, round 2's 400, and the pilot's pool maps."""
    round1, round2, pool = settings["round1_heldout_split"], settings["heldout_split"], settings["pool_split"]
    pool_rows = records(data_root, pool)
    return ([("heldout", round1, i, r) for i, r in enumerate(records(data_root, round1))]
            + [("test_detour", "test", i, r) for i, r in enumerate(records(data_root, "test")) if is_detour(r["map_desc"])]
            + [("heldout78", round2, i, r) for i, r in enumerate(records(data_root, round2))]
            + [("moved", pool, i, pool_rows[i]) for i in pilot_indices(data_root, settings)])


def check_sets(evaluator, data_root, settings):
    """Validity: the pool's labels pass the reference evaluator, and it shares no layout with
    the released splits or either held-out set."""
    pool = records(data_root, settings["pool_split"])
    mine = {layout_key(r) for r in pool}
    shared = {split: len(mine & {layout_key(r) for r in records(data_root, split)})
              for split in ("train", "test", settings["round1_heldout_split"], settings["heldout_split"])}
    accepted = sum(bool(evaluator.score(r, r["text_output"])["success"]) for r in pool)
    return {"maps": len(pool), "detour_maps": sum(is_detour(r["map_desc"]) for r in pool),
            "labels_accepted": [accepted, len(pool)], "shared_layouts": shared,
            "valid": accepted == len(pool) and not any(shared.values())}


# Pilot ---------------------------------------------------------------------------------------

@torch.no_grad()
def sampler_check(model, prompt, generation, rollouts):
    """The largest total-variation distance, over sampled positions, between the distribution
    `generate` samples from (its processed scores) and the softmax the loss differentiates
    (the raw logits at the rollout temperature; text_rl.token_logprobs). Tokens the generation
    settings themselves suppress are removed from both."""
    model.eval()
    with amp(model.device):
        out = model.generate(**repeat_prompt(prompt, rollouts), **generation, use_cache=True,
                             output_scores=True, output_logits=True, return_dict_in_generate=True)
    suppressed = list(generation.get("suppress_tokens") or [])
    worst = 0.0
    for scores, logits in zip(out.scores, out.logits):
        raw = logits.float() / generation.get("temperature", 1.0)
        if suppressed:
            raw[:, suppressed] = float("-inf")
        distance = 0.5 * (scores.float().softmax(-1) - raw.softmax(-1)).abs().sum(-1)
        worst = max(worst, float(distance.max()))
    return worst


@torch.no_grad()
def signal(model, processor, evaluator, data_root, split, rows, indices, rl, eos_ids, log):
    """B′'s rollouts on the pilot maps: success of each, and whether it reached the token cap."""
    results = []
    cap = rl["generation"]["max_new_tokens"]
    for count, index in enumerate(indices):
        row = rows[index]
        prompt = to_device(encode(processor, data_root, split, row, training=False), model.device)
        completions, _ = rollout(model, prompt, rl, eos_ids)
        texts = [processor.tokenizer.decode(torch.cat([prompt["input_ids"][0], c]), skip_special_tokens=False)
                 for c in completions]
        record = {"row_index": index, "grid_size": len(row["map_desc"]),
                  "successes": [bool(evaluator.score(row, text)["success"]) for text in texts],
                  "truncated": [len(c) >= cap and int(c[-1]) not in eos_ids for c in completions]}
        results.append(record)
        log.write(json.dumps(record) + "\n"); log.flush()
        if count % 25 == 0:
            print(f"signal {count + 1}/{len(indices)}", flush=True)
    return results


def prompts_per_step(mixed_share, settings):
    """The smallest multiple of 8 with P x s at least `min_informative_groups`, at most
    `max_prompts_per_step`; None (stop) below `min_mixed_share`."""
    if mixed_share < settings["min_mixed_share"]:
        return None
    return min(8 * math.ceil(settings["min_informative_groups"] / mixed_share / 8), settings["max_prompts_per_step"])


def pilot_rules(sets, sampler, signal_rows, noise, settings):
    rollouts = [s for r in signal_rows for s in r["successes"]]
    mixed = sum(0 < sum(r["successes"]) < len(r["successes"]) for r in signal_rows) / len(signal_rows)
    truncated = sum(t for r in signal_rows for t in r["truncated"]) / len(rollouts)
    scale = noise["families"]["structured"]["chosen"]
    stop = []
    if not sets["valid"] or sampler > settings["sampler_tolerance"]:
        stop.append("a validity check failed")
    size = prompts_per_step(mixed, settings)
    if size is None:
        stop.append(f"mixed share {mixed:.3f} below {settings['min_mixed_share']}: even full-support sampling "
                    "rarely finds a success on these maps; the team decides between G = 16 and no-detour maps")
    if truncated > settings["max_truncated_share"]:
        stop.append(f"{truncated:.3f} of answers reached the token cap; raise it")
    if scale is None:
        stop.append("no scale met C's rule; the team decides")
    return {"mixed_share": mixed, "all_fail_share": sum(not any(r["successes"]) for r in signal_rows) / len(signal_rows),
            "all_success_share": sum(all(r["successes"]) for r in signal_rows) / len(signal_rows),
            "rollout_success": sum(rollouts) / len(rollouts), "truncated_share": truncated,
            "prompts_per_step": size, "scale": scale, "stop": stop}


def run_pilot(model, processor, evaluator, data_root, output, settings, rl, run_manifest):
    """On LIVR: set checks, the basis, the sampler check, the signal, C′'s scale, L's greedy."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    json_write(output / "manifest.json", run_manifest | {"settings": settings, "rl": rl})
    model.eval()
    model.config.livr_stage = 2
    sets = check_sets(evaluator, data_root, settings)
    split = settings["pool_split"]
    pool = records(data_root, split)
    adjuster = LatentAdjuster(model, settings["layer"])
    states = latent_states(model, processor, data_root, split, pool, adjuster)
    basis = principal_basis(states, [len(r["map_desc"]) for r in pool], settings["basis_k"])
    torch.save(basis, output / "basis.pt")
    print(json.dumps({k: basis[k] for k in ("captured", "k", "maps")}), flush=True)

    indices = pilot_indices(data_root, settings)
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    seed_all(settings["pilot_seed"])
    prompt = to_device(encode(processor, data_root, split, pool[indices[0]], training=False), model.device)
    sampler = sampler_check(model, prompt, rl["generation"], settings["sampler_rollouts"])
    with (output / "signal.jsonl").open("w", encoding="utf-8") as log:
        signal_rows = signal(model, processor, evaluator, data_root, split, pool, indices, rl, eos, log)

    generator = torch.Generator(device=model.device).manual_seed(settings["pilot_seed"])
    families = {"structured": (list(settings["scales"]),
                               lambda s, n: structured_offsets(basis, s, n, generator, model.device))}
    with (output / "noise.jsonl").open("w", encoding="utf-8") as log:
        noise_rows = noise_probe(model, processor, data_root, split, [(i, pool[i]) for i in indices], families,
                                 settings["draws"], evaluator, adjuster, generator, log, settings["max_new_tokens"])
    adjuster.remove()
    noise = summarise_noise(noise_rows)
    seed_all(settings["pilot_seed"])
    with (output / "greedy-livr.jsonl").open("w", encoding="utf-8") as log:
        greedy = greedy_eval(model, processor, evaluator, data_root, populations(data_root, settings), log,
                             settings["max_new_tokens"])
    rules = pilot_rules(sets, sampler, signal_rows, noise, settings)
    summary = {"checks": {"sets": sets, "sampler_max_tv": sampler, "sampler_tolerance": settings["sampler_tolerance"],
                          "valid": sets["valid"] and sampler <= settings["sampler_tolerance"]},
               "basis": {k: basis[k] for k in ("captured", "k", "maps")}, "pilot_rows": indices,
               "noise": noise, "rules": rules, "proceed": not rules["stop"],
               "livr_greedy": counts(greedy), "seconds": time.time() - began,
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    json_write(output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("checks", "basis", "rules", "proceed")}, indent=1), flush=True)
    return summary


def counts(greedy):
    return {p: [sum(r["success"] for r in greedy if r["population"] == p), sum(r["population"] == p for r in greedy)]
            for p in dict.fromkeys(r["population"] for r in greedy)}


# Arms ------------------------------------------------------------------------------------------

def recipe(rl, arm, seed, pilot, settings):
    """The arm's RL settings with the pilot's maps per step (and, for C′, its scale)."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if rl.get("train_split") != settings["pool_split"]:
        raise ValueError(f"The RL recipe trains on {rl.get('train_split')}, but the pilot measured {settings['pool_split']}")
    if bool(rl.get("latent")) != (arm == "c"):
        raise ValueError("B′ uses configs/text_rl_fair.json and C′ configs/latent_rl_fair.json")
    if not pilot["proceed"] or not pilot["checks"]["valid"]:
        raise ValueError("The pilot stopped or failed a validity check; no run is read without it")
    rl = json.loads(json.dumps(rl)) | {"prompts_per_step": pilot["rules"]["prompts_per_step"], "seed": seed}
    if arm == "c":
        scale = pilot["rules"]["scale"]
        rl["latent"] |= {"sigma_init": scale, "log_sigma_bounds": sigma_bounds(scale)}
    return rl


@torch.no_grad()
def identity_check(model, processor, evaluator, data_root, items, adjuster, max_new_tokens):
    """C′'s fresh heads have mu = 0: the mean adjustment must leave LIVR's greedy routes as they are."""
    plain = greedy_eval(model, processor, evaluator, data_root, items, NullLog(), max_new_tokens)
    with adjuster.mean():
        mean = greedy_eval(model, processor, evaluator, data_root, items, NullLog(), max_new_tokens)
    same = sum(a["route"] == b["route"] for a, b in zip(plain, mean))
    return {"maps": len(items), "same_routes": same, "valid": same == len(items)}


class NullLog:
    def write(self, text): pass
    def flush(self): pass


def train_arm(model, reference, processor, evaluator, config, data_root, output, settings, rl, arm, seed,
              pilot_run, source, run_manifest):
    """Train one arm and seed from LIVR; returns the latent adjuster for C′ (None for B′)."""
    pilot_run, output = Path(pilot_run), Path(output)
    pilot = json.loads((pilot_run / "summary.json").read_text())
    rl = recipe(rl, arm, seed, pilot, settings)
    output.mkdir(parents=True, exist_ok=False)
    manifest = run_manifest | {"arm": arm, "seed": seed, "rl": rl, "settings": settings, "pilot_run": str(pilot_run)}
    adjusters, identity = None, None
    if arm == "c":
        basis = torch.load(pilot_run / "basis.pt")
        add_latent_heads(model, rl["latent"], basis)
        adjusters = (LatentAdjuster(model, rl["latent"]["layer"]), LatentAdjuster(reference, rl["latent"]["layer"]))
        pool = records(data_root, settings["pool_split"])
        items = [("identity", settings["pool_split"], i, pool[i]) for i in pilot["pilot_rows"][:settings["identity_maps"]]]
        identity = identity_check(model, processor, evaluator, data_root, items, adjusters[0], settings["max_new_tokens"])
        if not identity["valid"]:
            json_write(output / "arm-summary.json", {"arm": arm, "seed": seed, "identity": identity})
            raise ValueError("C′'s fresh heads changed LIVR's routes; see arm-summary.json")
    json_write(output / "manifest.json", manifest | {"identity": identity})
    seed_all(seed)
    started = time.time()
    train_text_rl(model, reference, processor, data_root, output / "rl", config, rl, source,
                  scorer=lambda row, text: float(bool(evaluator.score(row, text)["success"])),
                  final_weights_only=True, adjusters=adjusters)
    json_write(output / "arm-summary.json", {"arm": arm, "seed": seed, "rl": rl, "identity": identity,
                                             "training_seconds": time.time() - started})
    if adjusters:
        adjusters[1].remove()
    return adjusters[0] if adjusters else None


def evaluate_arm(model, processor, evaluator, config, data_root, output, settings, arm, adjuster, run_manifest):
    """Greedy answers on every population, in each latent mode for C′ (sampled only on the
    primary maps), and the reference benchmark (C′ with the mean adjustment)."""
    output = Path(output)
    model.eval()
    model.config.livr_stage = 2
    items = populations(data_root, settings)
    modes = ({"plain": (contextlib.nullcontext, items)} if arm == "b" else
             {"mean": (adjuster.mean, items), "zero": (contextlib.nullcontext, items),
              "sample": (adjuster.sample, [i for i in items if i[0] in PRIMARY])})
    results = {}
    for mode, (context, chosen) in modes.items():
        seed_all(config["seed"])
        with context(), (output / f"greedy-{mode}.jsonl").open("w", encoding="utf-8") as log:
            results[mode] = counts(greedy_eval(model, processor, evaluator, data_root, chosen, log,
                                               settings["max_new_tokens"]))
    benchmark_mode = "plain" if arm == "b" else "mean"
    with modes[benchmark_mode][0]():
        run_benchmark(model, processor, config, data_root, output / f"benchmark-{benchmark_mode}",
                      run_manifest | {"arm": arm, "latent_mode": benchmark_mode})
    summary = json.loads((output / "arm-summary.json").read_text()) | {"greedy": results}
    json_write(output / "arm-summary.json", summary)
    return summary


# Readings (CPU) ------------------------------------------------------------------------------

def greedy_of(run, mode):
    return {(r["population"], r["row_index"]): bool(r["success"]) for r in jsonl(Path(run) / f"greedy-{mode}.jsonl")}


def sign_test(first, second, keys):
    """Per map, the first arm's successes summed over seeds minus the second's; exact
    two-sided sign test on the maps where they differ."""
    difference = {k: sum(f[k] for f in first) - sum(s[k] for s in second) for k in keys}
    better, worse = sum(d > 0 for d in difference.values()), sum(d < 0 for d in difference.values())
    return {"maps": len(keys), "first": [sum(f[k] for k in keys) for f in first],
            "second": [sum(s[k] for k in keys) for s in second], "maps_first_ahead": better,
            "maps_second_ahead": worse, "p": mcnemar(better, worse)}


def verdict(test, valid, moved):
    if not valid:
        return "withheld: the measurement failed a validity check"
    if not moved:
        return "RL did not move the policy at this budget; no reading on latent exploration"
    each = [a - b for a, b in zip(test["first"], test["second"])]
    if test["p"] < ALPHA and test["maps_first_ahead"] > test["maps_second_ahead"] and all(e >= 0 for e in each):
        return "C′ better: latent exploration adds value beyond text exploration on unseen detour maps"
    if test["p"] < ALPHA and test["maps_first_ahead"] < test["maps_second_ahead"] and all(e <= 0 for e in each):
        return "C′ worse: latent exploration hurt in this recipe"
    return "no detectable difference: no gain of about 7 points or more from latent exploration in this recipe"


def rescore(evaluator, data_root, split, run):
    """Rescore every rollout's completion with the reference evaluator; count disagreements."""
    rows = records(data_root, split)
    checked = disagree = 0
    for record in jsonl(Path(run) / "rl/rollouts.jsonl"):
        for completion, reward in zip(record["completions"], record["rewards"]):
            checked += 1
            disagree += float(bool(evaluator.score(rows[record["row"]], completion)["success"])) != reward
    return {"rollouts": checked, "disagreements": disagree, "valid": checked > 0 and disagree == 0}


def training_signal(run):
    steps = jsonl(Path(run) / "rl/steps.jsonl")
    kls = [s["kl"] for s in steps if s["kl"] is not None]
    out = {"steps": len(steps), "updated_steps": sum(s["updated"] for s in steps),
           "mixed_share": sum(s["groups"] - s["zero_variance_groups"] for s in steps) / sum(s["groups"] for s in steps),
           "verifier_calls": steps[-1]["verifier_calls"], "mean_kl": sum(kls) / len(kls) if kls else None}
    latent = [s for s in steps if "sigma_mean" in s]
    if latent:
        out |= {"sigma_first_last": [latent[0]["sigma_mean"], latent[-1]["sigma_mean"]],
                "mean_abs_mu_first_last": [latent[0]["mean_abs_mu"], latent[-1]["mean_abs_mu"]]}
    return out


def rescues_and_breaks(b_run, c_run):
    """Paired by step and map within a seed: groups mixed for C′ only, by what B′'s group did."""
    groups = lambda run: {(r["step"], r["row"]): r["rewards"] for r in jsonl(Path(run) / "rl/rollouts.jsonl")}
    b, c = groups(b_run), groups(c_run)
    keys = sorted(b.keys() & c.keys())  # all of them when the pairing check passes
    mixed = lambda rewards: 0 < sum(rewards) < len(rewards)
    return {"paired_groups": len(keys),
            "c_only_mixed_where_b_all_failed": sum(mixed(c[k]) and not any(b[k]) for k in keys),
            "c_only_mixed_where_b_all_succeeded": sum(mixed(c[k]) and all(b[k]) for k in keys),
            "b_only_mixed": sum(mixed(b[k]) and not mixed(c[k]) for k in keys),
            "both_mixed": sum(mixed(b[k]) and mixed(c[k]) for k in keys)}


def compare(runs, pilot_run, settings, livr_outcomes, data_root, evaluator=None, e_round1=None, e_round2=None):
    """`runs`: {"b": {seed: run}, "c": {seed: run}}. `livr_outcomes`: the LIVR rerun's
    benchmark outcomes. `evaluator` rescores the rollouts (validity); `e_round1`, `e_round2`:
    E's greedy.jsonl files on round 1's and round 2's held-out maps, for context."""
    pilot = json.loads((Path(pilot_run) / "summary.json").read_text())
    livr = {(r["population"], r["row_index"]): bool(r["success"]) for r in jsonl(Path(pilot_run) / "greedy-livr.jsonl")}
    seeds = sorted(runs["b"])
    if sorted(runs["c"]) != seeds:
        raise ValueError("Each seed needs both arms")
    arms = {name: [Path(runs[name][s]) for s in seeds] for name in ARMS}
    summaries = {name: [json.loads((run / "arm-summary.json").read_text()) for run in arms[name]] for name in ARMS}
    progress = {name: [json.loads((run / "rl/summary.json").read_text()) for run in arms[name]] for name in ARMS}
    completed = all(p["optimizer_steps"] == s["rl"]["steps"] for name in ARMS
                    for p, s in zip(progress[name], summaries[name]))
    pairing = {seed: [r["row"] for r in jsonl(b / "rl/rollouts.jsonl")] == [r["row"] for r in jsonl(c / "rl/rollouts.jsonl")]
               for seed, b, c in zip(seeds, arms["b"], arms["c"])}
    identity = all(s["identity"]["valid"] for s in summaries["c"])
    rescored = ({f"{name}{seed}": rescore(evaluator, data_root, settings["pool_split"], run)
                 for name in ARMS for seed, run in zip(seeds, arms[name])} if evaluator else None)
    valid = (pilot["checks"]["valid"] and completed and all(pairing.values()) and identity
             and (rescored is None or all(r["valid"] for r in rescored.values())))

    plain = {"b": [greedy_of(run, "plain") for run in arms["b"]], "c": [greedy_of(run, "mean") for run in arms["c"]]}
    keys = sorted(plain["b"][0])
    if any(sorted(g) != keys for name in ARMS for g in plain[name]):
        raise ValueError("The runs evaluated different maps")
    primary_keys = [k for k in keys if k[0] in PRIMARY]
    transfer_keys = [k for k in keys if k[0] == "heldout78"]
    moved_keys = [k for k in keys if k[0] == "moved"]
    moved = {f"{name}{seed}": sum(g[k] != livr[k] for k in moved_keys)
             for name in ARMS for seed, g in zip(seeds, plain[name])}
    policy_moved = all(m >= settings["moved_maps"] for m in moved.values())

    primary = sign_test(plain["c"], plain["b"], primary_keys)
    transfer = sign_test(plain["c"], plain["b"], transfer_keys)
    zero = [greedy_of(run, "zero") for run in arms["c"]]
    sampled = [greedy_of(run, "sample") for run in arms["c"]]
    test_rows = records(data_root, "test")
    detour_rows = {i for i, r in enumerate(test_rows) if is_detour(r["map_desc"])}
    livr_bench = {r["row_index"]: bool(r["success"]) for r in jsonl(livr_outcomes)}
    rows = sorted(livr_bench)

    def benchmark(run, mode):
        mine = {r["row_index"]: bool(r["success"]) for r in jsonl(Path(run) / f"benchmark-{mode}/responses.jsonl")}
        return {"all": paired(mine, livr_bench, rows),
                "detour": paired(mine, livr_bench, [r for r in rows if r in detour_rows]),
                "no_detour": paired(mine, livr_bench, [r for r in rows if r not in detour_rows])}

    context = {}
    for name, path, populations_ in (("e_round1", e_round1, PRIMARY), ("e_round2", e_round2, ("heldout78",))):
        if path:
            done = [bool(r["success"]) for r in jsonl(path) if r["population"] in populations_]
            context[name] = [sum(done), len(done)]
    return {
        "validity": {"pilot": pilot["checks"], "completed": completed, "pairing": pairing,
                     "identity": identity, "rescored": rescored, "valid": valid},
        "policy_moved": {"differs_from_l": moved, "threshold": settings["moved_maps"], "moved": policy_moved,
                         "extend_to": None if policy_moved else settings["extended_steps"]},
        "primary": primary | {"reading": verdict(primary, valid, policy_moved)},
        "secondary": {
            "transfer": transfer | {"reading": verdict(transfer, valid, policy_moved)},
            "mean_vs_zero": {seed: paired(m, z, primary_keys) for seed, m, z in zip(seeds, plain["c"], zero)},
            "sampled_delta": {seed: sum(s[k] for k in primary_keys) for seed, s in zip(seeds, sampled)},
            "benchmark_vs_l": {f"{name}{seed}": benchmark(run, "plain" if name == "b" else "mean")
                               for name in ARMS for seed, run in zip(seeds, arms[name])},
            "each_vs_l": {f"{name}{seed}": paired(g, livr, primary_keys)
                          for name in ARMS for seed, g in zip(seeds, plain[name])}},
        "diagnostics": {
            "training": {f"{name}{seed}": training_signal(run) for name in ARMS for seed, run in zip(seeds, arms[name])},
            "rescues_and_breaks": {seed: rescues_and_breaks(b, c) for seed, b, c in zip(seeds, arms["b"], arms["c"])},
            "l_success": {"primary": sum(livr[k] for k in primary_keys), "transfer": sum(livr[k] for k in transfer_keys)}},
        "context": context,
        "pilot_rules": pilot["rules"]}
