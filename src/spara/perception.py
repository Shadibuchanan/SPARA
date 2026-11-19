"""Perception benchmarks (docs/PERCEPTION_BENCHMARKS_SPEC.md): MMVP, V* and BLINK (val), after training on
8k Zebra-CoT rows (scripts/prepare_perception.py), for four arms from base Qwen2.5-VL-3B:

- "base": no training, no latents (the reference);
- "livr": LIVR's two stages with its four latents;
- "patch": LIVR's four global latents plus global + patch latents (spara.region_latents), with the globals
  hidden from the answer on every Stage 1 row and every other Stage 2 row, as EGD;
- "scaffold": Scaffolding Minds Stage 1 (scaffold and latent-prediction phases), with Zebra-CoT's first
  reasoning image as the helper.

Every arm is scored by greedy decoding on the same rows; `compare` applies the readings on CPU.
"""
import gc
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch

from .latent_battery import mcnemar
from .runtime import json_write, seed_all
from .serialization import encode, latent_kwargs
from .training import amp, to_device, train_stage

TRAIN = "zebra8k"
BENCHMARKS = ("mmvp", "vstar", "blink")
ARMS = ("base", "livr", "patch", "scaffold")
GREEDY = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}


def rows(root, split):
    folder = Path(root) / split
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    text = (folder / "rows.jsonl").read_text(encoding="utf-8")
    import hashlib
    if hashlib.sha256(text.encode()).hexdigest() != manifest["rows_sha256"]:
        raise ValueError(f"{split}: rows.jsonl does not match its manifest")
    return [json.loads(line) for line in text.splitlines()]


def items(root, limit=None):
    """(split, row) for every evaluation row, benchmark by benchmark; `limit` keeps the first rows of each
    (for pilots)."""
    return [(b, r) for b in BENCHMARKS for r in rows(root, b)[:limit]]


# Scoring -------------------------------------------------------------------------------------

def choices(prompt):
    """The option texts listed in a prompt as "(A) text" lines, in order."""
    return [m.group(2).strip() for m in re.finditer(r"^\(([A-Z])\)\s*(.+)$", prompt, re.M)]


def extract(text, options, texts=None):
    """The option letter an answer names: a leading letter ("B", "(B)", "B."), else the first "(X)", else
    the first standalone letter among the options; failing those, the one option whose text the answer
    equals or contains (case-insensitive, `texts` from the prompt). None if there is none."""
    letters = "".join(chr(65 + i) for i in range(options))
    text = text.strip()
    found = (re.match(rf"^\(?([{letters}])\)?(?:[\s.,:)]|$)", text) or re.search(rf"\(([{letters}])\)", text)
             or re.search(rf"\b([{letters}])\b", text))
    if found:
        return found.group(1)
    answer = text.lower().strip(" .")
    if texts and answer:
        exact = [i for i, t in enumerate(texts[:options]) if t.lower().strip(" .") == answer]
        named = [i for i, t in enumerate(texts[:options]) if t.lower().strip(" .") in answer]
        pick = exact if len(exact) == 1 else named if len(named) == 1 else []
        if pick:
            return chr(65 + pick[0])
    return None


def summarise(results):
    """Accuracy per benchmark and category, MMVP's pair accuracy (both questions of a pair right), and
    the mean of the three benchmarks."""
    out, groups = {}, defaultdict(list)
    for r in results:
        groups[r["benchmark"]].append(r)
    for name, group in groups.items():
        by = defaultdict(list)
        for r in group:
            by[r["category"]].append(r["correct"])
        out[name] = {"correct": sum(r["correct"] for r in group), "items": len(group),
                     "accuracy": sum(r["correct"] for r in group) / len(group),
                     "unparsed": sum(r["letter"] is None for r in group),
                     "categories": {c: sum(v) / len(v) for c, v in sorted(by.items())}}
        if name == "mmvp":
            pairs = defaultdict(list)
            for r in group:
                pairs[r["pair"]].append(r["correct"])
            out[name]["pair_accuracy"] = sum(all(v) for v in pairs.values()) / len(pairs)
    if all(b in out for b in BENCHMARKS):
        out["mean"] = sum(out[b]["accuracy"] for b in BENCHMARKS) / len(BENCHMARKS)
    return out


# Evaluation ----------------------------------------------------------------------------------

def generate_livr(model, processor, root, max_new_tokens):
    eos = model.generation_config.eos_token_id

    def answer(split, row):
        prompt = to_device(encode(processor, root, split, row, training=False, **serialization_of(model)),
                           model.device)
        with torch.no_grad(), amp(model.device):
            out = model.generate(**prompt, **GREEDY, max_new_tokens=max_new_tokens, use_cache=True)
        return processor.tokenizer.decode(out[0, prompt["input_ids"].shape[-1]:], skip_special_tokens=True)
    return answer


def serialization_of(model):
    """`encode`'s latents for `model`: none for the base model, LIVR's four by default, region latents."""
    if not model.config.livr_latent_ids:
        return {"latents": []}
    return latent_kwargs(model)


def generate_scaffold(model, processor, root, max_new_tokens):
    from .scaffolding import serialization as scaffold_serialization
    from .scaffolding.decoding import LatentDecoder
    decoder = LatentDecoder(model, GREEDY | {"max_new_tokens": max_new_tokens}, "none")

    def answer(split, row):
        prompt = to_device(scaffold_serialization.prompt(processor, root, split, row), model.device)
        with torch.no_grad(), amp(model.device):
            rollout = decoder.generate(prompt)
        return processor.tokenizer.decode(rollout.ids, skip_special_tokens=True)
    return answer


def evaluate(answer, data, path, flag=None):
    """Greedy answers to every (split, row); `flag` marks rows (e.g. hide the global latents)."""
    results = []
    with Path(path).open("w", encoding="utf-8") as log:
        for count, (split, row) in enumerate(data):
            text = answer(split, row | (flag or {}))
            letter = extract(text, row["options"], choices(row["text_input"]))
            record = {"benchmark": row["benchmark"], "category": row["category"], "id": row["id"],
                      "pair": row.get("pair"), "gold": row["text_output"], "answer": text, "letter": letter,
                      "correct": letter == row["text_output"]}
            results.append(record)
            log.write(json.dumps(record) + "\n"); log.flush()
            if count % 200 == 0:
                print(f"eval {count + 1}/{len(data)}", flush=True)
    return summarise(results)


# Arms ----------------------------------------------------------------------------------------

def hidden_every(data, period):
    """(split, row) pairs with the global latents hidden on every `period`-th row (1: every row)."""
    return [(s, r | {"livr_hide_globals": True}) if k % period == 0 else (s, r) for k, (s, r) in enumerate(data)]


def run_arm(arm, config, root, output, settings, describe, load_model, scaffold_config=None, limit=None):
    """Train `arm` (except "base") on the Zebra-CoT rows, then evaluate it on the three benchmarks.
    `load_model(config)` returns a fresh base model and processor; `describe(model, processor)` the run
    manifest, written once the arm's model is loaded; `limit` cuts every set (pilots)."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    train = [(TRAIN, r) for r in rows(root, TRAIN)][:limit or settings.get("train_rows")]
    data = items(root, limit)
    extra = {"arm": arm, "settings": settings, "train_rows": len(train), "eval_rows": len(data), "limit": limit,
             "data_manifests": {s: json.loads((Path(root) / s / "manifest.json").read_text())
                                for s in (TRAIN, *BENCHMARKS)}}
    write = lambda model, processor: json_write(output / "manifest.json", describe(model, processor) | extra)
    seed_all(config["seed"])
    tuned = config | {"epochs_per_stage": settings["epochs_per_stage"]}
    summary, times = {"arm": arm}, {}
    if arm == "scaffold":
        model, processor = train_scaffold_arm(scaffold_config | {"seed": config["seed"]}, root, output, train,
                                              write)
        times["training"] = time.time() - began
        answer = generate_scaffold(model, processor, root, settings["max_new_tokens"])
    else:
        model, processor = load_model(config | {"latent_tokens": 0} if arm == "base" else config)
        if arm == "patch":
            from .region_latents import extend_latents
            extend_latents(model, processor, config["seed"], "global+patch", mask=True, grid=settings["grid"],
                           text=settings["patch_text"])
        write(model, processor)
        if arm != "base":
            first = hidden_every(train, 1) if arm == "patch" else train
            second = hidden_every(train, 2) if arm == "patch" else train
            train_stage(model, processor, root, output / "stage1", tuned, 1, final_weights_only=True, data=first,
                        save_final=False)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            train_stage(model, processor, root, output / "stage2", tuned, 2, final_weights_only=True, data=second,
                        save_final=settings.get("save_weights", True))
            times["training"] = time.time() - began
        model.eval()
        model.config.livr_stage = 2
        answer = generate_livr(model, processor, root, settings["max_new_tokens"])
    seed_all(config["seed"])
    summary["evaluation"] = evaluate(answer, data, output / "answers.jsonl")
    if arm == "patch":  # descriptive: the answer reads the image and the patch latents only
        summary["evaluation_globals_hidden"] = evaluate(answer, data, output / "answers-hidden.jsonl",
                                                        {"livr_hide_globals": True})
    times["total"] = time.time() - began
    summary |= {"seconds": times,
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    json_write(output / "summary.json", summary)
    print(json.dumps(summary, indent=1), flush=True)
    return summary


def train_scaffold_arm(config, root, output, train, write=None):
    """Scaffolding Minds Stage 1 on the training rows: the scaffold phase (the encoder, through the frozen
    base model), then the latent-prediction phase from a fresh base model."""
    from .scaffolding.model import ScaffoldingEncoder, add_latent_projection, load_token_rows, train_language
    from .scaffolding.runtime import load_base
    from .scaffolding.training import scaffold_targets, train_predict, train_scaffold
    config = config | {"train_rows": len(train)}
    vlm, processor = load_base(config, device="cuda" if torch.cuda.is_available() else "cpu")
    encoder = ScaffoldingEncoder.from_vlm(vlm, config["latent_tokens"], config["scaffold"]["pooling_heads"])
    load_token_rows(vlm, encoder.context)
    vlm.requires_grad_(False)
    if write:
        write(vlm, processor)
    train_scaffold(vlm, encoder, processor, root, output / "scaffold", config, data=train)
    context = encoder.context
    del vlm
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    model, processor = load_base(config, device=encoder_device(encoder))
    load_token_rows(model, context)
    add_latent_projection(model, config["predict"]["latent_projection"])
    targets = scaffold_targets(encoder, processor, root, [r for _, r in train], config, model.device)
    del encoder
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    train_language(model)
    train_predict(model, processor, targets, root, output / "predict", config, data=train)
    model.eval()
    model.requires_grad_(False)
    return model, processor


def encoder_device(encoder):
    return next(encoder.parameters()).device


# Readings (CPU) ------------------------------------------------------------------------------

def rescore(root, run, name="answers.jsonl"):
    """Score a run's saved answers again with the current `extract` (no GPU); rewrites the file and
    returns the summary."""
    prompts = {(r["benchmark"], r["id"]): r for b in BENCHMARKS for r in rows(root, b)}
    path, results = Path(run) / name, []
    for record in map(json.loads, path.read_text().splitlines()):
        row = prompts[record["benchmark"], record["id"]]
        record["letter"] = extract(record["answer"], row["options"], choices(row["text_input"]))
        record["correct"] = record["letter"] == record["gold"]
        results.append(record)
    path.write_text("".join(json.dumps(r) + "\n" for r in results))
    return summarise(results)


def outcomes(run):
    return {(r["benchmark"], r["id"]): r["correct"]
            for r in map(json.loads, (Path(run) / "answers.jsonl").read_text().splitlines())}


def paired(first, second, keys):
    b = sum(first[k] and not second[k] for k in keys)
    c = sum(second[k] and not first[k] for k in keys)
    return {"first": sum(first[k] for k in keys), "second": sum(second[k] for k in keys), "items": len(keys),
            "only_first": b, "only_second": c, "p": mcnemar(b, c)}


def compare(runs):
    """The readings fixed in the specification. `runs`: {arm: run directory} for all four arms."""
    table = {arm: outcomes(path) for arm, path in runs.items()}
    keys = sorted(set.intersection(*(set(t) for t in table.values())))
    if any(len(t) != len(keys) for t in table.values()):
        raise ValueError("The arms were not scored on the same rows")
    by = lambda name: [k for k in keys if k[0] == name]
    tests = {f"{a}_vs_{b}": {"all": paired(table[a], table[b], keys),
                             **{name: paired(table[a], table[b], by(name)) for name in BENCHMARKS}}
             for a, b in (("patch", "livr"), ("livr", "base"), ("patch", "base"), ("scaffold", "livr"),
                          ("scaffold", "base"))}
    primary = tests["patch_vs_livr"]["all"]
    better, worse = primary["only_first"] > primary["only_second"], primary["only_first"] < primary["only_second"]
    reading = ("patch latents help perception" if primary["p"] < 0.05 and better else
               "patch latents hurt perception" if primary["p"] < 0.05 and worse else
               "no detectable difference between patch and LIVR latents")
    summaries = {arm: json.loads((Path(path) / "summary.json").read_text()) for arm, path in runs.items()}
    return {"items": len(keys), "tests": tests, "reading": reading,
            "accuracy": {arm: {k: v for k, v in s["evaluation"].items()} for arm, s in summaries.items()},
            "patch_globals_hidden": summaries.get("patch", {}).get("evaluation_globals_hidden")}


def main(argv=None):
    """python -m spara.perception --arm ARM --data-root data/perception --output RUN [--limit N]
    python -m spara.perception --compare base=RUN livr=RUN patch=RUN scaffold=RUN --output comparison.json"""
    import argparse
    parser = argparse.ArgumentParser(prog="python -m spara.perception")
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--compare", nargs="*", help="arm=run directory for every arm (CPU)")
    parser.add_argument("--config", default="configs/livr.json")
    parser.add_argument("--settings", default="configs/perception.json")
    parser.add_argument("--scaffold-config", default="configs/scaffolding_perception.json")
    parser.add_argument("--data-root", default="data/perception")
    parser.add_argument("--base-model", help="Verified local model downloaded by `spara.cli fetch-model`")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, help="Pilot: keep the first N training rows and N rows per benchmark")
    args = parser.parse_args(argv)
    if args.compare is not None:
        result = compare(dict(item.split("=", 1) for item in args.compare))
        json_write(Path(args.output), result)
        print(json.dumps({k: result[k] for k in ("items", "reading")}, indent=1))
        return
    if not args.arm:
        parser.error("Give --arm or --compare")
    from .runtime import load_model, manifest, read_config
    config = read_config(args.config)
    settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
    scaffold_config = None
    if args.base_model:
        config["local_model_path"] = str(Path(args.base_model).resolve())
    if args.arm == "scaffold":
        from .scaffolding.runtime import read_config as read_scaffold
        scaffold_config = read_scaffold(args.scaffold_config)
        if args.base_model:
            scaffold_config["local_model_path"] = config["local_model_path"]
    if not torch.cuda.is_available():
        raise RuntimeError("Use an allocated GPU for the actual 3B model")
    if args.arm == "scaffold":
        from .scaffolding.runtime import manifest as scaffold_manifest
        describe = lambda model, processor: scaffold_manifest(scaffold_config, model, processor,
                                                              Path(args.data_root) / TRAIN)
    else:
        describe = lambda model, processor: manifest(config, model, processor, Path(args.data_root) / TRAIN)
    run_arm(args.arm, config, args.data_root, args.output, settings, describe,
            lambda c: load_model(c, device="cuda"), scaffold_config, args.limit)


if __name__ == "__main__":
    main()
