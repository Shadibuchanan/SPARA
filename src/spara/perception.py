"""Perception arms trained from the base model on answer-only Visual-CoT rows (spara.viscot) and read on
V* (spara.vstar): direct SFT, LIVR's four latents, and global + patch latents with the globals hidden
(docs/PERCEPTION_JEPA_SPEC.md, "Global + patch latents"). Every arm sees the same rows in the same order
for the same steps, in two segments with LIVR's schedule restart between them."""
import json
import re
import time
from pathlib import Path

import torch

from . import viscot, vstar
from .runtime import json_write, load_model, manifest, seed_all
from .serialization import encode
from .training import amp, to_device, train_stage

ARMS = ("sft", "livr", "patch")


def prepare_model(arm, config, device):
    """The base model set up for `arm`. "sft": no latent tokens. "livr": LIVR's four. "patch": LIVR's four,
    then 4 x 4 patch latents per image, each masked to its tile and the prompt's text."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    model, processor = load_model(config | ({"latent_tokens": 0} if arm == "sft" else {}), device=device)
    if arm == "patch":
        from .region_latents import extend_latents
        extend_latents(model, processor, seed=config["seed"], layout="global+patch", mask=True)
    return model, processor


def arm_data(arm, data):
    """The rows as `arm` trains and answers on them: the patch arm's answers may not see the global latents."""
    if arm == "patch":
        from .region_latents import hidden
        return hidden(data)
    return list(data)


def normalized(text):
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip(".")


def validate(model, processor, root, data, path, max_new_tokens, use_latents=True):
    """Greedy answers to held-out Visual-CoT rows, scored by exact match after trimming, lower-casing and
    dropping a final full stop. A check that training took, not a benchmark. `use_latents` False leaves
    the model's latent tokens out of the prompt."""
    model.eval()
    correct = 0
    latents = model.config.livr_latent_tokens if use_latents else []
    if not use_latents:
        data = [(split, {k: v for k, v in row.items() if k != "livr_hide_globals"}) for split, row in data]
    with Path(path).open("w", encoding="utf-8") as log:
        for split, row in data:
            inputs = to_device(encode(processor, root, split, row, training=False,
                                      latents=latents), model.device)
            with torch.no_grad(), amp(model.device):
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                                           temperature=None, top_p=None, top_k=None, use_cache=True)
            response = processor.tokenizer.decode(generated[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            match = normalized(response) == normalized(row["text_output"])
            correct += match
            log.write(json.dumps({"dataset": row["dataset"], "question_id": row["question_id"],
                                  "response": response, "answer": row["text_output"], "correct": match}) + "\n")
    return {"correct": correct, "total": len(data), "accuracy": correct / len(data) if data else None}


def run_arm(arm, config, settings, viscot_root, vstar_root, output, smoke=False):
    """Train `arm` for the settings' two segments, then answer the validation rows and V*. `smoke` cuts
    the rows to the settings' smoke counts: it checks that the arm runs and how long a row takes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cut = settings["smoke"] if smoke else {}
    train = arm_data(arm, viscot.rows(viscot_root, "train"))[:cut.get("train")]
    held = arm_data(arm, viscot.rows(viscot_root, "validation"))[:cut.get("validation")]
    vstar.questions(vstar_root)
    seed_all(config["seed"])
    model, processor = prepare_model(arm, config, "cuda" if torch.cuda.is_available() else "cpu")
    run_manifest = manifest(config, model, processor, viscot_root) | {
        "arm": arm, "settings": settings, "smoke": smoke, "train_rows": len(train), "validation_rows": len(held)}
    json_write(output / "manifest.json", run_manifest)
    # Direct SFT has no bottleneck: both segments are plain. The settings may give a latent arm other
    # stages, e.g. {"stages": {"livr": [2, 2]}} trains LIVR's latents with no bottleneck segment.
    stages = tuple(settings.get("stages", {}).get(arm, (2, 2) if arm == "sft" else (1, 2)))
    if arm == "sft" and stages != (2, 2):
        raise ValueError("Direct SFT has no latent tokens to bottleneck")

    def score(folder):
        """Held-out rows and V* for the model as it stands, under `folder` of the run."""
        folder.mkdir(parents=True, exist_ok=True)
        scores = {"validation": validate(model, processor, viscot_root, held, folder / "validation.jsonl",
                                         settings["validation_max_new_tokens"]), "vstar": {}}
        for tokens in settings["vstar_max_image_tokens"]:
            scores["vstar"][str(tokens)] = vstar.evaluate(
                model, processor, vstar_root, folder / f"vstar-{tokens}", tokens * vstar.PATCH ** 2, "livr",
                run_manifest | {"livr_stage": model.config.livr_stage}, limit=cut.get("vstar"),
                hide_globals=arm == "patch")
        return scores

    summary = {"arm": arm, "stages": stages, "timing": {}, "after_segment": {}}
    last = len(stages) - 1
    for k, (stage, epochs) in enumerate(zip(stages, settings["epochs"])):
        started = time.perf_counter()
        train_stage(model, processor, viscot_root, output / f"segment{k + 1}",
                    config | {"epochs_per_stage": 1 if smoke else epochs}, stage, data=train,
                    final_weights_only=True, save_final=k == last and settings["save_weights"] and not smoke)
        summary["timing"][f"segment{k + 1}_seconds"] = time.perf_counter() - started
        if k != last and settings.get("score_each_segment"):  # read with the stage it was trained in
            summary["after_segment"][str(k + 1)] = score(output / f"after-segment{k + 1}")
            json_write(output / "summary.json", summary)
    if torch.cuda.is_available():
        summary["timing"]["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    summary |= score(output)
    json_write(output / "summary.json", summary)
    return summary


def score_weights(config, checkpoint, viscot_root, output, max_new_tokens, use_latents=True, hide_globals=False):
    """Held-out Visual-CoT rows for saved weights, or for the base model with no latent tokens."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plain = not checkpoint or not json.loads((Path(checkpoint) / "config.json").read_text())["livr_latent_ids"]
    model, processor = load_model(config | ({"latent_tokens": 0} if plain else {}), checkpoint, device="cuda")
    held = viscot.rows(viscot_root, "validation")
    if hide_globals and use_latents:
        from .region_latents import hidden
        held = hidden(held)
    result = validate(model, processor, viscot_root, held, output / "validation.jsonl", max_new_tokens, use_latents)
    json_write(output / "summary.json", result | {"checkpoint": checkpoint, "use_latents": use_latents,
                                                  "hide_globals": hide_globals and use_latents})
    return result
