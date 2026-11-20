"""Where a model looks on a V* image when it starts its answer: the language model's attention from the
last prompt token (and from each latent token) to the image tokens, per layer, averaged over heads.
V* ships a target box per question, so each map also gives the share of image attention that lands on
the target. `capture` keeps only the rows asked for, so a native-size image (10,000 tokens) fits."""
import json
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from . import vstar

TARGETS = "targets.json"
LAYER_BANDS = 3  # the report splits the layers into this many equal bands
MAX_OPENING = 2  # punctuation tokens a reply may open with before its letter


def fetch_targets(root, revision=vstar.REVISION):
    """Download each question's released annotation (target objects and boxes, x y w h in image pixels)
    into one file. The annotations are small files stored in git, so there is no content hash to check;
    each is checked against the question it belongs to."""
    root = Path(root)
    targets = {}
    for row in vstar.questions(root):
        name = str(Path(row["image"]).with_suffix(".json"))
        url = f"https://huggingface.co/datasets/{vstar.DATASET}/resolve/{revision}/{name}"
        with urllib.request.urlopen(url, timeout=120) as source:
            entry = json.load(source)
        if entry["question"] not in row["text"] or len(entry["bbox"]) != len(entry["target_object"]):
            raise ValueError(f"Annotation does not match its question: {name}")
        targets[row["image"]] = {"objects": entry["target_object"], "boxes": entry["bbox"]}
    (root / TARGETS).write_text(json.dumps({"revision": revision, "targets": targets}, indent=2) + "\n",
                                encoding="utf-8")
    return len(targets)


def targets(root):
    return json.loads((Path(root) / TARGETS).read_text(encoding="utf-8"))["targets"]


def attention_rows(module, hidden_states, attention_mask, position_embeddings, rows):
    """The attention weights of one decoder attention module for its last `rows` query positions,
    [batch, heads, rows, keys], computed from the module's own inputs as the library's eager
    implementation does. Only the kept rows are formed, so a 10,000-token prompt costs megabytes."""
    import torch
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, repeat_kv
    batch, length, _ = hidden_states.shape
    cos, sin = position_embeddings
    sections = module.rope_scaling["mrope_section"]
    query = module.q_proj(hidden_states[:, -rows:]).view(batch, rows, -1, module.head_dim).transpose(1, 2)
    key = module.k_proj(hidden_states).view(batch, length, -1, module.head_dim).transpose(1, 2)
    query, _ = apply_multimodal_rotary_pos_emb(query, query, cos[..., -rows:, :], sin[..., -rows:, :], sections)
    key, _ = apply_multimodal_rotary_pos_emb(key, key, cos, sin, sections)
    key = repeat_kv(key, module.num_key_value_groups)
    scores = torch.matmul(query, key.transpose(2, 3)).float() / module.head_dim ** 0.5
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, -rows:, :length].float()
    position = torch.arange(length, device=scores.device)
    future = position[None, :] > position[-rows:, None]  # causal, whether or not the mask already says so
    return scores.masked_fill(future, float("-inf")).softmax(-1)


@contextmanager
def capture(model, rows):
    """Record every decoder layer's attention for the last `rows` query positions, averaged over heads.
    Yields a list that fills with one [rows, keys] tensor per layer on each uncached forward pass."""
    import torch
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLAttention
    modules = [m for m in model.modules() if isinstance(m, Qwen2_5_VLAttention)]
    if len(modules) != model.config.num_hidden_layers:
        raise ValueError("Expected one attention module per decoder layer")
    kept = []

    def hook(module, args, kwargs):
        if args or kwargs.get("past_key_value") is not None:
            raise ValueError("Attention capture reads one uncached forward pass with keyword arguments")
        with torch.no_grad():
            weights = attention_rows(module, kwargs["hidden_states"], kwargs.get("attention_mask"),
                                     kwargs["position_embeddings"], rows)
        kept.append(weights[0].mean(0).cpu())

    handles = [module.register_forward_pre_hook(hook, with_kwargs=True) for module in modules]
    try:
        yield kept
    finally:
        for handle in handles:
            handle.remove()


def cells(box, scale_x, scale_y, grid_w, grid_h):
    """The grid cells a box touches, as (x0, y0, x1, y1) with exclusive ends. `box` is x y w h in the
    original image's pixels; the scales take it to the resized image."""
    x, y, w, h = box
    x0 = min(grid_w - 1, max(0, int(x * scale_x // vstar.PATCH)))
    y0 = min(grid_h - 1, max(0, int(y * scale_y // vstar.PATCH)))
    x1 = min(grid_w, max(x0 + 1, -int(-((x + w) * scale_x) // vstar.PATCH)))
    y1 = min(grid_h, max(y0 + 1, -int(-((y + h) * scale_y) // vstar.PATCH)))
    return x0, y0, x1, y1


def measure(model, processor, root, output, max_pixels, run_manifest, limit=None, hide_globals=False):
    """One forward pass per question in the project's prompt format. Writes maps.npz per question
    (attention to image tokens, [layers, rows, grid_h, grid_w], rows = the latent tokens then the two
    assistant-header tokens, then any punctuation the reply opens with; the last row is the position that
    produces the answer's letter) and rows.jsonl."""
    import numpy as np
    import torch
    from PIL import Image
    from .runtime import json_write
    from .training import amp, to_device
    output = Path(output)
    (output / "maps").mkdir(parents=True, exist_ok=False)
    latents = vstar.image_latents(model)
    rows_kept = len(latents) + 2
    boxes = targets(root)
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    json_write(output / "manifest.json", run_manifest | {
        "max_pixels": max_pixels, "latents": latents, "hide_globals": hide_globals and bool(latents),
        "rows": latents + ["<|im_start|>", "assistant"]})
    model.eval()
    questions = vstar.questions(root)[:limit]
    with (output / "rows.jsonl").open("w", encoding="utf-8") as log:
        for index, row in enumerate(questions):
            batch, tokens = vstar.encode(processor, root, row, max_pixels, "livr", latents)
            with Image.open(vstar.image_file(root, row)) as source:
                original = source.size
            grid_h, grid_w = (int(v) for v in batch["image_grid_thw"][0, 1:] // 2)
            if grid_h * grid_w != tokens:
                raise ValueError("Image grid does not match the image tokens")
            if hide_globals and latents:
                batch["livr_hide_globals"] = torch.tensor([True])
            # A reply may open with punctuation ("(B)", ".(A)"): the letter is then chosen one or two positions
            # later. Those opening tokens are appended, so the last row is where the letter itself is produced.
            opening = []
            for _ in range(MAX_OPENING + 1):
                inputs = to_device(batch, model.device)
                with capture(model, rows_kept + len(opening)) as kept, torch.no_grad(), amp(model.device):
                    logits = model(**inputs, use_cache=False).logits
                chosen = logits[0, -1].argmax().reshape(1, 1).cpu()
                first = processor.tokenizer.decode(chosen[0])
                if first.strip().strip("(.:") or len(opening) == MAX_OPENING:
                    break
                opening.append(first)
                batch["input_ids"] = torch.cat([batch["input_ids"], chosen], 1)
                batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(chosen)], 1)
            attention = torch.stack(kept)  # [layers, rows, keys]
            on_image = attention[:, :, (batch["input_ids"][0] == image_id)]
            maps = on_image.reshape(len(kept), rows_kept + len(opening), grid_h, grid_w)
            scale_x, scale_y = grid_w * vstar.PATCH / original[0], grid_h * vstar.PATCH / original[1]
            inside = torch.zeros(grid_h, grid_w, dtype=torch.bool)
            for box in boxes[row["image"]]["boxes"]:
                x0, y0, x1, y1 = cells(box, scale_x, scale_y, grid_w, grid_h)
                inside[y0:y1, x0:x1] = True
            share = on_image.sum(-1)  # [layers, rows]: attention that goes to the image at all
            ids = batch["input_ids"][0]
            latent_columns = torch.isin(ids, torch.tensor(model.config.livr_latent_ids or [-1]))
            if int(latent_columns.sum()) != len(latents):
                raise ValueError("The prompt does not carry each latent token once")
            on_latents = attention[:, -1][:, latent_columns]  # [layers, latents]: from the answer position
            target = maps[:, :, inside].sum(-1) / share.clamp_min(1e-12)  # of that, the share on the target
            np.savez_compressed(output / "maps" / f"{index:03d}.npz", maps=maps.numpy().astype(np.float16))
            log.write(json.dumps({
                "row_index": index, "question_id": row["question_id"], "category": row["category"],
                "image": row["image"], "label": row["label"], "first_token": first, "opening": opening, "image_tokens": tokens,
                "grid": [grid_h, grid_w], "target_cells": int(inside.sum()),
                "chance": float(inside.float().mean()),
                "image_share": share[:, -1].tolist(), "target_share": target[:, -1].tolist(),
                "prompt_tokens": int(ids.shape[0]), "latent_share_each": on_latents.tolist(),
                "latent_target_share": target[:, :len(latents)].mean(1).tolist() if latents else None}) + "\n")
            log.flush()
            print(f"{index + 1}/{len(questions)} {first!r}", flush=True)
    return output


def heat(values):
    """A [h, w] array in 0..1 as RGB: black, red, yellow, white."""
    import numpy as np
    v = np.clip(values, 0, 1)[..., None]
    return np.clip(np.concatenate([3 * v, 3 * v - 1, 3 * v - 2], -1), 0, 1)


def overlay(picture, grid, size, boxes=(), clip=99.5):
    """`picture` at `size` with `grid` (attention per image token) laid over it; the grid is scaled to its
    own `clip` percentile, so colours compare places within one map, not one map with another."""
    import numpy as np
    from PIL import Image, ImageDraw
    base = np.asarray(picture.convert("L").convert("RGB").resize(size), dtype=np.float32) / 255
    if grid is not None:
        top = max(float(np.percentile(grid, clip)), 1e-12)
        level = np.asarray(Image.fromarray((np.clip(grid / top, 0, 1) * 255).astype(np.uint8)).resize(
            size, Image.BILINEAR), dtype=np.float32) / 255
        alpha = (0.25 + 0.75 * level)[..., None]
        base = (1 - alpha) * base + alpha * heat(level)
    else:
        base = np.asarray(picture.convert("RGB").resize(size), dtype=np.float32) / 255
    panel = Image.fromarray((base * 255).astype(np.uint8))
    draw = ImageDraw.Draw(panel)
    for x, y, w, h in boxes:
        sx, sy = size[0] / picture.width, size[1] / picture.height
        frame = [x * sx - 2, y * sy - 2, (x + w) * sx + 2, (y + h) * sy + 2]
        draw.rectangle(frame, outline=(0, 255, 255), width=2)
    return panel


def band_means(values):
    """Mean of a per-layer list over each of LAYER_BANDS equal bands of layers, then over all layers."""
    step = len(values) // LAYER_BANDS
    bands = [sum(values[k * step:(k + 1) * step]) / step for k in range(LAYER_BANDS)]
    return bands + [sum(values) / len(values)]


def report(root, runs, answers, output, height=260):
    """One picture per question: the image with its target box, then each run's answer-position map
    (mean over layers and heads), then the latent tokens' map for runs that have latents. `runs` is
    {name: measure() folder}; `answers` is {name: responses.jsonl of that model's scored V* run}.
    Writes pictures/, index.html and summary.json."""
    import numpy as np
    from PIL import Image, ImageDraw
    root, output = Path(root), Path(output)
    (output / "pictures").mkdir(parents=True, exist_ok=False)
    boxes = targets(root)
    questions = vstar.questions(root)
    logs = {name: [json.loads(line) for line in (Path(folder) / "rows.jsonl").read_text().splitlines()]
            for name, folder in runs.items()}
    scored = {name: [json.loads(line) for line in Path(path).read_text().splitlines()]
              for name, path in answers.items()}
    for name in runs:  # rescored with the current rule, so every run is read the same way
        for entry, row in zip(scored[name], questions):
            entry["answer"] = vstar.extract(entry["response"], vstar.options(row["text"]))[0]
            entry["correct"] = entry["answer"] == row["label"]
    latent_runs = [name for name in runs if logs[name][0]["latent_target_share"] is not None]
    cards = []
    for index, row in enumerate(questions):
        with Image.open(vstar.image_file(root, row)) as source:
            picture = source.convert("RGB")
        size = (max(120, round(height * picture.width / picture.height)), height)
        target = boxes[row["image"]]
        panels = [("image, target in cyan", overlay(picture, None, size, target["boxes"]))]
        for name, folder in runs.items():
            maps = np.load(Path(folder) / "maps" / f"{index:03d}.npz")["maps"].astype(np.float32)
            entry, result = logs[name][index], scored[name][index]
            mark = "right" if result["correct"] else "wrong"
            share = band_means(entry["target_share"])[-1]
            panels.append((f"{name}: {result['answer']} ({mark}), {100 * share:.1f}% on target",
                           overlay(picture, maps[:, -1].mean(0), size, target["boxes"])))
            if name in latent_runs:
                count = len(entry["latent_share_each"][0])  # the latent rows come first
                panels.append((f"{name} latents ({count})", overlay(picture, maps[:, :count].mean((0, 1)), size,
                                                                     target["boxes"])))
        sheet = Image.new("RGB", (sum(p.width for _, p in panels) + 6 * (len(panels) - 1), height + 18), "white")
        draw, x = ImageDraw.Draw(sheet), 0
        for title, panel in panels:
            sheet.paste(panel, (x, 18))
            draw.text((x + 2, 3), title[:max(12, panel.width // 6)], fill=(0, 0, 0))
            x += panel.width + 6
        file = f"pictures/{index:03d}.jpg"
        sheet.save(output / file, format="JPEG", quality=82)
        cards.append({"file": file, "index": index, "category": row["category"], "label": row["label"],
                      "question": row["text"].split("\n")[0], "objects": target["objects"],
                      "chance": logs[next(iter(runs))][index]["chance"],
                      "correct": {name: scored[name][index]["correct"] for name in runs}})

    def mean(values):
        values = list(values)
        return sum(values) / len(values) if values else None

    summary = {}
    for name in runs:
        entries = logs[name]
        right = [scored[name][i]["correct"] for i in range(len(entries))]
        groups = {"all": entries, "right": [e for e, r in zip(entries, right) if r],
                  "wrong": [e for e, r in zip(entries, right) if not r]} | {
            category: [e for e in entries if e["category"] == category] for category in vstar.CATEGORIES}
        summary[name] = {"correct": sum(right), "first_token_agrees": sum(
            scored[name][i]["response"].strip().startswith(("".join(entries[i].get("opening", [])) + entries[i]["first_token"]).strip())
            for i in range(len(entries)))} | {
            group: {"questions": len(members),
                    "chance": mean(e["chance"] for e in members),
                    "target_share": [mean(band_means(e["target_share"])[k] for e in members) for k in range(LAYER_BANDS + 1)],
                    "image_share": [mean(band_means(e["image_share"])[k] for e in members) for k in range(LAYER_BANDS + 1)],
                    "latent_share": [mean(band_means([sum(layer) for layer in e["latent_share_each"]])[k]
                                          for e in members) for k in range(LAYER_BANDS + 1)],
                    "image_tokens": mean(e["image_tokens"] for e in members),
                    "prompt_tokens": mean(e["prompt_tokens"] for e in members),
                    "latent_target_share": mean(band_means(e["latent_target_share"])[-1] for e in members)
                    if name in latent_runs else None}
            for group, members in groups.items()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    page = ["<!doctype html><meta charset='utf-8'><title>V* attention maps</title>",
            "<style>body{font:14px system-ui;margin:16px;background:#fff;color:#111}img{max-width:100%;height:auto}"
            ".card{margin:0 0 22px}.q{margin:0 0 4px}.over{overflow-x:auto}</style>",
            "<h1>V* attention maps</h1>",
            "<p>One row per question, in file order. Each map is the attention from the position that produces "
            "the answer to the image tokens, averaged over all layers and heads, and scaled to its own "
            "99.5th percentile. The cyan box is the released target.</p>"]
    for card in cards:
        verdict = ", ".join(f"{name} {'right' if ok else 'wrong'}" for name, ok in card["correct"].items())
        page.append(f"<div class='card'><p class='q'><b>{card['index'] + 1}. {card['question']}</b> "
                    f"[{card['category']}, answer {card['label']}; target covers {100 * card['chance']:.2f}% of "
                    f"the image; {verdict}]</p><div class='over'><img loading='lazy' src='{card['file']}'></div></div>")
    (output / "index.html").write_text("\n".join(page), encoding="utf-8")
    return summary
