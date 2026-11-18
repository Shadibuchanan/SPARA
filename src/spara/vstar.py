"""V* (Wu and Xie, 2024): the released 191 multiple-choice questions on large images, and a zero-shot
evaluation. Images are resized here, once, to a stated pixel limit, so the number of image tokens is
the evaluation's own setting rather than the processor's default."""
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path

from .data import download, sha256

DATASET = "craigwu/vstar_bench"
REVISION = "d9ae62c903da0c98336e85c5ee89cd863b04b4da"
QUESTIONS = "test_questions.jsonl"
CATEGORIES = {"direct_attributes": 115, "relative_position": 76}
PATCH = 28  # one merged image token covers 28 x 28 pixels
FORMATS = ("chat", "livr")
GENERATION = {"do_sample": False, "max_new_tokens": 16, "temperature": None, "top_p": None, "top_k": None}
ASSISTANT = "<|im_start|>assistant"


def fetch(root, revision=REVISION):
    """Download the questions and the images they name, each checked against the hub's recorded hash."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/datasets/{DATASET}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(api, timeout=120) as response:
        info = json.load(response)
    if info["sha"] != revision:
        raise ValueError("Dataset revision must be an immutable commit hash")
    listed = {entry["rfilename"]: entry for entry in info["siblings"]}

    def get(name):
        entry = listed[name]
        expected = entry.get("lfs", {}).get("sha256")
        path = root.joinpath(*name.split("/")).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Unsafe dataset filename: {name}")
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{name}"
        if expected:
            download(url, path, expected)
        elif not path.exists():  # a small file stored in git: no content hash to check, so check its size
            path.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=120) as source:
                path.write_bytes(source.read())
        if entry.get("size") is not None and path.stat().st_size != entry["size"]:
            raise ValueError(f"Dataset size mismatch: {name}")
        return {"sha256": sha256(path), "bytes": path.stat().st_size}

    files = {QUESTIONS: get(QUESTIONS)}
    for row in questions(root, audit=False):
        if row["image"] not in files:
            files[row["image"]] = get(row["image"])
    (root / "manifest.json").write_text(json.dumps(
        {"dataset": DATASET, "revision": revision, "files": files}, indent=2) + "\n", encoding="utf-8")
    questions(root)
    return root


def questions(root, audit=True):
    """The released rows in file order: image, text (question, lettered options and the answer instruction),
    category, question_id and label (the correct letter)."""
    root = Path(root)
    rows = [json.loads(line) for line in (root / QUESTIONS).read_text(encoding="utf-8").splitlines() if line.strip()]
    if audit:
        if dict(Counter(row["category"] for row in rows)) != CATEGORIES:
            raise ValueError("V* questions do not match the released 115 + 76")
        for row in rows:
            if row["label"] not in options(row["text"]):
                raise ValueError(f"Question {row['question_id']}: the label is not one of its options")
            image_file(root, row)
    return rows


def image_file(root, row):
    base = Path(root).resolve()
    path = base.joinpath(*Path(row["image"]).parts).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"Missing or unsafe V* image: {path}")
    return path


def options(text):
    """{letter: text} for the options a question offers, in order."""
    return dict(re.findall(r"^\(([A-Z])\) *(.*?) *$", text, flags=re.MULTILINE))


def extract(response, choices):
    """(letter, how) for the option an answer gives, or (None, None). `choices` is `options(text)`.
    - how "letter": the answer starts with an offered letter in brackets ("(B) left"), alone ("B") or before
      punctuation ("B.", "B) left", "B: left"), or states it ("Answer: B", "the answer is (B)").
    - how "text": the whole answer is exactly one option's words ("left"), ignoring case and a final full
      stop. A model tuned on short answers may reply this way; the summary counts these separately.
    Leading full stops and colons are dropped first. Anything else is unparsed and counts as wrong. A bare
    letter before a word ("A dog") is not a choice."""
    text = response.strip().lstrip(".:").lstrip()  # the livr format has no newline after "assistant": ".(A)", ":B"
    allowed = "".join(choices)
    start = re.match(rf"\(([{allowed}])\)|([{allowed}])(?=[.):,]|\s*$)", text)
    if start:
        return start.group(1) or start.group(2), "letter"
    stated = re.search(rf"answer(?: is)?:?\s*\(?([{allowed}])\)?(?![A-Za-z0-9])", text, flags=re.IGNORECASE)
    if stated:
        return stated.group(1), "letter"
    plain = text.lower().rstrip(".").strip()
    same = [letter for letter, words in choices.items() if words.lower().rstrip(".").strip() == plain]
    return (same[0], "text") if plain and len(same) == 1 else (None, None)


def resized(image, max_pixels):
    """The image at the size the processor will keep: sides rounded to the patch size, at most `max_pixels`."""
    from qwen_vl_utils import smart_resize
    height, width = smart_resize(image.height, image.width, factor=PATCH, min_pixels=4 * PATCH * PATCH,
                                 max_pixels=max_pixels)
    return image if (width, height) == image.size else image.resize((width, height))


def encode(processor, root, row, max_pixels, prompt_format="chat", latents=()):
    """One question as model input: the image, then the released text.
    - "chat": the processor's own chat template with its generation prompt, as the base model was trained.
    - "livr": this project's serialization, the user turn, then `latents`, then the assistant header
      without a newline (spara.serialization.encode)."""
    from PIL import Image
    if prompt_format not in FORMATS:
        raise ValueError(f"Unknown prompt format: {prompt_format}")
    if prompt_format == "chat" and latents:
        raise ValueError("The chat format carries no latent tokens")
    with Image.open(image_file(root, row)) as source:
        picture = resized(source.convert("RGB"), max_pixels)
    messages = [{"role": "user", "content": [{"type": "image", "image": picture},
                                             {"type": "text", "text": row["text"]}]}]
    if prompt_format == "chat":
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt += "".join(latents) + ASSISTANT
    batch = dict(processor(text=[prompt], images=[picture], padding=False, return_tensors="pt"))
    tokens = int((batch["input_ids"] == processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum())
    if tokens != (picture.width // PATCH) * (picture.height // PATCH):
        raise ValueError("The processor resized the image again; the pixel limit is not the one stated")
    return batch, tokens


def image_latents(model):
    """The latent tokens a single-image prompt carries for `model`: none, LIVR's, or globals plus one image's
    patch latents (spara.region_latents, layout global+patch)."""
    tokens = getattr(model.config, "livr_latent_tokens", None) or []
    if isinstance(tokens, dict):
        if "per_image" not in tokens:
            raise ValueError("Region latents by map grid size do not apply to V* images")
        return tokens["globals"] + tokens["per_image"][0]
    return list(tokens)


def summarize(results):
    def part(rows):
        correct = sum(r["correct"] for r in rows)
        return {"correct": correct, "total": len(rows), "accuracy": correct / len(rows) if rows else None,
                "unparsed": sum(r["answer"] is None for r in rows),
                "answered_in_words": sum(r["how"] == "text" for r in rows)}
    return {"overall": part(results)} | {
        category: part([r for r in results if r["category"] == category]) for category in CATEGORIES}


def evaluate(model, processor, root, output, max_pixels, prompt_format, run_manifest, limit=None, hide_globals=False):
    """Greedy answers to every question in file order; one line per question, and a running summary.
    `hide_globals`: the answer may not see the global latents (a global + patch model trained that way)."""
    import torch
    from .runtime import json_write
    from .training import amp, to_device
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    latents = image_latents(model) if prompt_format == "livr" else []
    json_write(output / "manifest.json", run_manifest | {
        "max_pixels": max_pixels, "prompt_format": prompt_format, "latents": latents, "generation": GENERATION,
        "hide_globals": hide_globals})
    model.eval()
    results = []
    rows = questions(root)[:limit]
    with (output / "responses.jsonl").open("w", encoding="utf-8") as log:
        for index, row in enumerate(rows):
            batch, tokens = encode(processor, root, row, max_pixels, prompt_format, latents)
            if hide_globals:
                batch["livr_hide_globals"] = torch.tensor([True])
            inputs = to_device(batch, model.device)
            with torch.no_grad(), amp(model.device):
                generated = model.generate(**inputs, **GENERATION, use_cache=True)
            response = processor.tokenizer.decode(generated[0, inputs["input_ids"].shape[-1]:],
                                                  skip_special_tokens=True)
            answer, how = extract(response, options(row["text"]))
            result = {"row_index": index, "question_id": row["question_id"], "category": row["category"],
                      "label": row["label"], "answer": answer, "how": how, "correct": answer == row["label"],
                      "response": response, "image_tokens": tokens}
            results.append(result)
            log.write(json.dumps(result) + "\n"); log.flush()
            summary = summarize(results) | {"mean_image_tokens": sum(r["image_tokens"] for r in results) / len(results)}
            json_write(output / "summary.json", summary)
            print(f"{index + 1}/{len(rows)} {summary['overall']['correct']} correct", flush=True)
    return summary
