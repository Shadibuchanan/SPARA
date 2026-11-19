"""Prepare the perception data (docs/PERCEPTION_BENCHMARKS_SPEC.md): an 8k Zebra-CoT training set and the
MMVP, V* and BLINK (val) evaluation sets, as rows that `spara.serialization.encode` reads.

Each split is a folder under ROOT with rows.jsonl and images/. A row has `text_input` (one `<image>` per
image), `image_inputs` (paths relative to the split's folder), `text_output` (the answer: the final answer
for training rows, the gold option letter for evaluation rows) and metadata. Training rows also carry
`helper_input`, Zebra-CoT's first reasoning image, relative to ROOT, for the Scaffolding Minds baseline.

Every image is resized, keeping its aspect ratio, to at most MAX_PIXELS (1,024 merged image tokens in
Qwen2.5-VL), so no run depends on the processor's pixel limits.

Needs pyarrow, pillow and huggingface_hub (a separate environment; the run environment is unchanged).
Usage: python scripts/prepare_perception.py ROOT [--train 8000] [--seed 42]
"""
import argparse
import ast
import csv
import hashlib
import io
import json
import random
import re
from pathlib import Path

MAX_PIXELS = 1024 * 28 * 28
ZEBRA = "multimodal-reasoning-lab/Zebra-CoT"
CATEGORIES = {"2D Visual Reasoning - Visual Search": 27, "2D Visual Reasoning - Visual Jigsaw": 26,
              "3D Visual Reasoning - Multi-Hop Objects Counting": 40}  # shard counts
LETTER = "Answer with the option's letter from the given choices directly."
SHORT = "Answer the question using a single word or phrase."
MAX_ANSWER_WORDS = 4


def resized(data_or_image):
    from PIL import Image
    image = data_or_image if hasattr(data_or_image, "size") else Image.open(io.BytesIO(data_or_image))
    image = image.convert("RGB")
    w, h = image.size
    if w * h > MAX_PIXELS:
        scale = (MAX_PIXELS / (w * h)) ** 0.5
        image = image.resize((max(28, int(w * scale)), max(28, int(h * scale))), Image.LANCZOS)
    return image


def save(image, folder, name, fmt):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.{'png' if fmt == 'PNG' else 'jpg'}"
    image.save(path, fmt, **({"quality": 95} if fmt == "JPEG" else {}))
    return path


def write_rows(folder, rows):
    folder.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    (folder / "rows.jsonl").write_text(text, encoding="utf-8")
    manifest = {"rows": len(rows), "rows_sha256": hashlib.sha256(text.encode()).hexdigest(), "max_pixels": MAX_PIXELS}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(folder.name, len(rows), flush=True)


def download(repo, filename, root):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type="dataset", local_dir=str(root / "raw" / repo.split("/")[-1]))


# Zebra-CoT training rows ---------------------------------------------------------------------

def zebra_row(record):
    """(question with <image> placeholders, problem image keys in order, answer, suffix) or None."""
    answer = (record.get("Final Answer") or "").strip()
    question = record.get("Question") or ""
    keys = re.findall(r"\[(problem_image_\d+)\]", question)
    if not answer or not keys or len(answer.split()) > MAX_ANSWER_WORDS or "<image>" in question:
        return None
    if any(record.get(k) is None for k in keys) or record.get("reasoning_image_1") is None:
        return None
    text = re.sub(r"<image_start>\[problem_image_\d+\]<image_end>", "<image>", question)
    if text.count("<image>") != len(keys):
        return None
    suffix = LETTER if re.fullmatch(r"[A-Ea-e]", answer) else SHORT
    return text.strip() + "\n" + suffix, keys, answer


def zebra(root, total, seed):
    import pyarrow.parquet as pq
    folder = root / "zebra8k"
    per = [total // len(CATEGORIES) + (i < total % len(CATEGORIES)) for i in range(len(CATEGORIES))]
    rows, skipped = [], {}
    for (category, shards), want in zip(CATEGORIES.items(), per):
        pool = []
        for shard in range(shards):
            table = pq.read_table(download(ZEBRA, f"{category}/train-{shard:05d}-of-{shards:05d}.parquet", root))
            for i, record in enumerate(table.to_pylist()):
                parsed = zebra_row(record)
                if parsed is None:
                    skipped[category] = skipped.get(category, 0) + 1
                else:
                    pool.append((shard, i, record, parsed))
            if len(pool) >= 2 * want or shard == shards - 1:
                break
        if len(pool) < want:
            raise ValueError(f"Only {len(pool)} usable rows in {category}")
        for shard, i, record, (text, keys, answer) in random.Random(seed).sample(pool, want):
            name = f"{category.split(' - ')[-1].replace(' ', '_').lower()}-{shard}-{i}"
            images = [save(resized(record[k]["bytes"]), folder / "images", f"{name}-{k}", "JPEG") for k in keys]
            helper = save(resized(record["reasoning_image_1"]["bytes"]), folder / "helpers", name, "JPEG")
            rows.append({"text_input": text, "image_inputs": [str(p.relative_to(folder)) for p in images],
                         "text_output": answer, "helper_input": str(helper.relative_to(root)),
                         "benchmark": "zebra-cot", "category": category, "id": name})
    random.Random(seed).shuffle(rows)
    write_rows(folder, rows)
    (folder / "skipped.json").write_text(json.dumps(skipped, indent=1) + "\n", encoding="utf-8")


# Evaluation sets -----------------------------------------------------------------------------

def options_text(options):
    return "\n".join(f"({chr(65 + i)}) {o}" for i, o in enumerate(options))


def mmvp(root):
    from PIL import Image
    folder, rows = root / "mmvp", []
    with open(download("MMVP/MMVP", "Questions.csv", root), encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            index = int(record["Index"])
            options = [o.strip() for o in re.split(r"\([a-z]\)", record["Options"]) if o.strip()]
            gold = "abcdefg".index(record["Correct Answer"].strip().strip("()").lower())
            image = resized(Image.open(download("MMVP/MMVP", f"MMVP Images/{index}.jpg", root)))
            path = save(image, folder / "images", str(index), "PNG")
            rows.append({"text_input": f"<image>\n{record['Question'].strip()}\n{options_text(options)}\n{LETTER}",
                         "image_inputs": [str(path.relative_to(folder))], "text_output": chr(65 + gold),
                         "options": len(options), "benchmark": "mmvp", "category": "mmvp", "id": str(index),
                         "pair": (index - 1) // 2})
    write_rows(folder, rows)


def vstar(root):
    """V*'s released questions and option order (its labels are spread across A to D)."""
    from PIL import Image
    folder, rows = root / "vstar", []
    for line in Path(download("craigwu/vstar_bench", "test_questions.jsonl", root)).read_text().splitlines():
        record = json.loads(line)
        lines = record["text"].split("\n")
        question = lines[0]
        options = [re.sub(r"^\([A-Z]\)\s*", "", l) for l in lines[1:] if re.match(r"^\([A-Z]\)", l)]
        gold = ord(record["label"]) - 65
        image = resized(Image.open(download("craigwu/vstar_bench", record["image"], root)))
        path = save(image, folder / "images", record["question_id"], "PNG")
        rows.append({"text_input": f"<image>\n{question}\n{options_text(options)}\n{LETTER}",
                     "image_inputs": [str(path.relative_to(folder))], "text_output": chr(65 + gold),
                     "options": len(options), "benchmark": "vstar", "category": record["category"],
                     "id": record["question_id"], "released_label": record["label"]})
    write_rows(folder, rows)


BLINK_TASKS = ["Art_Style", "Counting", "Forensic_Detection", "Functional_Correspondence", "IQ_Test", "Jigsaw",
               "Multi-view_Reasoning", "Object_Localization", "Relative_Depth", "Relative_Reflectance",
               "Semantic_Correspondence", "Spatial_Relation", "Visual_Correspondence", "Visual_Similarity"]


def blink(root):
    """BLINK's val split with its released prompts (which list the choices); the images precede the prompt."""
    import pyarrow.parquet as pq
    folder, rows = root / "blink", []
    for task in BLINK_TASKS:
        for record in pq.read_table(download("BLINK-Benchmark/BLINK", f"{task}/val-00000-of-00001.parquet",
                                             root)).to_pylist():
            keys = [k for k in ("image_1", "image_2", "image_3", "image_4") if record.get(k)]
            images = [save(resized(record[k]["bytes"]), folder / "images", f"{record['idx']}-{k}", "PNG")
                      for k in keys]
            choices = record["choices"] if isinstance(record["choices"], list) else ast.literal_eval(record["choices"])
            prompt = record["prompt"].strip()
            rows.append({"text_input": "<image>\n" * len(images) + prompt + "\n" + LETTER,
                         "image_inputs": [str(p.relative_to(folder)) for p in images],
                         "text_output": record["answer"].strip().strip("()"), "options": len(choices),
                         "benchmark": "blink", "category": task, "id": record["idx"]})
    write_rows(folder, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--train", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", default="mmvp,vstar,blink,zebra")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(","))
    for name, run in (("mmvp", lambda: mmvp(root)), ("vstar", lambda: vstar(root)),
                      ("blink", lambda: blink(root)), ("zebra", lambda: zebra(root, args.train, args.seed))):
        if name in only:
            run()


if __name__ == "__main__":
    main()
