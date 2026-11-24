"""Visual-CoT (Shao et al., 2024) as answer-only training data: each released question with its final
answer, without the bounding-box turn or the cropped second image. A seeded sample of the released list
is kept, and only its images are taken from the 139 GB archive, read as one stream and never stored."""
import hashlib
import io
import json
import random
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

from .data import download

DATASET = "deepcs233/Visual-CoT"
REVISION = "223d2d8c1146fda2bb918801b8276c587b78b61c"
BASE = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}"
LIST = "viscot_363k.json"
LIST_SHA = "39c2a901d357fcb1eb377f6046a9124ddb9ac97f1620ca73b87af0340a7f7f8b"
LIST_ROWS = 404120
PARTS = [f"cot_images_tar_split/cot_images_{i:02d}" for i in range(13)]
BOX_REQUEST = " Please provide the bounding box coordinate of the region that can help you answer the question better."
SPLIT = "images"  # the folder under the data root that holds every kept image
ARCHIVE_PREFIX, LIST_PREFIX = "cot_image_data/", "cot/"


def answer_only(source):
    """One released row as a training row: the image, a newline and the question; the answer is the last turn."""
    turns = source["conversations"]
    first = turns[0]["value"]
    image, crop = source["image"]
    if (len(turns) != 4 or not first.startswith("<image>\n") or not first.endswith(BOX_REQUEST)
            or turns[2]["value"] != "<image>" or not crop.startswith(image + "###") or not image.startswith(LIST_PREFIX)):
        raise ValueError(f"Unexpected Visual-CoT row: {source['dataset']} {source['question_id']}")
    return {"text_input": first[:-len(BOX_REQUEST)], "text_output": turns[3]["value"], "image_inputs": [image],
            "dataset": source["dataset"], "question_id": source["question_id"]}


def select(root, train, validation, seed):
    """Download the released list and keep a seeded sample: `validation` rows, then `train` rows on images
    the validation rows do not use. Writes train.jsonl, validation.jsonl and manifest.json."""
    root = Path(root)
    download(f"{BASE}/{LIST}", root / LIST, LIST_SHA)
    released = json.loads((root / LIST).read_text(encoding="utf-8"))
    if len(released) != LIST_ROWS:
        raise ValueError("The Visual-CoT list is not the pinned release")
    order = list(range(len(released)))
    random.Random(seed).shuffle(order)
    held = [answer_only(released[i]) for i in order[:validation]]
    held_images = {row["image_inputs"][0] for row in held}
    kept = []
    for i in order[validation:]:
        if len(kept) == train:
            break
        row = answer_only(released[i])
        if row["image_inputs"][0] not in held_images:
            kept.append(row)
    if len(kept) != train:
        raise ValueError("Not enough rows for the requested sample")
    for name, rows in (("train", kept), ("validation", held)):
        (root / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = {"dataset": DATASET, "revision": REVISION, "list_sha256": LIST_SHA, "seed": seed,
                "train": {"rows": len(kept), "by_dataset": dict(Counter(row["dataset"] for row in kept))},
                "validation": {"rows": len(held), "by_dataset": dict(Counter(row["dataset"] for row in held))},
                "images": None}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def rows(root, name):
    """(split, row) pairs for `train_stage`: every image lives under the root's one image folder."""
    lines = (Path(root) / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [(SPLIT, json.loads(line)) for line in lines if line.strip()]


class Parts(io.RawIOBase):
    """The archive's parts as one byte stream. Each part is hashed as it passes and checked against the
    hub's recorded hash when it ends. A connection that fails, or closes before the part's recorded size
    (which the HTTP client reports as an ordinary end of data), resumes at the same byte."""

    def __init__(self, parts, opener=urllib.request.urlopen, retries=5):
        self.parts, self.opener, self.retries = list(parts), opener, retries  # [(url, sha256, bytes)]
        self.index, self.offset, self.response, self.digest = 0, 0, None, hashlib.sha256()
        self.verified = []

    def readable(self):
        return True

    def _open(self):
        request = urllib.request.Request(self.parts[self.index][0],
                                         headers={"Range": f"bytes={self.offset}-"} if self.offset else {})
        self.response = self.opener(request, timeout=300)

    def readinto(self, buffer):
        while self.index < len(self.parts):
            url, expected, size = self.parts[self.index]
            block, failures = b"", 0
            while not block and self.offset < size:
                try:
                    if self.response is None:
                        self._open()
                    block = self.response.read(len(buffer))
                    if not block:
                        raise ConnectionError(f"{url} closed at byte {self.offset} of {size}")
                except OSError:
                    failures += 1
                    if failures > self.retries:
                        raise
                    self.response = None
                    time.sleep(min(60, 5 * failures))
            if block:
                self.digest.update(block)
                self.offset += len(block)
                buffer[:len(block)] = block
                return len(block)
            if self.digest.hexdigest() != expected:
                raise ValueError(f"Checksum mismatch: {url}")
            self.verified.append(url.rsplit("/", 1)[-1])
            self.index, self.offset, self.response, self.digest = self.index + 1, 0, None, hashlib.sha256()
        return 0


def part_hashes():
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{DATASET}/revision/{REVISION}?blobs=true",
                                timeout=120) as response:
        listed = {entry["rfilename"]: entry for entry in json.load(response)["siblings"]}
    return [(f"{BASE}/{name}", listed[name]["lfs"]["sha256"], listed[name]["size"]) for name in PARTS]


def take_images(root, max_image_tokens, stream=None):
    """Read the archive once and keep the sampled rows' images, each resized to at most `max_image_tokens`
    image tokens and saved as JPEG. Stops when every image is found; images already present are kept."""
    from PIL import Image
    from .vstar import PATCH, resized
    root = Path(root)
    base = (root / SPLIT).resolve()
    wanted = {row["image_inputs"][0] for name in ("train", "validation") for _, row in rows(root, name)}
    missing = {name for name in wanted if not (base / name).is_file()}
    stream = stream if stream is not None else Parts(part_hashes())
    seen = 0
    if missing:
        with tarfile.open(fileobj=io.BufferedReader(stream, 1 << 22), mode="r|") as archive:
            for member in archive:
                seen += 1
                name = LIST_PREFIX + member.name[len(ARCHIVE_PREFIX):]
                if not member.isfile() or not member.name.startswith(ARCHIVE_PREFIX) or name not in missing:
                    continue
                path = base.joinpath(*name.split("/")).resolve()
                if not path.is_relative_to(base):
                    raise ValueError(f"Unsafe archive member: {member.name}")
                path.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(io.BytesIO(archive.extractfile(member).read())) as source:
                    picture = resized(source.convert("RGB"), max_image_tokens * PATCH * PATCH)
                temporary = path.with_name(path.name + ".partial")
                picture.save(temporary, format="JPEG", quality=95)
                temporary.replace(path)
                missing.discard(name)
                if len(missing) % 500 == 0:
                    print(f"{len(wanted) - len(missing)}/{len(wanted)} images after {seen} archive members", flush=True)
                if not missing:
                    break
    if missing:
        raise ValueError(f"{len(missing)} sampled images are not in the archive, e.g. {sorted(missing)[:3]}")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["images"] = {"count": len(wanted), "max_image_tokens": max_image_tokens, "format": "JPEG quality 95",
                          "archive_parts_verified": getattr(stream, "verified", [])}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
