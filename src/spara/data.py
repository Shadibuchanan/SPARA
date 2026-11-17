"""Immutable, hash-checked Mirage release and input-only record access."""
import hashlib
import json
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

MIRAGE_COMMIT = "53f26de2682025e146781c5b198ec93bdfe4c4d6"
BASE = f"https://raw.githubusercontent.com/UMass-Embodied-AGI/Mirage/{MIRAGE_COMMIT}"
ARCHIVES = {
    "train": ("vsp_spatial_planning.tar.gz", "df07b7f215d1a461dc1efc27c626caff11e22d5ea8ac694e2ba7be6a0487193c", 1000),
    "test": ("vsp_spatial_planning_test.tar.gz", "0272164f29dbed8b191977f7fa71cff8964b701bb84af293bf7525ada8394956", 400),
}
REFERENCE_SHA = "52cc1aab0d017fc0be9d7190f6f707330e2af7bb9e1c84f9055fb4d61b1c3771"


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download(url, dest, expected):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        temporary = dest.with_suffix(dest.suffix + ".partial")
        with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as out:
            while block := source.read(1024 * 1024):
                out.write(block)
        if sha256(temporary) != expected:
            raise ValueError(f"Checksum mismatch: {url}")
        temporary.replace(dest)
    if sha256(dest) != expected:
        raise ValueError(f"Checksum mismatch: {dest}")


def prepare(root):
    """Keep source archives; extract without allowing path/link escapes."""
    root = Path(root)
    for split, (name, digest, _) in ARCHIVES.items():
        archive = root / "archives" / name
        download(f"{BASE}/data/vsp_spatial_planning/{name}", archive, digest)
        target = root / split
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                path = (target / member.name).resolve()
                if not path.is_relative_to(target.resolve()) or not (member.isfile() or member.isdir()):
                    raise ValueError(f"Unsafe archive member: {member.name}")
            tar.extractall(target, filter="data")
    download(f"{BASE}/src/task.py", root / "reference" / "task.py", REFERENCE_SHA)
    report = audit(root)
    (root / "manifest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def records(root, split):
    """Rows of the released train or test split, or of a generated set "fresh/<name>"
    (spara.fresh_maps)."""
    if split.startswith(FRESH):
        return fresh_records(root, split)
    if split not in ARCHIVES:
        raise ValueError("Use the original train or test split, or a fresh/<name> set")
    path = Path(root) / split / f"{split}_direct.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != ARCHIVES[split][2]:
        raise ValueError(f"Unexpected {split} count: {len(rows)}")
    return rows


FRESH = "fresh/"


def released_layouts(root):
    return {layout_key(row) for split in ARCHIVES for row in records(root, split)}


def fresh_records(root, split):
    folder = Path(root) / split
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    path = folder / "maps.jsonl"
    if sha256(path) != manifest["jsonl_sha256"]:
        raise ValueError(f"{split} records do not match their manifest")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    keys = [layout_key(row) for row in rows]
    if len(rows) != manifest["count"] or len(set(keys)) != len(keys):
        raise ValueError(f"{split} has the wrong count or repeated layouts")
    if set(keys) & released_layouts(root):
        raise ValueError(f"{split} shares a layout with the released train or test split")
    return rows


def image_path(root, split, row):
    # The original paths are relative to Mirage's repo; extraction is split-local.
    # Generated sets store paths relative to their own folder, under img/.
    parts = Path(row["image_input"].replace("\\", "/")).parts
    marker = "imgs_test" if split == "test" else "img"
    if marker not in parts:
        raise ValueError(f"Unexpected released image path: {row['image_input']}")
    base = (Path(root) / split).resolve()
    path = base.joinpath(*parts[parts.index(marker):]).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"Missing or unsafe input image: {path}")
    return path


def image_paths(root, split, row):
    """Several images of one row (`image_inputs`, relative to the split's folder), each inside that folder."""
    base = (Path(root) / split).resolve()
    paths = [base.joinpath(*Path(p.replace("\\", "/")).parts).resolve() for p in row["image_inputs"]]
    for path in paths:
        if not path.is_relative_to(base) or not path.is_file():
            raise ValueError(f"Missing or unsafe input image: {path}")
    return paths


def layout_key(row):
    return json.dumps(row["map_desc"], separators=(",", ":"))


def audit(root):
    root = Path(root)
    report = {"mirage_commit": MIRAGE_COMMIT, "reference_sha256": REFERENCE_SHA, "splits": {}}
    layouts = {}
    for split, (name, digest, count) in ARCHIVES.items():
        if sha256(root / "archives" / name) != digest:
            raise ValueError("Source archive was modified")
        rows = records(root, split)
        # Compare the released JSONL bytes too, not just the row count.
        with tarfile.open(root / "archives" / name) as tar:
            member = next(m for m in tar if m.name.removeprefix("./") == f"{split}_direct.jsonl")
            if tar.extractfile(member).read() != (root / split / f"{split}_direct.jsonl").read_bytes():
                raise ValueError("Released records were modified")
        image_hashes = [sha256(image_path(root, split, row)) for row in rows]
        relative_images = {image_path(root, split, row).relative_to((root / split).resolve()).as_posix(): digest
                           for row, digest in zip(rows, image_hashes)}
        with tarfile.open(root / "archives" / name) as tar:
            checked = set()
            for member in tar:
                relative = member.name.removeprefix("./")
                if relative in relative_images:
                    expected = hashlib.sha256(tar.extractfile(member).read()).hexdigest()
                    if relative_images[relative] != expected:
                        raise ValueError(f"Released input image was modified: {relative}")
                    checked.add(relative)
            if checked != relative_images.keys():
                raise ValueError("Input image is not in the pinned source archive")
        layouts[split] = [layout_key(row) for row in rows]
        report["splits"][split] = {
            "count": count, "archive_sha256": digest,
            "jsonl_sha256": sha256(root / split / f"{split}_direct.jsonl"),
            "per_grid": dict(sorted(Counter(len(r["map_desc"]) for r in rows).items())),
            "input_image_sha256": image_hashes,
        }
    common = set(layouts["train"]) & set(layouts["test"])
    report["shared_layouts"] = len(common)
    report["test_rows_with_train_layout"] = sum(x in common for x in layouts["test"])
    return report
