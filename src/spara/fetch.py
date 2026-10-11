"""Streaming model download for login nodes with small memory allowances."""
import json
import urllib.request
from pathlib import Path
from .data import sha256


def fetch_model(model_id, revision, destination, weights=True):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{model_id}/revision/{revision}?blobs=true") as response:
        info = json.load(response)
    if info["sha"] != revision:
        raise ValueError("Model revision must be an immutable commit hash")
    manifest = {}
    extensions = (".json", ".txt", ".jinja") + ((".safetensors",) if weights else ())
    for entry in info["siblings"]:
        name = entry["rfilename"]
        if not name.endswith(extensions):
            continue
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe model filename")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            print(f"Downloading {name}", flush=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            with urllib.request.urlopen(f"https://huggingface.co/{model_id}/resolve/{revision}/{name}", timeout=180) as response, temporary.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            temporary.replace(path)
        digest = sha256(path)
        expected = entry.get("lfs", {}).get("sha256")
        if expected and expected != digest:
            raise ValueError(f"Model hash mismatch: {name}")
        if entry.get("size") is not None and path.stat().st_size != entry["size"]:
            raise ValueError(f"Model size mismatch: {name}")
        manifest[name] = {"sha256": digest, "bytes": path.stat().st_size}
    (root / "download-manifest.json").write_text(json.dumps({"revision": revision, "files": manifest}, indent=2), encoding="utf-8")
    return root
