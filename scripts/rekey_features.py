"""Re-key a features file saved with map-ID keys (job 7116225) to `perception_jepa.map_key`
(split, image). CPU only; the tensors are unchanged.

Map IDs repeat across levels within the released training split, so the features job kept one map
per repeated ID. Each saved map is found again by its split, ID and layout; the maps it dropped are
reported, and the arm computes their targets at its start (`perception_jepa.add_targets`).

Usage: python scripts/rekey_features.py FEATURES_RUN OUTPUT --data-root data/vsp
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from spara.data import records
from spara.perception_jepa import FEATURES_FILE, map_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("features_run")
    parser.add_argument("output")
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    source = Path(args.features_run) / FEATURES_FILE
    saved = torch.load(source, weights_only=False)
    tables = {}
    renamed = {}
    for (split, map_id), desc in zip([tuple(k) for k in saved["keys"]], saved["map_desc"]):
        table = tables.setdefault(split, records(args.data_root, split))
        found = [r for r in table if r["map_id"] == map_id and r["map_desc"] == desc]
        if len(found) != 1:
            raise ValueError(f"{split} map {map_id}: {len(found)} rows with that ID and layout")
        renamed[(split, map_id)] = map_key(split, found[0])
    if len(set(renamed.values())) != len(renamed):
        raise ValueError("Two saved maps share an image")
    out = saved | {"keys": [renamed[tuple(k)] for k in saved["keys"]],
                   "groups": {name: [renamed[tuple(k)] for k in keys] for name, keys in saved["groups"].items()}}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.save(out, output / FEATURES_FILE)
    repeated = {split: sum(1 for r in table if sum(o["map_id"] == r["map_id"] for o in table) > 1)
                for split, table in tables.items()}
    report = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "maps": len(renamed), "rows_sharing_an_id": repeated}
    (output / "rekey.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
