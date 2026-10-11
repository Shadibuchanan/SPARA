"""CPU-only forensic checks of saved LIVR results; never changes benchmark scoring.

This diagnostic recomputes routes independently, tests input noninterference,
and measures overlap sensitivity. It is not an alternative production evaluator.
"""
import argparse
from collections import deque
import copy
import json
from pathlib import Path
import re

import torch
from transformers import AutoProcessor

from spara.data import audit, layout_key, records
from spara.evaluation import ReferenceEvaluator, summarize
from spara.serialization import encode


def walk(grid, actions):
    starts = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == 1]
    assert len(starts) == 1
    r, c = starts[0]
    off_grid = 0
    for action in actions:
        dr, dc = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}[action]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < len(grid) and 0 <= nc < len(grid[0])):
            off_grid += 1
            continue
        r, c = nr, nc
        if grid[r][c] == -1:
            return False, off_grid
    return grid[r][c] == 2, off_grid


def shortest(grid):
    start = next((r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == 1)
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (r, c), distance = queue.popleft()
        if grid[r][c] == 2:
            return distance
        for nr, nc in [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]:
            if (0 <= nr < len(grid) and 0 <= nc < len(grid[0])
                    and grid[nr][nc] != -1 and (nr, nc) not in seen):
                seen.add((nr, nc))
                queue.append(((nr, nc), distance + 1))
    raise AssertionError("Test map has no safe solution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--data-root", default="data/vsp")
    parser.add_argument("--processor", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = Path(args.run)
    # Match JSON's string keys for per-grid counts in persisted manifests.
    provenance = json.loads(json.dumps(audit(args.data_root)))
    train = records(args.data_root, "train")
    test = records(args.data_root, "test")
    outputs = [json.loads(line) for line in (run / "evaluation/responses.jsonl").read_text().splitlines()]
    assert len(outputs) == 400 and [x["row_index"] for x in outputs] == list(range(400))
    assert not any("text_output" in row for row in test)
    manifest = json.loads((run / "evaluation/manifest.json").read_text())
    assert manifest["data_manifest"] == provenance
    assert manifest["effective_generation"].get("num_return_sequences", 1) == 1
    for stage in (1, 2):
        stage_manifest = json.loads((run / f"stage{stage}-manifest.json").read_text())
        assert stage_manifest["data_manifest"] == provenance
    evaluator = ReferenceEvaluator(args.data_root)
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    train_layouts = {layout_key(row) for row in train}
    overlap, no_overlap = [], []
    strict_format, independent_success, successful_off_grid, successful_nonoptimal = [], [], [], []
    prefix_count = poison_count = 0
    for index, (row, result) in enumerate(zip(test, outputs)):
        rescored = evaluator.score(row, result["raw_response"])
        assert all(result[key] == value for key, value in rescored.items())
        answer = result["extracted_answer"]
        # Generated routes are full words or single-letter directions, comma/space separated.
        words = re.findall(r"[A-Za-z]+", answer)
        vocabulary = {"UP": "U", "DOWN": "D", "LEFT": "L", "RIGHT": "R", **{k: k for k in "UDLR"}}
        clean = bool(words) and all(w.upper() in vocabulary for w in words) and not re.sub(r"[A-Za-z,\s]", "", answer)
        if not clean:
            strict_format.append(index)
        actions = re.findall(r"[UDLR]", answer.upper())
        success, off_grid = walk(row["map_desc"], actions)
        assert success == result["success"], f"Independent route mismatch at row {index}"
        independent_success.append(success)
        if clean:
            assert actions == [vocabulary[word.upper()] for word in words]
        if success and off_grid:
            successful_off_grid.append(index)
        if success and len(actions) != shortest(row["map_desc"]):
            successful_nonoptimal.append(index)
        (overlap if layout_key(row) in train_layouts else no_overlap).append(result)

        batch = encode(processor, args.data_root, "test", row, training=False)
        assert "labels" not in batch
        prefix = processor.tokenizer.decode(batch["input_ids"][0], skip_special_tokens=False)
        assert result["raw_response"].startswith(prefix), f"Input prefix mismatch at row {index}"
        continuation = result["raw_response"][len(prefix):]
        assert continuation == result["answer"], f"Scorer included prompt text at row {index}"
        prefix_count += 1
        poisoned = copy.deepcopy(row)
        poisoned.update(map_desc=[["FORBIDDEN_METADATA"]], text_output="FORBIDDEN_ANSWER",
                        image_output="FORBIDDEN_HELPER_FILE.png", map_id="FORBIDDEN_ID")
        other = encode(processor, args.data_root, "test", poisoned, training=False)
        assert batch.keys() == other.keys()
        assert all(torch.equal(batch[key], other[key]) for key in batch), f"Metadata affected inputs at row {index}"
        poison_count += 1
    assert summarize(outputs) == json.loads((run / "evaluation/summary.json").read_text())
    report = {
        "rows": len(outputs), "reference_summary": summarize(outputs),
        "independent_grid_walk_successes": sum(independent_success),
        "nonstandard_route_format_rows": strict_format,
        "actual_input_prefixes_matched": prefix_count,
        "metadata_answer_helper_poison_checks_passed": poison_count,
        "data_and_images_match_original_archives_and_run_manifests": True,
        "test_records_contain_no_answer_field": True,
        "one_returned_sequence_per_example": True,
        "overlap": summarize(overlap), "overlap_excluded_diagnostic": summarize(no_overlap),
        "successful_routes_with_off_grid_moves": successful_off_grid,
        "successful_nonshortest_routes": successful_nonoptimal,
        "effective_generation": {k: manifest["effective_generation"].get(k) for k in
                                 ["do_sample", "temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens", "num_return_sequences"]},
        "interpretation": "No production metric or data changed. These checks cannot establish equivalence to unpublished LIVR code or identify the cause of the paper-score gap.",
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
