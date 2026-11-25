"""Can a model reach a correct route by writing its own reasoning first?
(docs/DETOUR_SCALING_SPEC.md, "Written reasoning without labels: the probe")

Label-free reasoning (STaR, expert iteration on reasoning) can start only if the model's own
sampled reasoning sometimes ends in a route the verifier accepts. For each map this samples
several answers, with or without an instruction to reason first, and scores each answer with
the reference evaluator. No label, layout fact or helper image enters the prompt."""
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from .data import records
from .evaluation import ReferenceEvaluator
from .fresh_maps import GOAL, START, find, shortest_path
from .runtime import json_write, seed_all
from .serialization import encode
from .text_rl import repeat_prompt
from .training import amp, to_device

REASON = ("\nBefore answering, think step by step: find the start, the goal and every hole on the map, "
          "then trace a safe path one move at a time. Then put the final answer in \\boxed{}.")
KEPT_TEXTS = 20  # maps whose sampled answers are stored in full, for reading


def away(desc):
    """Moves away from the goal the shortest safe route needs (0 for a direct map)."""
    (sr, sc), (gr, gc) = find(desc, START), find(desc, GOAL)
    return (shortest_path(desc) - abs(sr - gr) - abs(sc - gc)) // 2


def summarise(rows):
    groups = defaultdict(list)
    for r in rows:
        for key in ("all", f"size {r['grid_size']}", f"away {min(r['away'], 3)}"):
            groups[key].append(r)
    return {key: {"maps": len(g),
                  "solved_by_any": sum(any(r["successes"]) for r in g),
                  "mean_success": sum(sum(r["successes"]) / len(r["successes"]) for r in g) / len(g),
                  "mean_tokens": sum(sum(r["tokens"]) / len(r["tokens"]) for r in g) / len(g),
                  "capped": sum(sum(r["capped"]) for r in g)}
            for key, g in sorted(groups.items())}


@torch.no_grad()
def run_probe(model, processor, data_root, output, split, indices, samples, generation, reason, run_manifest):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    model.eval()
    seed_all(run_manifest.get("seed", 42))
    evaluator = ReferenceEvaluator(data_root)
    rows = records(data_root, split)
    latents = model.config.livr_latent_tokens
    eos = {processor.tokenizer.convert_tokens_to_ids("<|im_end|>"), processor.tokenizer.eos_token_id}
    json_write(output / "manifest.json", run_manifest | {
        "split": split, "indices": [indices[0], indices[-1] + 1] if indices == list(range(indices[0], indices[-1] + 1))
        else indices, "samples": samples, "generation": generation, "reason": reason,
        "instruction": REASON if reason else None, "latent_tokens": latents, "evaluator": evaluator.provenance})
    results = []
    with (output / "maps.jsonl").open("w", encoding="utf-8") as log:
        for count, index in enumerate(indices):
            row = rows[index]
            asked = row | {"text_input": row["text_input"] + (REASON if reason else "")}
            prompt = to_device(encode(processor, data_root, split, asked, training=False, latents=latents,
                                      generation_prompt=not latents), model.device)
            with amp(model.device):
                generated = model.generate(**repeat_prompt(prompt, samples), **generation, use_cache=True)
            completions = generated[:, prompt["input_ids"].shape[-1]:]
            texts, tokens, capped = [], [], []
            for c in completions:
                ends = [i for i, t in enumerate(c.tolist()) if t in eos]
                tokens.append(ends[0] + 1 if ends else len(c))
                capped.append(not ends)
                texts.append(processor.tokenizer.decode(c[:tokens[-1]], skip_special_tokens=True))
            record = {"row_index": index, "grid_size": len(row["map_desc"]), "away": away(row["map_desc"]),
                      "successes": [bool(evaluator.score(row, t)["success"]) for t in texts],
                      "tokens": tokens, "capped": capped}
            if count < KEPT_TEXTS:
                record["texts"] = texts
            results.append(record)
            log.write(json.dumps(record) + "\n")
            log.flush()
            if count % 10 == 0:
                print(f"probe {count + 1}/{len(indices)}", json.dumps(summarise(results)["all"]), flush=True)
    summary = {"maps": len(results), "by": summarise(results), "seconds": time.time() - began,
               "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None}
    json_write(output / "summary.json", summary)
    print(json.dumps(summary, indent=1), flush=True)
    return summary
