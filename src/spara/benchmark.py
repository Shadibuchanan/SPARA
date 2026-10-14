"""Reference benchmark evaluation and the multi-sample variance diagnostic."""
import json
from pathlib import Path

import torch

from .data import records
from .evaluation import ReferenceEvaluator, summarize
from .runtime import json_write, seed_all
from .serialization import encode
from .text_rl import completion_mask, repeat_prompt
from .training import amp, to_device


def run_benchmark(model, processor, config, data_root, output, run_manifest):
    """One sampled answer per test record in original order: the reported protocol."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    evaluator = ReferenceEvaluator(data_root)
    model.eval()
    generation = model.generation_config.to_dict() | config["generation"]
    run_manifest = run_manifest | {"evaluator": evaluator.provenance, "effective_generation": generation}
    json_write(output / "manifest.json", run_manifest)
    # Reset after loading/initialization so sampling begins from the recorded seed.
    seed_all(config["seed"])
    outputs = []
    with (output / "responses.jsonl").open("w", encoding="utf-8") as log:
        for index, row in enumerate(records(data_root, "test")):
            inputs = to_device(encode(processor, data_root, "test", row, training=False), model.device)
            with torch.no_grad(), amp(model.device):
                generated = model.generate(**inputs, **config["generation"], use_cache=True)
            result = evaluator.score(row, processor.tokenizer.decode(generated[0], skip_special_tokens=False))
            result["row_index"] = index
            outputs.append(result)
            log.write(json.dumps(result) + "\n"); log.flush()
            json_write(output / "summary.json", summarize(outputs))
            print(f"{index+1}/400 {summarize(outputs)['overall']}", flush=True)
    return summarize(outputs)


def sample_test_maps(model, processor, config, data_root, samples, path):
    """Diagnostic, not the benchmark: K samples per test map to estimate per-map success."""
    evaluator = ReferenceEvaluator(data_root)
    model.eval()
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    seed_all(config["seed"])
    with Path(path).open("w", encoding="utf-8") as log:
        for index, row in enumerate(records(data_root, "test")):
            prompt = to_device(encode(processor, data_root, "test", row, training=False), model.device)
            with torch.no_grad(), amp(model.device):
                generated = model.generate(**repeat_prompt(prompt, samples), **config["generation"], use_cache=True)
            completions = generated[:, prompt["input_ids"].shape[-1]:]
            lengths = completion_mask(completions, eos).sum(-1)
            results = [evaluator.score(row, processor.tokenizer.decode(
                torch.cat([prompt["input_ids"][0], completions[i, :lengths[i]]]), skip_special_tokens=False))
                for i in range(samples)]
            log.write(json.dumps({"row_index": index, "map_id": row["map_id"], "grid_size": len(row["map_desc"]),
                                  "successes": [bool(r["success"]) for r in results],
                                  "answers": [r["extracted_answer"] for r in results]}) + "\n")
            if index % 50 == 0:
                print("sampled", index, flush=True)
