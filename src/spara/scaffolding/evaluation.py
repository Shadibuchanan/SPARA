"""Reference-protocol evaluation with the LIVR baseline's decoding settings."""
import json
from pathlib import Path

import torch

from ..data import records
from ..evaluation import ReferenceEvaluator, summarize
from ..runtime import json_write, seed_all
from ..training import amp
from . import serialization
from .decoding import LatentDecoder
from .training import to_device


def evaluate(model, processor, data_root, output, config, run_manifest):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    evaluator = ReferenceEvaluator(data_root)
    # Stage 1 checkpoints have no heads: their latent is the predicted prior (delta = 0).
    mode = config["evaluation_latent"] if hasattr(model, "latent_heads") else "none"
    model.eval()
    decoder = LatentDecoder(model, config["generation"], mode)
    settings = {key: value for key, value in decoder.settings.to_dict().items() if not key.startswith("_")}
    run_manifest.update(evaluator=evaluator.provenance, effective_generation=settings, latent_mode=mode)
    json_write(output / "manifest.json", run_manifest)
    # Reset after loading so sampling begins from the recorded seed.
    seed_all(config["seed"])
    outputs = []
    rows = records(data_root, "test")
    with (output / "responses.jsonl").open("w", encoding="utf-8") as log:
        for index, row in enumerate(rows):
            prompt = to_device(serialization.prompt(processor, data_root, "test", row), model.device)
            with torch.no_grad(), amp(model.device):
                rollout = decoder.generate(prompt)
            text = processor.tokenizer.decode(prompt["input_ids"][0].tolist() + rollout.ids, skip_special_tokens=False)
            result = evaluator.score(row, text)
            result.update(row_index=index, latent_block=rollout.block is not None, block_index=rollout.block,
                          generated_tokens=len(rollout.sampled))
            outputs.append(result)
            log.write(json.dumps(result) + "\n"); log.flush()
            json_write(output / "summary.json", summarize(outputs) | {"latent_mode": mode})
            print(f"{index + 1}/{len(rows)} {summarize(outputs)['overall']}", flush=True)
    return summarize(outputs)
