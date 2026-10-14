"""Vanilla SFT pilot: prefix check on the real processor, two optimizer steps,
one checkpoint save and two evaluation records on the real 3B model."""
import argparse
import json
import resource
import time
from pathlib import Path

import torch
from transformers import AutoProcessor

from spara import training
from spara.data import records
from spara.evaluation import ReferenceEvaluator
from spara.model import LATENT_TOKENS
from spara.runtime import json_write, load_model, read_config, seed_all
from spara.serialization import encode


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sft.json")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_config(args.config) | {"local_model_path": str(Path(args.base_model).resolve())}
    seed_all(config["seed"])
    report = {"hardware": torch.cuda.get_device_name(), "config": args.config}
    started = time.perf_counter()
    model, processor = load_model(config, device="cuda")
    report["load"] = {"seconds": time.perf_counter() - started, "peak_rss_gib": rss_gib()}
    assert model.config.livr_latent_ids == [] and model.config.livr_stage == 2

    # Prefix check: the SFT prefix is the LIVR prefix minus exactly the latent IDs.
    livr = AutoProcessor.from_pretrained(args.base_model)
    livr.tokenizer.add_tokens(LATENT_TOKENS, special_tokens=True)
    ids = livr.tokenizer.convert_tokens_to_ids(LATENT_TOKENS)
    assert ids == [151665, 151666, 151667, 151668]
    assert model.get_input_embeddings().num_embeddings > max(ids)
    for split, row in (("train", records(args.data_root, "train")[0]), ("test", records(args.data_root, "test")[0])):
        training_row = split == "train"  # test rows carry no answer
        a = encode(livr, args.data_root, split, row, training=training_row)
        b = encode(processor, args.data_root, split, row, training=training_row, latents=[])
        kept = ~torch.isin(a["input_ids"][0], torch.tensor(ids))
        assert a["input_ids"].shape[-1] - b["input_ids"].shape[-1] == 4
        assert torch.equal(a["input_ids"][0][kept], b["input_ids"][0]) and torch.equal(a["pixel_values"], b["pixel_values"])
        image_tokens = b["input_ids"].eq(model.config.image_token_id).sum().item()
        assert image_tokens == a["input_ids"].eq(model.config.image_token_id).sum().item()
        assert b["image_grid_thw"].tolist() == a["image_grid_thw"].tolist()
        # Image size follows grid size; the 5x5 train row is 121 tokens.
        assert not training_row or (image_tokens == 121 and b["image_grid_thw"].tolist() == [[1, 22, 22]])
        assert processor.tokenizer.decode(b["input_ids"][0]).endswith("<|im_end|>\n<|im_start|>assistant" if not training_row else "<|im_end|>")
        if training_row:
            assert torch.equal(a["labels"][0][kept], b["labels"][0])
            assert processor.tokenizer.decode(b["labels"][0][b["labels"][0] != -100]) == row["text_output"] + "<|im_end|>"
            train_batch = b
    # With no latent IDs a segment labelled Stage 1 uses the standard mask.
    batch = training.to_device(train_batch, model.device)
    with torch.no_grad(), training.amp(model.device):
        model.config.livr_stage = 1
        stage1 = model(**batch, use_cache=False).logits
        model.config.livr_stage = 2
        stage2 = model(**batch, use_cache=False).logits
    assert torch.equal(stage1, stage2)
    report["prefix_check"] = {"passed": True, "livr_minus_sft_tokens": 4, "image_tokens": 121}

    # Two optimizer steps through the unchanged training loop: 16 rows, one epoch.
    torch.cuda.reset_peak_memory_stats()
    rows = records(args.data_root, "train")[:16]
    training.records = lambda *_: rows
    pilot = config | {"epochs_per_stage": 1, "order_cycle": 1}
    started = time.perf_counter()
    final = training.train_stage(model, processor, args.data_root, args.output / "train", pilot, 2)
    steps = [json.loads(line) for line in (args.output / "train/steps.jsonl").read_text().splitlines()]
    assert len(steps) == 2 and all(torch.isfinite(torch.tensor([s["loss"], s["grad_norm"]])).all() for s in steps)
    report["train"] = {"seconds": time.perf_counter() - started, "steps": steps,
                       "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                       "checkpoint": json.loads((final / "complete.json").read_text()) | {"progress": None},
                       "peak_rss_gib": rss_gib()}

    # Two evaluation records exactly as the evaluate command generates them.
    evaluator = ReferenceEvaluator(args.data_root)
    model.eval()
    model.config.use_cache = True
    seed_all(config["seed"])
    outputs = []
    for index, row in enumerate(records(args.data_root, "test")[:2]):
        inputs = training.to_device(encode(processor, args.data_root, "test", row, training=False,
                                           latents=model.config.livr_latent_tokens), model.device)
        with torch.no_grad(), training.amp(model.device):
            generated = model.generate(**inputs, **config["generation"], use_cache=True)
        result = evaluator.score(row, processor.tokenizer.decode(generated[0], skip_special_tokens=False))
        outputs.append(result | {"row_index": index, "raw_response": result["raw_response"][-300:]})
    report["evaluation"] = outputs
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["peak_rss_gib"] = rss_gib()
    json_write(args.output / "pilot-report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "evaluation"}, indent=2, default=str))


if __name__ == "__main__":
    main()
