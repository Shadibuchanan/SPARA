import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from spara.scaffolding import training
from spara.scaffolding.model import ScaffoldingEncoder, ScaffoldingQwen, add_heads, train_language
from spara.scaffolding.runtime import read_config, require_final

ROWS = list(range(4))


class Processor:
    def save_pretrained(self, path): pass

    class tokenizer:
        @staticmethod
        def decode(ids, skip_special_tokens=False):
            return " ".join(map(str, ids))


class Evaluator:
    """Rewards even-length rollouts so groups get nonzero advantages."""
    def score(self, row, text):
        return {"success": len(text.split()) % 2 == 0}


def tiny_recipe(**overrides):
    config = read_config("configs/scaffolding.json") | {"gradient_checkpointing": False, "warmup_ratio": 0.0}
    config["scaffold"] = config["scaffold"] | {"epochs": 1, "gradient_accumulation": 2, "pooling_heads": 4}
    config["predict"] = config["predict"] | {"epochs": 2, "gradient_accumulation": 3}
    config["rl"] = config["rl"] | {"steps": 2, "prompts_per_step": 2, "group_size": 3, "head_hidden_size": 16,
                                   "learning_rate": 1e-3, "rollout": config["rl"]["rollout"] | {"max_new_tokens": 4}}
    return config | overrides


@pytest.fixture
def fake_data(monkeypatch, tiny_sample):
    def sample(processor, root, row, config, with_helper=False):
        batch = {"prompt": {k: v.clone() for k, v in tiny_sample["prompt"].items()},
                 "answer": torch.tensor([[14 + row % 3, 15, 6]])}
        batch["prompt"]["pixel_values"] += row * 0.01
        if with_helper:
            batch["helper"] = {k: v.clone() for k, v in tiny_sample["helper"].items()}
            batch["helper"]["pixel_values"] -= row * 0.01
        return batch
    monkeypatch.setattr(training, "records", lambda *_: ROWS)
    monkeypatch.setattr(training.serialization, "training_sample", sample)
    monkeypatch.setattr(training.serialization, "prompt", lambda p, r, s, row: sample(p, r, row, None)["prompt"])
    monkeypatch.setattr(training.serialization, "helper", lambda p, r, row, c: sample(p, r, row, c, True)["helper"])


def test_three_phases_run_end_to_end(tiny_scaffold, fake_data, tmp_path):
    config = tiny_recipe()
    vlm = tiny_scaffold
    encoder = ScaffoldingEncoder.from_vlm(vlm, 4, 4)
    before = encoder.pooling.queries.detach().clone()
    final = training.train_scaffold(vlm, encoder, Processor(), "unused", tmp_path / "scaffold", config)
    progress = json.loads((final / "complete.json").read_text())["progress"]
    assert progress["phase"] == "scaffold" and progress["completed"] and progress["optimizer_steps"] == 2
    assert not torch.equal(before, encoder.pooling.queries)
    targets = training.scaffold_targets(ScaffoldingEncoder.from_pretrained(final), Processor(), "unused", ROWS,
                                        config, torch.device("cpu"))
    assert targets.shape == (4, 4, 32)

    final = training.train_predict(vlm, Processor(), targets, "unused", tmp_path / "predict", config,
                                   encoder_path=final)
    steps = [json.loads(line) for line in (tmp_path / "predict/steps.jsonl").read_text().splitlines()]
    assert len(steps) == 4 and {"latent_loss", "latent_loss_normalized", "ce_loss"} <= steps[0].keys()

    policy = ScaffoldingQwen.from_pretrained(final, attn_implementation="eager")
    reference = ScaffoldingQwen.from_pretrained(final, attn_implementation="eager")
    add_heads(policy, config["rl"])
    heads = {k: v.clone() for k, v in policy.latent_heads.state_dict().items()}
    final = training.train_rl(policy, reference, Processor(), "unused", tmp_path / "rl", config, Evaluator(),
                              source=final)
    steps = [json.loads(line) for line in (tmp_path / "rl/steps.jsonl").read_text().splitlines()]
    assert len(steps) == 2
    assert {"reward_mean", "reward_std_within_group", "pass_at_group", "kl", "grad_norm"} <= steps[0].keys()
    assert any(not torch.equal(v, policy.latent_heads.state_dict()[k]) for k, v in heads.items())
    reloaded = ScaffoldingQwen.from_pretrained(final, attn_implementation="eager")
    assert reloaded.config.scaffold_phase == "rl" and hasattr(reloaded, "latent_heads")


def test_mid_epoch_resume_matches_uninterrupted_prediction(tiny_scaffold, fake_data, tmp_path):
    config = tiny_recipe(checkpoint_every_steps=1)
    targets = torch.randn(4, 4, 32, generator=torch.Generator().manual_seed(5))
    training.train_predict(tiny_scaffold, Processor(), targets, "unused", tmp_path / "full", config)
    checkpoint = tmp_path / "full/step-1"
    resumed = ScaffoldingQwen.from_pretrained(checkpoint, attn_implementation="eager")
    train_language(resumed)
    training.train_predict(resumed, Processor(), targets, "unused", tmp_path / "resumed", config, checkpoint)
    for key, expected in tiny_scaffold.state_dict().items():
        torch.testing.assert_close(expected, resumed.state_dict()[key], atol=0, rtol=0)


def test_prompt_order_and_final_checkpoint_contract(tmp_path):
    order = lambda epoch: torch.randperm(1000, generator=torch.Generator().manual_seed(42 + epoch)).tolist()
    assert training.prompt_indices(0, 8, 1000, 42) == order(0)[:8]
    # Positions 992..1007 finish epoch 0 and start epoch 1's fresh shuffle.
    assert training.prompt_indices(62, 16, 1000, 42) == order(0)[992:] + order(1)[:8]
    from spara.runtime import json_write
    config = read_config("configs/scaffolding.json")
    complete = {"phase": "predict", "completed": True, "optimizer_steps": 96, "config": config}
    for changes in [{"pilot": True}, {"completed": False}, {"optimizer_steps": 95}, {"phase": "scaffold"}]:
        json_write(tmp_path / "complete.json", {"progress": complete | changes})
        with pytest.raises(ValueError, match="completed predict"):
            require_final(tmp_path, "predict", config)
    json_write(tmp_path / "complete.json", {"progress": complete})
    require_final(tmp_path, "predict", config)
    # A different RL recipe may start from the same Stage 1 checkpoint ...
    require_final(tmp_path, "predict", config | {"rl": config["rl"] | {"variant": "text_only"}})
    # ... but Stage 1 settings must match.
    with pytest.raises(ValueError, match="recipe mismatch"):
        require_final(tmp_path, "predict", config | {"predict": config["predict"] | {"learning_rate": 1e-4}})


def test_rollouts_must_sample_the_policy_itself(tmp_path):
    config = json.loads(Path("configs/scaffolding.json").read_text())
    config["rl"]["rollout"]["top_p"] = 0.9
    (tmp_path / "bad.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="importance ratios"):
        read_config(tmp_path / "bad.json")


@pytest.fixture
def data_root():
    root = Path(os.environ.get("SPARA_TEST_DATA", "data/vsp"))
    if not (root / "manifest.json").is_file():
        pytest.skip("Run spara prepare-data first")
    return root


@pytest.mark.integration
def test_heatmap_values_encode_released_shortest_routes(data_root):
    from spara.data import records
    from spara.scaffolding.helper import value_iteration
    for row in records(data_root, "train"):
        moves = len(re.search(r"\\boxed\{(.*)\}", row["text_output"]).group(1).split(","))
        grid = np.asarray(row["map_desc"])
        start = tuple(np.argwhere(grid == 1)[0])
        assert np.isclose(value_iteration(grid)[start], 0.85 ** (moves - 1), rtol=1e-6), row["map_id"]


@pytest.mark.integration
def test_real_tokens_layout_helpers_and_reference_scoring(tiny_scaffold, data_root, tmp_path):
    from transformers import AutoProcessor, Qwen2_5_VLConfig
    from spara.data import records
    from spara.evaluation import ReferenceEvaluator
    from spara.scaffolding import serialization
    from spara.scaffolding.model import BOT, EOT, LATENT, add_scaffold_tokens, answer_sequence, scaffold_ids
    processor_path = Path(os.environ.get("SPARA_TEST_PROCESSOR", ".cache/processor"))
    if not processor_path.is_dir():
        pytest.skip("Prepare the processor")
    processor = AutoProcessor.from_pretrained(processor_path)
    config = Qwen2_5_VLConfig(**{k: v for k, v in tiny_scaffold.config.to_dict().items() if not k.startswith("scaffold")})
    model = ScaffoldingQwen(config)
    add_scaffold_tokens(model, processor, 4)
    assert scaffold_ids(model.config) == list(range(151665, 151672))
    recipe = read_config("configs/scaffolding.json")
    row = records(data_root, "train")[0]
    altered = dict(row, map_desc="MUST_NOT_APPEAR", map_id="MUST_NOT_APPEAR")
    prompt = serialization.prompt(processor, data_root, "train", altered)
    decoded = processor.tokenizer.decode(prompt["input_ids"][0])
    assert decoded.endswith("<|im_start|>assistant") and "MUST_NOT_APPEAR" not in decoded
    assert prompt["input_ids"].eq(151655).sum().item() == 121
    ids, bot = answer_sequence(prompt["input_ids"], serialization.answer(processor, row), model.config)
    text = processor.tokenizer.decode(ids[0])
    assert text.endswith("<|im_start|>assistant" + BOT + LATENT * 4 + EOT + row["text_output"] + "<|im_end|>")
    # The reference evaluator still finds the boxed route next to the structural tokens.
    assert ReferenceEvaluator(data_root).score(row, text)["success"]
    for kind in ("value_heatmap", "released"):
        helper = serialization.helper(processor, data_root, row, recipe | {"helper_image": kind})
        assert helper["image_grid_thw"].tolist() == [[1, 22, 22]]
    processor.save_pretrained(tmp_path)
    restored = AutoProcessor.from_pretrained(tmp_path)
    assert restored.tokenizer.convert_tokens_to_ids(model.config.scaffold_tokens) == scaffold_ids(model.config)


def test_pilot_correctness_checks_pass_on_tiny_model(tiny_scaffold, fake_data, monkeypatch, tmp_path):
    from PIL import Image
    from spara.scaffolding import pilot
    monkeypatch.setattr(pilot, "records", lambda *_: ROWS)
    monkeypatch.setattr(pilot, "helper_image", lambda *_: Image.new("RGB", (8, 8)))
    monkeypatch.setattr(pilot, "ReferenceEvaluator", lambda *_: Evaluator())
    pilot.run_pilot(tiny_recipe(), "unused", tmp_path / "pilot", tiny_scaffold, Processor())
    summary = json.loads((tmp_path / "pilot/summary.json").read_text())
    # Learning checks need the real model; every correctness check must hold even here.
    for name, result in summary["checks"].items():
        assert result["passed"] or name.endswith("tiny_fit"), name
    assert summary["estimated_gpu_hours"]["total"] > 0


def test_cli_phases_set_trainability_before_the_manifest(tiny_scaffold, fake_data, monkeypatch, tmp_path):
    import argparse
    import copy
    from spara.scaffolding import cli, runtime
    base = copy.deepcopy(tiny_scaffold)
    reload_encoder = ScaffoldingEncoder.from_pretrained
    monkeypatch.setattr(runtime, "load_base", lambda config, device: (copy.deepcopy(base), Processor()))
    monkeypatch.setattr(runtime, "load_checkpoint", lambda path, config, device="cpu", dtype=None:
                        (ScaffoldingQwen.from_pretrained(path, attn_implementation="eager"), Processor()))
    monkeypatch.setattr(runtime, "require_final", lambda *_: None)
    monkeypatch.setattr(ScaffoldingEncoder, "from_pretrained", classmethod(lambda cls, path, device="cpu": reload_encoder(path)))
    monkeypatch.setattr("spara.data.records", lambda *_: ROWS)
    monkeypatch.setattr("spara.evaluation.ReferenceEvaluator", lambda *_: Evaluator())
    config, parser = tiny_recipe(), argparse.ArgumentParser()
    args = lambda phase, **paths: argparse.Namespace(**({"phase": phase, "data_root": "unused", "resume": None,
                                                        "encoder": None, "checkpoint": None,
                                                        "output": str(tmp_path / phase)} | paths))
    vlm, _, lineage, run = cli.train(args("scaffold"), config, parser)
    assert not any(p.requires_grad for p in vlm.parameters()) and lineage["encoder_trainable_parameters"] > 0
    encoder = run()
    model, _, _, run = cli.train(args("predict", encoder=str(encoder)), config, parser)
    assert all(p.requires_grad != name.startswith("visual.") for name, p in model.named_parameters())
    checkpoint = run()
    model, _, _, run = cli.train(args("rl", checkpoint=str(checkpoint)), config, parser)
    assert all(p.requires_grad != name.startswith("visual.") for name, p in model.named_parameters())
    assert model.latent_heads.mean[0].weight.requires_grad
    assert json.loads((run() / "complete.json").read_text())["progress"]["completed"]
