"""Recipe checks, model loading, provenance and final-checkpoint validation."""
import json
import math
import os
import platform
import subprocess
from pathlib import Path

import torch
from transformers import AutoProcessor

from ..data import sha256
from .model import ScaffoldingQwen, add_scaffold_tokens, scaffold_ids

PHASES = ("scaffold", "predict", "rl")
TRAIN_ROWS = 1000

# Choices the paper leaves open or that adapt it to this benchmark; mirrored in
# docs/SCAFFOLDING_BASELINE_SPEC.md.
ASSUMPTIONS = {
    "benchmark": "Mirage VSP 3x3-6x6 split and reference evaluator, not the paper's 8x8-32x32 FrozenLake",
    "backbone": "Qwen2.5-VL-3B-Instruct, matching the LIVR baseline, not the paper's 7B",
    "helper": "value-function heatmap from map_desc (App. E.1 code), one flat colour per cell at 320x320",
    "latent_layout": "one block: assistant header (no newline), BOT, K slots, EOT, answer",
    "special_tokens": "BOT/EOT rows copied from vision start/end; slot and query rows N(0, initializer_range)",
    "latent_definition": "z is the final-norm hidden state at the K query positions, optionally through a "
                         "zero-initialized d->d linear projection (predict.latent_projection)",
    "pooling": "one pre-norm cross-attention block plus d->d MLP over [helper tokens; queries]",
    "vision_tower": "the VLM's own vision tower stays frozen in every phase, as for LIVR",
    "latent_loss": "predict.latent_reduction: 'sum' is eq. 2 as written (sum over d, mean over K); 'mean' is the "
                   "MSE reading. The literal sum with no projection collapsed the language model in job 6257067",
    "epochs": "paper's 3 epochs and batch 32 per Stage 1 phase, applied to 1,000 rows",
    "rl_schedule": "cosine with 5% warmup; prompts per step and step count are not in the paper",
    "rollouts": "temperature-only policy sampling; one update per rollout batch (GRPO default)",
    "kl": "exact per-token KL over the vocabulary against the frozen Stage 1 model",
    "suppression": "slot/query/EOT tokens never sampled; BOT blocked after the single block",
    "rl_data": "RL prompts are the same 1,000 training rows",
    "rl_stop": "RL stops after rl.stop_after_zero_reward_steps consecutive all-zero-reward steps (no gradient)",
    "inference": "delta sampled from the learned Gaussian at evaluation (paper §4)",
}


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("method") != "scaffolding-minds":
        raise ValueError("Not a Scaffolding Minds recipe")
    for phase in ("scaffold", "predict"):
        if config[phase]["microbatch"] != 1:
            raise ValueError("The implementation uses microbatch one with gradient accumulation")
    rollout, rl = config["rl"]["rollout"], config["rl"]
    if rollout.get("top_k", 0) or rollout.get("top_p", 1.0) != 1.0 or rollout.get("repetition_penalty", 1.0) != 1.0:
        raise ValueError("Rollouts must sample the policy itself (temperature only) for exact importance ratios")
    if rl["variant"] not in {"scaffolding", "text_only"}:
        raise ValueError(f"Unknown RL variant: {rl['variant']}")
    if config["predict"]["latent_reduction"] not in {"sum", "mean"}:
        raise ValueError("predict.latent_reduction must be 'sum' or 'mean'")
    if config["predict"]["latent_projection"] not in {None, "linear"}:
        raise ValueError("predict.latent_projection must be null or 'linear'")
    if int(rl["stop_after_zero_reward_steps"]) < 0:
        raise ValueError("rl.stop_after_zero_reward_steps must be >= 0 (0 disables)")
    if config["helper_image"] not in {"value_heatmap", "released", "row"}:
        raise ValueError(f"Unknown helper image: {config['helper_image']}")
    return config


def recipe(config, phase):
    """Settings that determine a checkpoint of `phase` (later phases may vary)."""
    later = PHASES[PHASES.index(phase) + 1:]
    ignored = {"local_model_path", "checkpoint_every_steps", "generation", "evaluation_latent", *later}
    return {key: value for key, value in config.items() if key not in ignored}


def expected_steps(config, phase):
    if phase == "rl":
        return config["rl"]["steps"] * config["rl"]["ppo_epochs"]
    settings = config[phase]
    return settings["epochs"] * math.ceil(config.get("train_rows", TRAIN_ROWS) / settings["gradient_accumulation"])


def require_final(path, phase, config):
    progress = json.loads((Path(path) / "complete.json").read_text(encoding="utf-8"))["progress"]
    if (progress.get("pilot") or progress.get("phase") != phase or not progress.get("completed")
            or progress.get("optimizer_steps") != expected_steps(config, phase)):
        raise ValueError(f"Expected a completed {phase} checkpoint, not a pilot or intermediate one")
    if recipe(progress["config"], phase) != recipe(config, phase):
        raise ValueError("Checkpoint recipe mismatch")
    return progress


def load_base(config, device="cpu"):
    source = config.get("local_model_path", config["model_id"])
    revision = {} if config.get("local_model_path") else {"revision": config["model_revision"]}
    if config.get("local_model_path"):
        downloaded = json.loads((Path(source) / "download-manifest.json").read_text())
        if downloaded["revision"] != config["model_revision"]:
            raise ValueError("Local model revision disagrees with the recipe")
    processor = AutoProcessor.from_pretrained(source, **revision)
    model = ScaffoldingQwen.from_pretrained(source, torch_dtype=torch.float32,
                                           attn_implementation=config["attention_backend"], **revision)
    add_scaffold_tokens(model, processor, config["latent_tokens"])
    model.config.scaffold_phase = "base"
    return model.to(device), processor


def load_checkpoint(path, config, device="cpu", dtype=torch.float32):
    if not (Path(path) / "complete.json").is_file():
        raise ValueError("Checkpoint is incomplete or was not written by this pipeline")
    processor = AutoProcessor.from_pretrained(path)
    model = ScaffoldingQwen.from_pretrained(path, torch_dtype=dtype, attn_implementation=config["attention_backend"])
    if processor.tokenizer.convert_tokens_to_ids(model.config.scaffold_tokens) != scaffold_ids(model.config):
        raise ValueError("Checkpoint tokenizer and scaffold token IDs disagree")
    return model.to(device), processor


def manifest(config, model, processor, data_root, extra=None):
    root = Path(__file__).resolve().parents[3]
    hashes = {str(p.relative_to(root)): sha256(p) for folder in ("src", "configs", "scripts")
              for p in sorted((root / folder).rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    count = lambda trainable: sum(p.numel() for p in model.parameters() if p.requires_grad == trainable)
    return {
        "method": "Scaffolding Minds (arXiv:2608.19669v2), SPARA VSP adaptation",
        "config": config, "python": platform.python_version(), "torch": torch.__version__,
        "packages": subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
        "source_sha256": hashes, "data_manifest": json.loads((Path(data_root) / "manifest.json").read_text()),
        "scaffold_tokens": dict(zip(model.config.scaffold_tokens, scaffold_ids(model.config))),
        "model_files": json.loads((Path(config["local_model_path"]) / "download-manifest.json").read_text())
        if config.get("local_model_path") else {"revision": config["model_revision"]},
        "trainable_parameters": count(True), "frozen_parameters": count(False),
        "processor": processor.to_dict(), "image_processor": processor.image_processor.to_dict(),
        "generation_config": model.generation_config.to_dict(),
        "hardware": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "cuda": torch.version.cuda, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "assumptions": ASSUMPTIONS, **(extra or {}),
    }
