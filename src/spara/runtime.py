"""Run provenance, deterministic setup and checkpoint state."""
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor

from .data import sha256
from .model import LivrQwen, LATENT_TOKENS, freeze_vision, initialize_latents


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["microbatch"] != 1:
        raise ValueError("The baseline implements the reference microbatch of one")
    return config


def require_final_checkpoint(path, stage, config):
    progress = json.loads((Path(path) / "complete.json").read_text())["progress"]
    expected_steps = config["epochs_per_stage"] * ((1000 + config["gradient_accumulation"] - 1) // config["gradient_accumulation"])
    if (progress.get("pilot") or progress.get("stage") != stage
            or progress.get("epoch") != config["epochs_per_stage"]
            or progress.get("next_row") != 0 or progress.get("optimizer_steps") != expected_steps):
        raise ValueError(f"Expected completed Stage {stage} training, not a pilot or intermediate checkpoint")
    for key, value in config.items():
        if key not in {"local_model_path", "checkpoint_every_steps"} and progress["config"].get(key) != value:
            raise ValueError(f"Checkpoint recipe mismatch: {key}")


def load_model(config, checkpoint=None, device="cpu"):
    source = str(checkpoint) if checkpoint else config.get("local_model_path", config["model_id"])
    if checkpoint and not (Path(source) / "complete.json").is_file():
        raise ValueError("Checkpoint is incomplete or was not written by this training pipeline")
    revision = {} if checkpoint or config.get("local_model_path") else {"revision": config["model_revision"]}
    if not checkpoint and config.get("local_model_path"):
        downloaded = json.loads((Path(source) / "download-manifest.json").read_text())
        if downloaded["revision"] != config["model_revision"]:
            raise ValueError("Local model revision disagrees with the recipe")
    processor = AutoProcessor.from_pretrained(source, **revision)
    # FP32 trainable parameters and optimizer states; autocast supplies BF16 ops.
    model = LivrQwen.from_pretrained(source, torch_dtype=torch.float32,
                                    attn_implementation=config["attention_backend"], **revision)
    latents = config.get("latent_tokens", 4)
    if checkpoint:
        ids = processor.tokenizer.convert_tokens_to_ids(LATENT_TOKENS) if latents else []
        if ids != model.config.livr_latent_ids:
            raise ValueError("Checkpoint tokenizer/latent IDs disagree")
    elif latents:
        initialize_latents(model, processor)
    else:
        # Vanilla SFT control: no latent tokens and no Stage 1 bottleneck.
        model.config.livr_latent_ids, model.config.livr_latent_tokens = [], []
        model.config.livr_stage = 2
        model.config.livr_serialization = "user-end,assistant-header-no-newline,answer,im-end"
    freeze_vision(model)
    return model.to(device), processor


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def manifest(config, model, processor, data_root):
    root = Path(__file__).resolve().parents[2]
    hashes = {str(p.relative_to(root)): sha256(p) for folder in ("src", "configs", "scripts")
              for p in sorted((root / folder).rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    return {
        "config": config, "python": platform.python_version(), "torch": torch.__version__,
        "packages": subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
        "source_sha256": hashes, "data_manifest": json.loads((Path(data_root) / "manifest.json").read_text()),
        "latent_ids": model.config.livr_latent_ids,
        "model_files": json.loads((Path(config["local_model_path"]) / "download-manifest.json").read_text()) if config.get("local_model_path") else {"revision": config["model_revision"]},
        "trainable_parameters": {name: p.numel() for name, p in model.named_parameters() if p.requires_grad},
        "frozen_parameters": {name: p.numel() for name, p in model.named_parameters() if not p.requires_grad},
        "processor": processor.to_dict(), "image_processor": processor.image_processor.to_dict(),
        "generation_config": model.generation_config.to_dict(),
        "hardware": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "cuda": torch.version.cuda, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "assumptions": {
            "embeddings": "normal(0, config.initializer_range); full old embeddings/head trainable",
            "loss": "released answer plus im_end; no trailing newline; equal per-example mean",
            "precision": "FP32 parameters/AdamW state, BF16 autocast, no scaler",
            "clipping": config["max_grad_norm"], "adamw": "PyTorch, decay excludes biases and norm parameters",
            "checkpoint": "final stage weights; fresh optimizer/schedule at stage transition",
        },
    }


def make_optimizer(model, config):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if name.endswith("bias") or "norm" in name.lower() else decay).append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": config["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0}], lr=config["learning_rate"],
        betas=tuple(config["betas"]), eps=config["epsilon"],
        fused=next(model.parameters()).device.type == "cuda")


def save_checkpoint(path, model, processor, optimizer, scheduler, progress):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model.save_pretrained(path)
    processor.save_pretrained(path)
    state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "progress": progress, "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}
    torch.save(state, path / "training_state.pt")
    json_write(path / "complete.json", {"progress": progress, "save_seconds": time.perf_counter() - started,
                                        "bytes": sum(p.stat().st_size for p in path.iterdir() if p.is_file())})


def restore_training(path, optimizer, scheduler):
    if not (Path(path) / "complete.json").is_file():
        raise ValueError("Incomplete checkpoint")
    # Only use checkpoints produced by this run: this state contains Python RNG objects.
    state = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    return state["progress"]
