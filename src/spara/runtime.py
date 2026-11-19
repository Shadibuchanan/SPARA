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


def require_final_rl_checkpoint(path, config, rl, phase="text-rl"):
    """Completed RL from the LIVR checkpoint: text-only ("text-rl") or latent ("latent-rl")."""
    progress = json.loads((Path(path) / "complete.json").read_text())["progress"]
    if rl.get("steps"):  # B′ and C′: a fixed number of steps from a stream of map permutations
        incomplete = progress.get("optimizer_steps") != rl["steps"]
    else:
        expected_steps = rl["epochs"] * ((1000 + rl["prompts_per_step"] - 1) // rl["prompts_per_step"])
        incomplete = (progress.get("epoch") != rl["epochs"] or progress.get("next_row") != 0
                      or progress.get("optimizer_steps") != expected_steps)
    if progress.get("pilot") or progress.get("phase") != phase or incomplete:
        label = "text-only" if phase == "text-rl" else "latent"
        raise ValueError(f"Expected completed {label} RL, not an intermediate checkpoint")
    for key, value in config.items():
        if key not in {"local_model_path", "checkpoint_every_steps"} and progress["config"].get(key) != value:
            raise ValueError(f"Checkpoint recipe mismatch: {key}")
    for key, value in rl.items():
        if key != "checkpoint_every_steps" and progress["rl"].get(key) != value:
            raise ValueError(f"RL recipe mismatch: {key}")
    return progress


def checkpoint_phase(path):
    return json.loads((Path(path) / "complete.json").read_text())["progress"].get("phase")


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
        region = getattr(model.config, "livr_region_latents", None)  # spara.region_latents
        tokens = region["tokens"] if region else LATENT_TOKENS
        ids = processor.tokenizer.convert_tokens_to_ids(tokens) if latents else []
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


def make_optimizer(model, config, rates=None):
    """AdamW without decay on biases and norms. `rates` maps a parameter-name prefix to
    its own learning rate; those groups follow the two default ones."""
    rates = rates or {}
    groups = {(prefix, decay): [] for prefix in (None, *rates) for decay in (True, False)}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            prefix = next((p for p in rates if name.startswith(p)), None)
            groups[prefix, not (name.endswith("bias") or "norm" in name.lower())].append(parameter)
    return torch.optim.AdamW([
        {"params": params, "weight_decay": config["weight_decay"] if decay else 0.0}
        | ({} if prefix is None else {"lr": rates[prefix]})
        for (prefix, decay), params in groups.items()], lr=config["learning_rate"],
        betas=tuple(config["betas"]), eps=config["epsilon"],
        fused=next(model.parameters()).device.type == "cuda")


class _Shadow:
    """Named CPU copies of a model's trainable parameters, for make_optimizer."""

    def __init__(self, pairs):
        self.pairs = pairs

    def named_parameters(self):
        return ((name, cpu) for name, _, cpu in self.pairs)

    def parameters(self):
        return (cpu for _, _, cpu in self.pairs)


def offloaded_optimizer(model, config, rates=None):
    """make_optimizer's AdamW with its fp32 master weights and moments on the CPU. Before each
    step the GPU gradients move to the CPU copies (and are freed); after it the updated weights
    move back. The update is AdamW on the same fp32 values, so it saves about 3x the weights in
    GPU memory (28 GiB of moments for the 3B model) at the cost of a CPU step and two transfers."""
    pairs = [(name, p, torch.nn.Parameter(p.detach().to("cpu", copy=True)))
             for name, p in model.named_parameters() if p.requires_grad]
    optimizer = make_optimizer(_Shadow(pairs), config, rates)

    def to_cpu(*_):
        for _, gpu, cpu in pairs:
            cpu.grad = None if gpu.grad is None else gpu.grad.detach().to("cpu")
            gpu.grad = None

    @torch.no_grad()
    def to_gpu(*_):
        for _, gpu, cpu in pairs:
            gpu.copy_(cpu)

    optimizer.register_step_pre_hook(to_cpu)
    optimizer.register_step_post_hook(to_gpu)
    return optimizer


def save_checkpoint(path, model, processor, optimizer, scheduler, progress, training_state=True):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model.save_pretrained(path)
    processor.save_pretrained(path)
    state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "progress": progress, "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
             "python_rng": random.getstate(), "numpy_rng": np.random.get_state()}
    # A completed stage needs only weights; omitting its optimizer state saves storage.
    if training_state:
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
