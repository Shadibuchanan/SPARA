"""Scaffolding Minds baseline commands: pilot, the three training phases, evaluation.

    python -m spara.scaffolding.cli train --phase scaffold --base-model M --output S1A
    python -m spara.scaffolding.cli train --phase predict --base-model M --encoder S1A/final --output S1B
    python -m spara.scaffolding.cli train --phase rl --checkpoint S1B/final --output S2
    python -m spara.scaffolding.cli evaluate --checkpoint S2/final --output EVAL
    python -m spara.scaffolding.cli probe --checkpoint S1B/final --output PROBE
"""
import argparse
import gc
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m spara.scaffolding.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("pilot", "train", "evaluate", "probe"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--config", default="configs/scaffolding.json")
        cmd.add_argument("--data-root", default="data/vsp")
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--base-model", help="Verified local model downloaded by `spara.cli fetch-model`")
        if command == "train":
            cmd.add_argument("--phase", choices=["scaffold", "predict", "rl"], required=True)
            cmd.add_argument("--encoder", help="Final Scaffolding Phase checkpoint (predict phase)")
            cmd.add_argument("--checkpoint", help="Final Latent-Prediction checkpoint (rl phase)")
            cmd.add_argument("--resume", help="Trusted same-phase checkpoint including optimizer state")
        if command in ("evaluate", "probe"):
            cmd.add_argument("--checkpoint", required=True, help="Final predict (Stage 1) or rl (Stage 2) checkpoint")
        if command == "evaluate":
            cmd.add_argument("--evaluation-latent", choices=["sample", "mean", "none"],
                             help="Override the recipe's Stage 2 latent: sampled delta, its mean, or delta = 0")
        if command == "probe":
            cmd.add_argument("--sigmas", default="0.05,0.1,0.2,0.5,1,2",
                             help="Comma-separated isotropic adjustment scales")
            cmd.add_argument("--rows", type=int, default=64, help="Training maps, sampled with the recipe seed")
            cmd.add_argument("--draws", type=int, default=4, help="Adjustments per map and sigma")
    args = parser.parse_args(argv)

    import torch
    from ..data import audit
    from ..runtime import json_write, seed_all
    from .runtime import read_config, manifest
    config = read_config(args.config)
    if getattr(args, "evaluation_latent", None):
        config["evaluation_latent"] = args.evaluation_latent
    if args.base_model:
        config["local_model_path"] = str(Path(args.base_model).resolve())
    seed_all(config["seed"])
    audit(args.data_root)
    if not torch.cuda.is_available():
        raise RuntimeError("Use an allocated GPU for the actual 3B model; CPU tests use tiny random weights")
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists; choose a new directory")
    if args.command == "pilot":
        from .pilot import run_pilot
        from .runtime import load_base
        model, processor = load_base(config, device="cuda")
        run_pilot(config, args.data_root, output, model, processor, manifest(config, model, processor, args.data_root))
    elif args.command == "train":
        model, processor, lineage, run = train(args, config, parser)
        # train_* functions own `output`; provenance sits beside it, written before training.
        json_write(output.with_name(output.name + "-manifest.json"), manifest(config, model, processor, args.data_root, lineage))
        run()
    else:
        from .evaluation import evaluate
        from .probe import latent_probe
        from .runtime import load_checkpoint, require_final
        progress = json.loads((Path(args.checkpoint) / "complete.json").read_text())["progress"]
        if progress.get("phase") not in {"predict", "rl"}:
            parser.error(f"{args.command.capitalize()} a final predict (Stage 1) or rl (Stage 2) checkpoint")
        require_final(args.checkpoint, progress["phase"], config)
        model, processor = load_checkpoint(args.checkpoint, config, device="cuda")
        model.requires_grad_(False)
        run_manifest = manifest(config, model, processor, args.data_root,
                                {"checkpoint": str(args.checkpoint), "checkpoint_phase": progress["phase"]})
        if args.command == "probe":
            sigmas = [float(s) for s in args.sigmas.split(",")]
            latent_probe(model, processor, args.data_root, output, config, sigmas, args.rows, args.draws, run_manifest)
        else:
            evaluate(model, processor, args.data_root, output, config, run_manifest)


def train(args, config, parser):
    """Load inputs for one phase; returns the model to describe, its lineage and a deferred run.
    Trainability is set here, before the manifest records it."""
    import torch
    from ..data import records
    from ..evaluation import ReferenceEvaluator
    from .model import ScaffoldingEncoder, add_heads, add_latent_projection, load_token_rows, train_language
    from .runtime import load_base, load_checkpoint, require_final
    from .training import scaffold_targets, train_predict, train_rl, train_scaffold
    output, resume = Path(args.output), args.resume
    if args.phase == "scaffold":
        if args.encoder or args.checkpoint:
            parser.error("The Scaffolding Phase starts from the pinned base model")
        vlm, processor = load_base(config, device="cuda")
        if resume:
            encoder = ScaffoldingEncoder.from_pretrained(resume, device="cuda")
        else:
            encoder = ScaffoldingEncoder.from_vlm(vlm, config["latent_tokens"], config["scaffold"]["pooling_heads"])
        load_token_rows(vlm, encoder.context)
        vlm.requires_grad_(False)
        lineage = {"checkpoint_lineage": {"source": resume, "resume_optimizer": bool(resume)},
                   "encoder_trainable_parameters": sum(p.numel() for p in encoder.parameters() if p.requires_grad)}
        return vlm, processor, lineage, lambda: train_scaffold(vlm, encoder, processor, args.data_root,
                                                               output, config, resume)
    if args.phase == "predict":
        if not args.encoder:
            parser.error("The Latent-Prediction Phase needs the final Scaffolding Phase checkpoint (--encoder)")
        require_final(args.encoder, "scaffold", config)
        encoder = ScaffoldingEncoder.from_pretrained(args.encoder, device="cuda")
        if resume:
            model, processor = load_checkpoint(resume, config, device="cuda")
        else:
            model, processor = load_base(config, device="cuda")
            load_token_rows(model, encoder.context)
            add_latent_projection(model, config["predict"]["latent_projection"])
        targets = scaffold_targets(encoder, processor, args.data_root, records(args.data_root, "train"),
                                   config, model.device)
        del encoder
        gc.collect(); torch.cuda.empty_cache()
        train_language(model)
        lineage = {"checkpoint_lineage": {"encoder": args.encoder, "source": resume, "resume_optimizer": bool(resume)}}
        return model, processor, lineage, lambda: train_predict(model, processor, targets, args.data_root,
                                                                output, config, resume, args.encoder)
    if resume:
        reference_path = json.loads((Path(resume) / "complete.json").read_text())["progress"]["reference"]
        model, processor = load_checkpoint(resume, config, device="cuda")
    else:
        if not args.checkpoint:
            parser.error("Scaffolding RL needs the final Latent-Prediction checkpoint (--checkpoint)")
        reference_path = args.checkpoint
        model, processor = load_checkpoint(args.checkpoint, config, device="cuda")
        if config["rl"]["variant"] == "scaffolding":
            add_heads(model, config["rl"])
    train_language(model)
    require_final(reference_path, "predict", config)
    reference, _ = load_checkpoint(reference_path, config, device="cuda",
                                   dtype=getattr(torch, config["rl"]["reference_dtype"]))
    evaluator = ReferenceEvaluator(args.data_root)
    lineage = {"checkpoint_lineage": {"reference": reference_path, "source": resume or args.checkpoint,
                                      "resume_optimizer": bool(resume)}}
    return model, processor, lineage, lambda: train_rl(model, reference, processor, args.data_root, output,
                                                       config, evaluator, resume, reference_path)


if __name__ == "__main__":
    main()
