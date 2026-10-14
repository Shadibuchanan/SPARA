"""Commands separate preparation, diagnostic pilots, training and evaluation."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="spara")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("prepare-data")
    data.add_argument("--data-root", default="data/vsp")
    fetch = sub.add_parser("fetch-model")
    fetch.add_argument("--config", default="configs/livr.json")
    fetch.add_argument("--output", required=True)
    fetch.add_argument("--processor-only", action="store_true")
    for command in ("pilot", "train", "evaluate"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--config", default="configs/livr.json")
        cmd.add_argument("--data-root", default="data/vsp")
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--base-model", help="Verified local model downloaded by fetch-model")
        if command == "pilot":
            cmd.add_argument("--continue-from", help="Prior pilot directory with a completed Stage 1 roundtrip")
        if command != "pilot":
            cmd.add_argument("--checkpoint")
        if command == "train":
            cmd.add_argument("--stage", type=int, choices=[1, 2], required=True)
            cmd.add_argument("--resume", help="Trusted same-stage checkpoint including optimizer state")
    args = parser.parse_args()
    if args.command == "prepare-data":
        from .data import prepare
        report = prepare(args.data_root)
        print(json.dumps({k: v for k, v in report.items() if k != "splits"}, indent=2))
        return
    if args.command == "fetch-model":
        from .fetch import fetch_model
        config = json.loads(Path(args.config).read_text())
        print(fetch_model(config["model_id"], config["model_revision"], args.output, not args.processor_only))
        return
    import torch
    from .runtime import read_config, seed_all, load_model, manifest, json_write, require_final_checkpoint
    from .data import audit
    config = read_config(args.config)
    if args.base_model: config["local_model_path"] = str(Path(args.base_model).resolve())
    seed_all(config["seed"])
    audit(args.data_root)
    if not torch.cuda.is_available():
        raise RuntimeError("Use an allocated GPU for the actual 3B model; CPU correctness tests use tiny random weights")
    if args.command == "pilot":
        from .pilot import run_pilot
        run_pilot(config, args.data_root, args.output, args.continue_from)
    elif args.command == "train":
        from .training import train_stage
        source = args.resume or args.checkpoint
        if args.resume and args.checkpoint:
            parser.error("Specify resume OR weights-only checkpoint")
        # A one-segment run without latents is labelled Stage 2 and starts from the base model.
        if args.stage == 2 and not source and config.get("segments", 2) != 1:
            parser.error("Stage 2 requires final Stage 1 weights")
        if args.checkpoint and args.stage == 2:
            require_final_checkpoint(args.checkpoint, 1, config)
        model, processor = load_model(config, source, device="cuda")
        if args.checkpoint and args.stage == 2 and model.config.livr_stage != 1:
            parser.error("Stage 2 weights must come from Stage 1; use --resume for within-stage resume")
        if args.checkpoint and args.stage == 1:
            parser.error("Stage 1 starts from the pinned base model; use --resume to continue")
        run_manifest = manifest(config, model, processor, args.data_root)
        run_manifest["checkpoint_lineage"] = {"source": source, "resume_optimizer": bool(args.resume)}
        # Save provenance before training; train_stage owns a new output directory.
        output = Path(args.output)
        if output.exists(): parser.error("Output already exists; choose a new directory")
        json_write(output.with_name(output.name + "-manifest.json"), run_manifest)
        train_stage(model, processor, args.data_root, output, config, args.stage, args.resume)
    else:
        from .data import records
        from .evaluation import ReferenceEvaluator, summarize
        from .serialization import encode
        from .training import amp, to_device
        if not args.checkpoint:
            parser.error("Evaluation requires the final Stage 2 checkpoint")
        require_final_checkpoint(args.checkpoint, 2, config)
        model, processor = load_model(config, args.checkpoint, device="cuda")
        if model.config.livr_stage != 2:
            parser.error("Benchmark evaluation requires Stage 2")
        output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
        evaluator = ReferenceEvaluator(args.data_root)
        model.eval()
        generation = model.generation_config.to_dict() | config["generation"]
        run_manifest = manifest(config, model, processor, args.data_root)
        run_manifest.update(evaluator=evaluator.provenance, effective_generation=generation, checkpoint=str(args.checkpoint))
        json_write(output / "manifest.json", run_manifest)
        # Reset after loading/initialization so sampling begins from the recorded seed.
        seed_all(config["seed"])
        outputs = []
        with (output / "responses.jsonl").open("w", encoding="utf-8") as log:
            for index, row in enumerate(records(args.data_root, "test")):
                inputs = to_device(encode(processor, args.data_root, "test", row, training=False, latents=model.config.livr_latent_tokens), model.device)
                with torch.no_grad(), amp(model.device):
                    generated = model.generate(**inputs, **config["generation"], use_cache=True)
                result = evaluator.score(row, processor.tokenizer.decode(generated[0], skip_special_tokens=False))
                result["row_index"] = index
                outputs.append(result)
                log.write(json.dumps(result) + "\n"); log.flush()
                json_write(output / "summary.json", summarize(outputs))
                print(f"{index+1}/400 {summarize(outputs)['overall']}", flush=True)


if __name__ == "__main__":
    main()
