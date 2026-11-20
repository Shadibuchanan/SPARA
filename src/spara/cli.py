"""Commands separate preparation, diagnostic pilots, training and evaluation."""
import argparse
import contextlib
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
    for command in ("pilot", "train", "train-rl", "evaluate", "battery", "latent-probe", "explore-probe",
                    "planning-probe", "detour-probe", "guided-search", "detour-scaling", "jepa-judge",
                    "expert-iteration", "expert-round2", "perception-jepa", "region-latents", "fair-rl",
                    "latent-diagnostics", "latent-distillation"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--config", default="configs/livr.json")
        cmd.add_argument("--data-root", default="data/vsp")
        cmd.add_argument("--output", required=True)
        cmd.add_argument("--base-model", help="Verified local model downloaded by fetch-model")
        if command == "pilot":
            cmd.add_argument("--continue-from", help="Prior pilot directory with a completed Stage 1 roundtrip")
        if command != "pilot":
            cmd.add_argument("--checkpoint")
        if command in ("train-rl", "evaluate", "latent-probe", "explore-probe", "planning-probe", "detour-probe"):
            cmd.add_argument("--rl-config", default="configs/latent_rl.json"
                             if command.endswith("-probe") and command != "detour-probe" else "configs/text_rl.json")
        if command == "evaluate":
            cmd.add_argument("--latent-mode", choices=["sample", "mean", "zero"], default="sample",
                             help="Latent RL checkpoints: sampled adjustment, its mean, or none (LIVR's latents)")
        if command == "latent-probe":
            cmd.add_argument("--sigmas", default="0.01,0.02,0.05,0.1,0.2,0.5,1,2")
            cmd.add_argument("--rows", type=int, default=100, help="Random training maps")
            cmd.add_argument("--draws", type=int, default=2, help="Adjustments per map and sigma")
        if command == "explore-probe":
            cmd.add_argument("--split", required=True, help="Fresh map set, fresh/<name> (spara.fresh_maps)")
            cmd.add_argument("--basis-k", type=int, default=16, help="Principal directions per latent slot")
            cmd.add_argument("--samples", type=int, default=8, help="Sampled answers per map in the size sweep")
            cmd.add_argument("--draws", type=int, default=4, help="Adjustments per map and scale")
            cmd.add_argument("--isotropic", default="1.2,1.6,2.0,2.4", help="Isotropic sigmas")
            cmd.add_argument("--structured", default="0.5,1,2,4,8", help="Structured scales, in across-map spreads")
            cmd.add_argument("--noise-sizes", default="5,6", help="Grid sizes for the noise probe")
            cmd.add_argument("--max-new-tokens", type=int, default=128, help="Greedy answer cap")
        if command == "planning-probe":
            cmd.add_argument("--basis", help="basis.pt from explore-probe (structured latent basis); needed for headroom")
            cmd.add_argument("--fresh-split", default="", help="Fresh map set for the state-probe replication")
            cmd.add_argument("--fresh-sizes", default="5,6", help="Grid sizes of the fresh set to use")
            cmd.add_argument("--radii", default="1,2,4,8",
                             help="Headroom norm bounds per latent slot, in spreads; 'none' skips headroom")
            cmd.add_argument("--steps", type=int, default=40, help="Headroom optimisation steps per map")
            cmd.add_argument("--learning-rate", type=float, default=0.25, help="Headroom Adam learning rate")
            cmd.add_argument("--max-candidates", type=int, default=64, help="Shortest routes enumerated per map")
            cmd.add_argument("--max-new-tokens", type=int, default=128, help="Greedy answer cap")
        if command == "detour-probe":
            cmd.add_argument("--test-routes", required=True, help="Greedy test routes: a planning-probe routes.jsonl")
            cmd.add_argument("--fresh-routes", default="", help="Greedy fresh routes: an explore-probe sweep.jsonl")
            cmd.add_argument("--fresh-split", default="", help="Fresh map set of those routes, fresh/<name>")
        if command == "detour-scaling":
            cmd.add_argument("--arm", required=True, help="An arm named in the recipe: livr, n98, n500 or n2000")
            cmd.add_argument("--recipe", default="configs/detour_scaling.json")
        if command == "expert-iteration":
            cmd.add_argument("--stage", choices=["search", "train"], required=True)
            cmd.add_argument("--settings", default="configs/expert_iteration.json")
            cmd.add_argument("--probe-points", help="search: points.jsonl of a detour-probe run (validity check)")
            cmd.add_argument("--reference", help="search: the JEPA judge run's searches-verdict_only.jsonl")
            cmd.add_argument("--labels", choices=["search", "shortest"],
                             help="train: the found routes, or the same maps' shortest-route labels")
            cmd.add_argument("--routes", help="train: routes.jsonl of the search run")
        if command == "expert-round2":
            cmd.add_argument("--stage", choices=["batching-check", "search", "train"], required=True)
            cmd.add_argument("--settings", default="configs/expert_round2.json")
            cmd.add_argument("--batch", type=int, help="search: policy reads per pass, overriding the settings")
            cmd.add_argument("--read-precision", choices=["bf16", "tf32"],
                             help="search: precision of policy reads, overriding the settings")
            cmd.add_argument("--check-split", default="fresh/jepa-train",
                             help="batching-check: a fresh set with maps of every size from 3 to 8")
            cmd.add_argument("--round1-routes", help="search, train: routes.jsonl of round 1's search")
            cmd.add_argument("--round1-greedy", help="search: greedy.jsonl of round 1's E (loading check)")
            cmd.add_argument("--livr-checkpoint", help="search: LIVR's final Stage 2 checkpoint, for L's answers")
            cmd.add_argument("--arm", choices=["e2", "eb2"], help="train: E2, or EB2 under the bottleneck")
            cmd.add_argument("--search-run", help="train: the round 2 search job's output")
        if command == "perception-jepa":
            cmd.add_argument("--stage", choices=["gate", "train", "features"], required=True,
                             help="features: E's latents and cells for the fixed-target retry (gate 3 runs on CPU, "
                                  "scripts/fixed_target_gate.py); train: EJ2, or EJF from gate 3's output")
            cmd.add_argument("--settings", default="configs/perception_jepa.json")
            cmd.add_argument("--round2-settings", default="configs/expert_round2.json")
            cmd.add_argument("--round1-routes", required=True, help="routes.jsonl of round 1's search")
            cmd.add_argument("--search-run", required=True, help="the round 2 search job's output")
            cmd.add_argument("--e2-checkpoint", help="gate: E2's final weights, for its descriptive swap check")
            cmd.add_argument("--gate-run", help="train: the gate job's output (heads, centring means, weight)")
        if command == "region-latents":
            cmd.add_argument("--stage", choices=["train", "swap", "dropout", "tracking"], default="train",
                             help="swap: the swap test on a masked global + region checkpoint (EGR); "
                                  "dropout: EGD, global dropout from EGR's weights; "
                                  "tracking: the planning probe's state probe on a region-latent run's weights")
            cmd.add_argument("--recorded-masked", help="swap: masked.jsonl of the EGR run, for the reproduction check")
            cmd.add_argument("--recorded-run", help="tracking: the run whose greedy.jsonl and masked_hidden.jsonl "
                                                    "the routes must reproduce")
            cmd.add_argument("--modes", default="greedy,masked_hidden", help="tracking: answer modes, in order")
            cmd.add_argument("--settings", default="configs/region_latents.json")
            cmd.add_argument("--round2-settings", default="configs/expert_round2.json")
            cmd.add_argument("--round1-routes", help="train: routes.jsonl of round 1's search")
            cmd.add_argument("--search-run", help="train: the round 2 search job's output")
            cmd.add_argument("--gate-run", help="train: gate 3's output, the fixed targets (job 7118222)")
        if command == "fair-rl":
            cmd.add_argument("--stage", choices=["pilot", "train"], required=True,
                             help="pilot: on LIVR, before any run; train: one arm and seed, then its evaluation")
            cmd.add_argument("--settings", default="configs/fair_rl.json")
            cmd.add_argument("--arm", choices=["b", "c", "n"],
                             help="train: B′ (text only), C′ (structured latent) or C″ (fixed latent noise)")
            cmd.add_argument("--noise-basis", help="train, C″: the confirmed basis (a latent-diagnostics sweep's "
                                                   "bases.pt or its directory)")
            cmd.add_argument("--seed", type=int, help="train: map order and sampling seed")
            cmd.add_argument("--pilot-run", help="train: the pilot's output (maps per step, scale, basis)")
            cmd.add_argument("--low-memory", action="store_true",
                             help="train: AdamW state on the CPU and a bfloat16 KL reference, for GPUs under 80 GB")
        if command == "latent-diagnostics":
            cmd.add_argument("--part", choices=["sweep", "gradients", "checkpoints", "weights", "analyse", "confirm"],
                             required=True,
                             help="docs/LATENT_DIAGNOSTICS_SPEC.md: sweep, gradients (+ search) and checkpoints on a "
                                  "GPU; weights, analyse and confirm on the CPU")
            cmd.add_argument("--settings", default="configs/latent_diagnostics.json")
            cmd.add_argument("--pilot-run", help="The B′/C′ pilot's output (basis, maps per step, scale)")
            cmd.add_argument("--layers", help="sweep: comma-separated decoder layers, or none (text baseline only; default: the settings')")
            cmd.add_argument("--text-baseline", action="store_true", help="sweep: also the T = 1 text rollouts")
            cmd.add_argument("--answer-noise", action="store_true", help="sweep: also noise on the answer tokens")
            cmd.add_argument("--runs", help="checkpoints, weights: JSON {name: final checkpoint dir}, LIVR as 'livr'")
            cmd.add_argument("--sweeps", help="analyse: comma-separated sweep outputs")
            cmd.add_argument("--gradients-run", help="analyse: the gradients part's output")
            cmd.add_argument("--checkpoints-run", help="analyse: the checkpoints part's output")
            cmd.add_argument("--weights-run", help="analyse: the weights part's output")
            cmd.add_argument("--bases-from", help="sweep: reuse this earlier sweep's bases.pt (no capture)")
            cmd.add_argument("--text-run", help="confirm: the sweep holding the text baseline")
            cmd.add_argument("--reference-run", help="confirm: the earlier sweep, for the pooled (descriptive) reading")
            cmd.add_argument("--smoke", action="store_true", help="A few maps and draws per part (in-job check)")
        if command == "latent-distillation":
            cmd.add_argument("--part", choices=["search", "train", "compare"], required=True,
                             help="docs/LATENT_DISTILLATION_SPEC.md: search and train on a GPU; compare on the CPU")
            cmd.add_argument("--settings", default="configs/latent_distillation.json")
            cmd.add_argument("--noise-basis", help="search, train: the confirmed basis (bases.pt or its directory)")
            cmd.add_argument("--arm", choices=["S", "D"], help="train: S (found routes) or D (plus the latent target)")
            cmd.add_argument("--seed", type=int, help="train: data order and benchmark sampling seed")
            cmd.add_argument("--search-run", help="train, compare: the search's output")
            cmd.add_argument("--offload-optimizer", action="store_true", help="train: AdamW state on the CPU")
            cmd.add_argument("--accept-search", action="store_true",
                             help="train: accept a search that failed only its replay bar (spec amendment of 2026-11-19)")
            cmd.add_argument("--smoke", action="store_true", help="A few maps; two training steps; no benchmark")
            cmd.add_argument("--runs", help="compare: JSON {S42, S43, D42, D43: run directory}")
            cmd.add_argument("--livr-greedy", help="compare: LIVR's greedy answers (fair pilot's greedy-livr.jsonl)")
            cmd.add_argument("--livr-benchmark", help="compare: LIVR's benchmark responses.jsonl")
        if command == "jepa-judge":
            cmd.add_argument("--settings", default="configs/jepa_judge.json")
            cmd.add_argument("--probe-points", required=True, help="points.jsonl of a detour-probe run (validity check)")
        if command == "guided-search":
            cmd.add_argument("--probe-points", required=True, help="points.jsonl of a detour-probe run (validity check)")
            cmd.add_argument("--fresh-split", default="fresh/probe", help="Fresh map set whose detour maps to search")
            cmd.add_argument("--budget", type=int, default=64, help="Policy calls per map")
        if command == "train":
            cmd.add_argument("--stage", type=int, choices=[1, 2], required=True)
            cmd.add_argument("--grounding", help="Step-grounding settings (docs/STEP_GROUNDING_SPEC.md); Stage 2 only")
        if command == "battery":
            cmd.add_argument("--limit", type=int, help="Pilot: first LIMIT train and test rows of each grid size")
            cmd.add_argument("--max-new-tokens", type=int, default=128,
                             help="Greedy answer cap; LIVR's longest test answer is about 20 tokens")
        if command in ("train", "train-rl"):
            cmd.add_argument("--resume", help="Trusted same-stage checkpoint including optimizer state")
            if command == "train-rl":
                cmd.add_argument("--max-steps", type=int, help="Pilot: stop after this many updates")
                cmd.add_argument("--rl-seed", type=int, help="Seed for RL map order and rollout sampling (default: recipe seed)")
                cmd.add_argument("--evaluate-in-memory", type=int, metavar="SAMPLES",
                                 help="Skip saving final weights; run the benchmark and a SAMPLES-per-map diagnostic in memory")
                cmd.add_argument("--sigma-init", type=float,
                                 help="Latent RL: initial sigma chosen by latent-probe (overrides the recipe's)")
                cmd.add_argument("--evaluate-latent-modes", action="store_true",
                                 help="Latent RL: after training, run the benchmark with the mean, sampled and zero adjustment")
            cmd.add_argument("--final-weights-only", action="store_true",
                             help="Omit optimizer/RNG state from the final checkpoint to save storage")
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
    # Round 2's batching check runs the real model on CPU by design (full-precision reference).
    cpu_check = ((args.command == "expert-round2" and args.stage == "batching-check")
                 or (args.command == "latent-diagnostics" and args.part in ("weights", "analyse", "confirm"))
                 or (args.command == "latent-distillation" and args.part == "compare"))
    if not torch.cuda.is_available() and not cpu_check:
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
        grounding = None
        if args.grounding:
            from .grounding import StepGrounding, read_settings
            if args.stage != 2 or args.resume or not args.checkpoint:
                parser.error("Step grounding trains Stage 2 from final Stage 1 weights, without --resume")
            settings = read_settings(args.grounding)
            grounding = StepGrounding(settings, model.config.hidden_size, model.config.num_hidden_layers)
            run_manifest["grounding"] = settings
        # Save provenance before training; train_stage owns a new output directory.
        output = Path(args.output)
        if output.exists(): parser.error("Output already exists; choose a new directory")
        json_write(output.with_name(output.name + "-manifest.json"), run_manifest)
        train_stage(model, processor, args.data_root, output, config, args.stage, args.resume,
                    args.final_weights_only, grounding)
    elif args.command == "train-rl":
        from .text_rl import train_text_rl
        rl = json.loads(Path(args.rl_config).read_text(encoding="utf-8"))
        if args.rl_seed is not None:
            rl["seed"] = args.rl_seed
        latent = rl.get("latent")
        if args.sigma_init is not None:
            if not latent:
                parser.error("--sigma-init applies to latent RL recipes only")
            from .latent_rl import sigma_bounds
            latent |= {"sigma_init": args.sigma_init, "log_sigma_bounds": sigma_bounds(args.sigma_init)}
        if latent and not latent.get("sigma_init"):
            parser.error("Set the latent sigma from latent-probe (--sigma-init or the recipe)")
        if (args.evaluate_latent_modes and not latent) or (latent and args.evaluate_in_memory is not None):
            parser.error("--evaluate-latent-modes is for latent RL; --evaluate-in-memory for text-only RL")
        if not args.checkpoint:
            parser.error("RL requires the final Stage 2 LIVR checkpoint (also its KL reference)")
        require_final_checkpoint(args.checkpoint, 2, config)
        output = Path(args.output)
        if output.exists(): parser.error("Output already exists; choose a new directory")
        reference, _ = load_model(config, args.checkpoint, device="cuda")
        model, processor = load_model(config, args.resume or args.checkpoint, device="cuda")
        adjusters = None
        if latent:
            from .latent_rl import LatentAdjuster, add_latent_heads
            if getattr(model, "latent_heads", None) is None:
                if args.resume:
                    parser.error("Resume checkpoint lacks latent heads")
                add_latent_heads(model, latent)
            adjusters = (LatentAdjuster(model, latent["layer"]), LatentAdjuster(reference, latent["layer"]))
        run_manifest = manifest(config, model, processor, args.data_root)
        run_manifest["rl"] = rl
        run_manifest["checkpoint_lineage"] = {"source": args.checkpoint, "resume": args.resume}
        run_manifest["assumptions"] |= {
            "rl_objective": ("Scaffolding Minds Eq. 6 text term" + (" plus its latent term" if latent else " only")
                             + ": clipped ratio, k3 KL to the LIVR checkpoint, per-sequence token mean then mean over rollouts"),
            "rl_updates": "one optimizer step per rollout batch (old policy = current policy)",
            "rl_logprobs": "logits divided by rollout temperature; top-p not applied",
            "rl_schedule": "constant learning rate; fresh AdamW with LIVR betas/epsilon/weight decay",
            "rl_reward": "1 if the reference evaluator reports success, else 0",
        }
        if latent:
            run_manifest["assumptions"] |= {
                "latent_policy": f"a = z + delta at the output of decoder layer {latent['layer']} (one-based), "
                                 "delta ~ N(mu, sigma^2) from Scaffolding's Gaussian heads reading RMS-normalised z",
                "latent_objective": "Scaffolding eq. 6 latent term: clipped ratio of the Gaussian log-probability "
                                    f"({latent['logprob_reduction']} over K x d), gradient into z and the heads",
                "latent_reference": "the KL reference reads the rollout's latent action, so KL covers text only",
            }
        json_write(output.with_name(output.name + "-manifest.json"), run_manifest)
        if "seed" in rl:
            seed_all(rl["seed"])
        train_text_rl(model, reference, processor, args.data_root, output, config, rl,
                      args.checkpoint, args.resume, final_weights_only=args.final_weights_only,
                      max_steps=args.max_steps, save_final=args.evaluate_in_memory is None,
                      adjusters=adjusters)
        if args.evaluate_latent_modes and args.max_steps is None:
            from .benchmark import run_benchmark
            del reference
            adjusters[1].remove()
            torch.cuda.empty_cache()
            for mode in ("mean", "sample", "zero"):
                context = {"sample": adjusters[0].sample, "mean": adjusters[0].mean}.get(mode, contextlib.nullcontext)()
                with context:
                    run_benchmark(model, processor, config, args.data_root,
                                  output.with_name(f"{output.name}-evaluation-{mode}"),
                                  run_manifest | {"checkpoint": f"in-memory {output}", "latent_mode": mode})
        if args.evaluate_in_memory is not None:
            from .benchmark import run_benchmark, sample_test_maps
            del reference
            torch.cuda.empty_cache()
            run_benchmark(model, processor, config, args.data_root, output.with_name(output.name + "-evaluation"),
                          run_manifest | {"checkpoint": f"in-memory {output}"})
            sample_test_maps(model, processor, config, args.data_root, args.evaluate_in_memory,
                             output.with_name(output.name + "-multisample.jsonl"))
    elif args.command == "latent-probe":
        from .latent_rl import latent_probe
        rl = json.loads(Path(args.rl_config).read_text(encoding="utf-8"))
        if not args.checkpoint:
            parser.error("The latent probe requires the final Stage 2 LIVR checkpoint")
        require_final_checkpoint(args.checkpoint, 2, config)
        model, processor = load_model(config, args.checkpoint, device="cuda")
        model.config.livr_latent_layer = rl["latent"]["layer"]
        latent_probe(model, processor, args.data_root, args.output, [float(x) for x in args.sigmas.split(",")],
                     args.rows, args.draws, config["seed"],
                     manifest(config, model, processor, args.data_root) | {"checkpoint": str(args.checkpoint), "rl": rl})
    elif args.command == "explore-probe":
        from .explore_probe import explore_probe
        if not args.split.startswith("fresh/"):
            parser.error("--split must name a fresh map set, fresh/<name>")
        if not args.checkpoint:
            parser.error("The exploration probe requires the final Stage 2 LIVR checkpoint")
        rl = json.loads(Path(args.rl_config).read_text(encoding="utf-8"))
        require_final_checkpoint(args.checkpoint, 2, config)
        model, processor = load_model(config, args.checkpoint, device="cuda")
        fresh = json.loads((Path(args.data_root) / args.split / "manifest.json").read_text(encoding="utf-8"))
        floats = lambda text: [float(x) for x in text.split(",")]
        explore_probe(model, processor, args.data_root, args.output, args.split, rl,
                      manifest(config, model, processor, args.data_root)
                      | {"checkpoint": str(args.checkpoint), "rl": rl, "fresh_manifest": fresh},
                      config["seed"], basis_k=args.basis_k, samples=args.samples, draws=args.draws,
                      isotropic=floats(args.isotropic), structured=floats(args.structured),
                      noise_sizes=[int(x) for x in args.noise_sizes.split(",")], max_new_tokens=args.max_new_tokens)
    elif args.command == "planning-probe":
        from .planning_probe import planning_probe
        if not args.checkpoint:
            parser.error("The planning probe requires the final Stage 2 LIVR checkpoint")
        if args.fresh_split and not args.fresh_split.startswith("fresh/"):
            parser.error("--fresh-split must name a fresh map set, fresh/<name>")
        radii = [] if args.radii.strip().lower() in ("", "none") else [float(x) for x in args.radii.split(",")]
        if radii and not args.basis:
            parser.error("The headroom test needs --basis; pass --radii none to skip it")
        rl = json.loads(Path(args.rl_config).read_text(encoding="utf-8"))
        require_final_checkpoint(args.checkpoint, 2, config)
        model, processor = load_model(config, args.checkpoint, device="cuda")
        if model.config.livr_stage != 2:
            parser.error("The planning probe requires Stage 2")
        extra = {}
        if args.fresh_split:
            extra["fresh_manifest"] = json.loads((Path(args.data_root) / args.fresh_split / "manifest.json")
                                                 .read_text(encoding="utf-8"))
        planning_probe(model, processor, args.data_root, args.output, args.basis, rl["latent"]["layer"],
                       manifest(config, model, processor, args.data_root)
                       | {"checkpoint": str(args.checkpoint), "rl": rl} | extra,
                       config["seed"], fresh_split=args.fresh_split or None,
                       fresh_sizes=[int(x) for x in args.fresh_sizes.split(",")],
                       radii=radii, steps=args.steps,
                       learning_rate=args.learning_rate, max_candidates=args.max_candidates,
                       max_new_tokens=args.max_new_tokens)
    elif args.command == "detour-probe":
        from .detour_probe import detour_probe
        if not args.checkpoint:
            parser.error("The detour probe requires the final Stage 2 LIVR checkpoint")
        if bool(args.fresh_routes) != bool(args.fresh_split):
            parser.error("--fresh-routes and --fresh-split go together")
        if args.fresh_split and not args.fresh_split.startswith("fresh/"):
            parser.error("--fresh-split must name a fresh map set, fresh/<name>")
        rl = json.loads(Path(args.rl_config).read_text(encoding="utf-8"))
        require_final_checkpoint(args.checkpoint, 2, config)
        model, processor = load_model(config, args.checkpoint, device="cuda")
        if model.config.livr_stage != 2:
            parser.error("The detour probe requires Stage 2")
        extra = {}
        if args.fresh_split:
            extra["fresh_manifest"] = json.loads((Path(args.data_root) / args.fresh_split / "manifest.json")
                                                 .read_text(encoding="utf-8"))
        detour_probe(model, processor, args.data_root, args.output, args.test_routes, args.fresh_routes or None,
                     args.fresh_split or None, rl["generation"],
                     manifest(config, model, processor, args.data_root)
                     | {"checkpoint": str(args.checkpoint), "rl_config": args.rl_config} | extra, config["seed"])
    elif args.command == "guided-search":
        from .evaluation import ReferenceEvaluator
        from .route_search import search_pilot
        if not args.checkpoint:
            parser.error("The search pilot requires the final Stage 2 LIVR checkpoint")
        if args.fresh_split and not args.fresh_split.startswith("fresh/"):
            parser.error("--fresh-split must name a fresh map set, fresh/<name>")
        require_final_checkpoint(args.checkpoint, 2, config)
        model, processor = load_model(config, args.checkpoint, device="cuda")
        if model.config.livr_stage != 2:
            parser.error("The search pilot requires Stage 2")
        extra = {}
        if args.fresh_split:
            extra["fresh_manifest"] = json.loads((Path(args.data_root) / args.fresh_split / "manifest.json")
                                                 .read_text(encoding="utf-8"))
        search_pilot(model, processor, ReferenceEvaluator(args.data_root), args.data_root, args.output,
                     args.probe_points, args.fresh_split or None,
                     manifest(config, model, processor, args.data_root) | {"checkpoint": str(args.checkpoint)} | extra,
                     config["seed"], budget=args.budget)
    elif args.command == "jepa-judge":
        from .evaluation import ReferenceEvaluator
        from .jepa import run_jepa
        if not args.checkpoint:
            parser.error("The JEPA judge reads the final Stage 2 LIVR checkpoint's states")
        settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
        settings["veto"] = float(settings["veto"])  # "inf" allowed in JSON as a string
        require_final_checkpoint(args.checkpoint, 2, config)
        model, processor = load_model(config, args.checkpoint, device="cuda")
        if model.config.livr_stage != 2:
            parser.error("The JEPA judge requires Stage 2")
        model.requires_grad_(False)
        fresh = {split: json.loads((Path(args.data_root) / split / "manifest.json").read_text(encoding="utf-8"))
                 for split in (settings["train_split"], settings["heldout_split"], settings["fresh_split"])}
        run_jepa(model, processor, ReferenceEvaluator(args.data_root), args.data_root, args.output, settings,
                 manifest(config, model, processor, args.data_root)
                 | {"checkpoint": str(args.checkpoint), "fresh_manifests": fresh}, args.probe_points)
    elif args.command == "expert-iteration":
        from .evaluation import ReferenceEvaluator
        from .expert_iteration import run_search, run_train
        if not args.checkpoint:
            parser.error("Expert iteration starts from the final Stage 2 LIVR checkpoint")
        if args.stage == "search" and not args.probe_points:
            parser.error("The search stage needs --probe-points for its validity check")
        if args.stage == "train" and not (args.labels and args.routes):
            parser.error("The train stage needs --labels and --routes")
        settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
        require_final_checkpoint(args.checkpoint, 2, config)
        model, processor = load_model(config, args.checkpoint, device="cuda")
        if model.config.livr_stage != 2:
            parser.error("Expert iteration requires Stage 2")
        fresh = {split: json.loads((Path(args.data_root) / split / "manifest.json").read_text(encoding="utf-8"))
                 for split in (settings["pool_split"], settings["heldout_split"], settings["fresh_split"])}
        base = manifest(config, model, processor, args.data_root) | {"checkpoint": str(args.checkpoint),
                                                                    "fresh_manifests": fresh}
        if args.stage == "search":
            model.requires_grad_(False)
            run_search(model, processor, ReferenceEvaluator(args.data_root), args.data_root, args.output, settings,
                       base, args.probe_points, args.reference)
        else:
            run_train(model, processor, ReferenceEvaluator(args.data_root), config, args.data_root, args.output,
                      settings, args.labels, args.routes, base)
    elif args.command == "expert-round2":
        from . import expert_round2 as round2
        from .evaluation import ReferenceEvaluator
        if not args.checkpoint:
            parser.error("Round 2 starts from E, round 1's fine-tune on found routes")
        settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
        if args.batch:
            settings["batch"] = args.batch
        if args.read_precision:
            settings["read_precision"] = args.read_precision
        round2.require_round1_checkpoint(args.checkpoint, config, settings)
        if args.stage == "batching-check":
            model, processor = load_model(config, args.checkpoint, device="cpu")
            model.requires_grad_(False)
            Path(args.output).mkdir(parents=True, exist_ok=False)
            json_write(Path(args.output) / "manifest.json", manifest(config, model, processor, args.data_root)
                       | {"checkpoint": str(args.checkpoint), "threads": torch.get_num_threads()})
            round2.batching_check(model, processor, args.data_root, args.check_split, args.output)
            return
        needed = {"search": ("round1_routes", "round1_greedy", "livr_checkpoint"),
                  "train": ("round1_routes", "arm", "search_run")}[args.stage]
        if any(not getattr(args, name) for name in needed):
            parser.error(f"The {args.stage} stage needs " + ", ".join("--" + n.replace("_", "-") for n in needed))
        if args.stage == "search":
            require_final_checkpoint(args.livr_checkpoint, 2, config)
        fresh = {split: json.loads((Path(args.data_root) / split / "manifest.json").read_text(encoding="utf-8"))
                 for split in (settings["pool_split"], settings["heldout_split"])}
        model, processor = load_model(config, args.checkpoint, device="cuda")
        if model.config.livr_stage != 2:
            parser.error("Round 2 requires a Stage 2 checkpoint")
        base = manifest(config, model, processor, args.data_root) | {"checkpoint": str(args.checkpoint),
                                                                    "fresh_manifests": fresh}
        evaluator = ReferenceEvaluator(args.data_root)
        if args.stage == "search":
            round2.run_search(model, processor, evaluator, args.data_root, args.output, settings,
                              base | {"livr_checkpoint": str(args.livr_checkpoint)}, args.round1_routes,
                              args.round1_greedy)
            del model
            torch.cuda.empty_cache()
            model, processor = load_model(config, args.livr_checkpoint, device="cuda")
            model.requires_grad_(False)
            round2.livr_greedy(model, processor, evaluator, args.data_root, args.output, settings)
        else:
            round2.run_arm(model, processor, evaluator, config, args.data_root, args.output, settings, args.arm,
                           args.round1_routes, args.search_run, base)
    elif args.command == "fair-rl":
        from . import fair_rl
        from .evaluation import ReferenceEvaluator
        settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
        if not args.checkpoint:
            parser.error("B′ and C′ start from the LIVR rerun's final Stage 2 checkpoint (also the KL reference)")
        require_final_checkpoint(args.checkpoint, 2, config)
        if args.stage == "train" and (args.arm is None or args.seed is None or not args.pilot_run):
            parser.error("The train stage needs --arm, --seed and --pilot-run")
        if args.stage == "train" and args.arm == "n" and not args.noise_basis:
            parser.error("C″ needs --noise-basis (the confirmed latent-diagnostics basis)")
        output = Path(args.output)
        if output.exists():
            parser.error("Output already exists; choose a new directory")
        fresh = {split: json.loads((Path(args.data_root) / split / "manifest.json").read_text(encoding="utf-8"))
                 for split in (settings["pool_split"], settings["round1_heldout_split"], settings["heldout_split"])}
        evaluator = ReferenceEvaluator(args.data_root)
        arm = "b" if args.stage == "pilot" else args.arm  # the pilot measures B′'s sampler
        rl = json.loads(Path(settings["rl_configs"][arm]).read_text(encoding="utf-8"))
        if args.stage == "pilot":
            model, processor = load_model(config, args.checkpoint, device="cuda")
            model.requires_grad_(False)
            base = manifest(config, model, processor, args.data_root) | {"checkpoint": str(args.checkpoint),
                                                                        "fresh_manifests": fresh}
            fair_rl.run_pilot(model, processor, evaluator, args.data_root, output, settings, rl, base)
        else:
            reference, _ = load_model(config, args.checkpoint, device="cuda")
            if args.low_memory:
                # Same update and loss; about 35 GiB of GPU memory instead of about 70.
                reference.to(torch.bfloat16)
                rl |= {"offload_optimizer": True, "reference_dtype": "bfloat16"}
            model, processor = load_model(config, args.checkpoint, device="cuda")
            base = manifest(config, model, processor, args.data_root) | {"checkpoint": str(args.checkpoint),
                                                                        "fresh_manifests": fresh}
            base["assumptions"] |= {
                "rl_objective": "Scaffolding Minds Eq. 6 text term" + (" plus its latent term" if arm == "c" else "")
                                + ": one update per batch, k3 KL to LIVR, mean over the rollouts of groups whose rewards differ",
                "rl_sampler": "the model's own distribution: temperature, top-p, top-k and repetition penalty all set "
                              "so generate samples from the softmax the loss differentiates (checked in the pilot)",
                "rl_maps": "a stream of seeded permutations of the pool, cut into full batches; the same per seed for both arms"}
            if arm == "n":
                base["assumptions"]["latent_noise"] = (
                    f"a = z + offset at the output of decoder layer {rl['noise']['layer']}: offset = c * sum_j u_j w_j eps_j "
                    f"per latent slot, eps ~ N(0, 1), c = {rl['noise']['scale']}, in the confirmed {rl['noise']['family']} "
                    "basis; one draw per rollout; the update replays the offset on the recomputed z; no heads, no latent "
                    "term; no noise at evaluation")
            if arm == "c":
                base["assumptions"]["latent_policy"] = ("a = z + sum_j u_j s_j c_j at the output of decoder layer "
                                                        f"{rl['latent']['layer']}, c ~ N(mu, sigma^2) from heads over "
                                                        "the pilot's principal basis; log-probability summed over the coefficients")
            adjuster = fair_rl.train_arm(model, reference, processor, evaluator, config, args.data_root, output,
                                         settings, rl, arm, args.seed, args.pilot_run, args.checkpoint, base,
                                         args.noise_basis)
            del reference
            torch.cuda.empty_cache()
            fair_rl.evaluate_arm(model, processor, evaluator, config, args.data_root, output, settings, arm, adjuster,
                                 base)
    elif args.command == "latent-distillation":
        from . import latent_distillation as distillation
        from .evaluation import ReferenceEvaluator
        settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
        output = Path(args.output)
        evaluator = ReferenceEvaluator(args.data_root)
        if args.part == "compare":
            if not (args.runs and args.search_run and args.livr_greedy and args.livr_benchmark):
                parser.error("compare needs --runs, --search-run, --livr-greedy and --livr-benchmark")
            result = distillation.compare(json.loads(args.runs), args.search_run, args.livr_greedy,
                                          args.livr_benchmark, args.data_root, evaluator, settings)
            json_write(output, result)
            print(json.dumps({k: result[k] for k in ("validity", "primary")}, indent=1, default=str))
            return
        if output.exists():
            parser.error("Output already exists; choose a new directory")
        if not (args.checkpoint and args.noise_basis):
            parser.error("search and train start from the LIVR rerun's final Stage 2 checkpoint and need --noise-basis")
        require_final_checkpoint(args.checkpoint, 2, config)
        if args.part == "train" and (args.arm is None or args.seed is None or not args.search_run):
            parser.error("The train part needs --arm, --seed and --search-run")
        fresh = {split: json.loads((Path(args.data_root) / split / "manifest.json").read_text(encoding="utf-8"))
                 for split in (settings["pool_split"], settings["heldout_split"], settings["transfer_split"])}
        model, processor = load_model(config, args.checkpoint, device="cuda")
        if model.config.livr_stage != 2:
            parser.error("Latent distillation starts from Stage 2")
        base = manifest(config, model, processor, args.data_root) | {"checkpoint": str(args.checkpoint),
                                                                    "fresh_manifests": fresh}
        if args.part == "search":
            model.requires_grad_(False)
            distillation.run_search(model, processor, evaluator, args.data_root, output, settings, args.noise_basis,
                                    base, smoke=args.smoke)
        else:
            distillation.run_train(model, processor, evaluator, config, args.data_root, output, settings, args.arm,
                                   args.seed, args.search_run, args.noise_basis, base, offload=args.offload_optimizer,
                                   smoke=args.smoke, accept_search=args.accept_search)
    elif args.command == "latent-diagnostics":
        from . import latent_diagnostics as diagnostics
        from .evaluation import ReferenceEvaluator
        settings = diagnostics.load_settings(args.settings, smoke=args.smoke)
        output = Path(args.output)
        if output.exists() and args.part not in ("analyse", "confirm"):
            parser.error("Output already exists; choose a new directory")
        runs = json.loads(args.runs) if args.runs else {}
        if args.part == "weights":
            diagnostics.run_weights(runs, output)
            return
        if args.part == "analyse":
            sweeps = [p for p in (args.sweeps or "").split(",") if p]
            diagnostics.analyse(sweeps, args.gradients_run, args.checkpoints_run, args.weights_run, output, settings,
                                [name for name in runs])
            return
        if args.part == "confirm":
            if not (args.sweeps and args.text_run):
                parser.error("confirm needs --sweeps (the confirmation sweep) and --text-run")
            diagnostics.confirm(args.sweeps, args.text_run, settings, output, args.reference_run)
            return
        if not args.pilot_run:
            parser.error("The GPU parts need --pilot-run (the B′/C′ pilot)")
        evaluator = ReferenceEvaluator(args.data_root)
        if args.part == "checkpoints":
            for name, path in runs.items():
                if not (Path(path) / "complete.json").is_file():
                    parser.error(f"{name}: {path} is not a completed checkpoint")
            diagnostics.run_checkpoints(lambda path: load_model(config, path, device="cuda"), runs, evaluator,
                                        args.data_root, output, settings, args.pilot_run)
            return
        if not args.checkpoint:
            parser.error("The sweep and gradients parts run on the LIVR rerun's final Stage 2 checkpoint")
        require_final_checkpoint(args.checkpoint, 2, config)
        if args.part == "sweep":
            model, processor = load_model(config, args.checkpoint, device="cuda")
            base = manifest(config, model, processor, args.data_root) | {"checkpoint": str(args.checkpoint)}
            layers = ([] if args.layers == "none" else [int(x) for x in args.layers.split(",")]) if args.layers \
                else settings["layers"]
            diagnostics.run_sweep(model, processor, evaluator, args.data_root, output, settings, layers,
                                  args.text_baseline, args.answer_noise, base, args.bases_from)
        else:
            # As C′ trained in low-memory mode: a bfloat16 reference; no optimizer is built here.
            reference, _ = load_model(config, args.checkpoint, device="cuda")
            reference.to(torch.bfloat16)
            model, processor = load_model(config, args.checkpoint, device="cuda")
            base = manifest(config, model, processor, args.data_root) | {"checkpoint": str(args.checkpoint)}
            diagnostics.run_gradients(model, reference, processor, evaluator, config, args.data_root, output,
                                      settings, args.pilot_run, base)
            del reference
            torch.cuda.empty_cache()
            diagnostics.run_search(model, processor, evaluator, args.data_root, output, settings, args.pilot_run)
    elif args.command == "perception-jepa":
        from . import expert_round2 as round2
        from . import perception_jepa as jepa
        from .evaluation import ReferenceEvaluator
        if not args.checkpoint:
            parser.error("The latent JEPA arm starts from E, round 1's fine-tune on found routes")
        round2_settings = json.loads(Path(args.round2_settings).read_text(encoding="utf-8"))
        settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
        round2.require_round1_checkpoint(args.checkpoint, config, round2_settings)
        needed = {"gate": "e2_checkpoint", "train": "gate_run", "features": None}[args.stage]
        if needed and not getattr(args, needed):
            parser.error(f"The {args.stage} stage needs --{needed.replace('_', '-')}")
        if args.stage == "gate":
            jepa.require_round2_checkpoint(args.e2_checkpoint, config, round2_settings)
        fresh = {split: json.loads((Path(args.data_root) / split / "manifest.json").read_text(encoding="utf-8"))
                 for split in (round2_settings["pool_split"], round2_settings["heldout_split"])}
        model, processor = load_model(config, args.checkpoint, device="cuda")
        if model.config.livr_stage != 2:
            parser.error("The latent JEPA arm requires a Stage 2 checkpoint")
        base = manifest(config, model, processor, args.data_root) | {"checkpoint": str(args.checkpoint),
                                                                    "fresh_manifests": fresh}
        evaluator = ReferenceEvaluator(args.data_root)
        if args.stage == "gate":
            def load_e2():
                e2, _ = load_model(config, args.e2_checkpoint, device="cuda")
                e2.requires_grad_(False)
                return e2
            jepa.run_gate(model, processor, evaluator, config, args.data_root, args.output, settings, round2_settings,
                          args.round1_routes, args.search_run, load_e2,
                          base | {"e2_checkpoint": str(args.e2_checkpoint)})
        elif args.stage == "features":
            model.requires_grad_(False)
            jepa.run_features(model, processor, config, args.data_root, args.output, settings, round2_settings,
                              args.round1_routes, args.search_run, base)
        else:
            jepa.run_jepa_arm(model, processor, evaluator, config, args.data_root, args.output, round2_settings,
                              args.gate_run, args.round1_routes, args.search_run,
                              base | {"gate_run": str(args.gate_run)})
    elif args.command == "region-latents":
        from . import expert_round2 as round2
        from .evaluation import ReferenceEvaluator
        from .region_latents import run_global_dropout, run_latent_swap, run_region_arm
        if not args.checkpoint:
            parser.error("Give --checkpoint: E for training, the EGR checkpoint for the swap test")
        round2_settings = json.loads(Path(args.round2_settings).read_text(encoding="utf-8"))
        settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
        if args.stage == "swap":  # load_model checks the checkpoint is complete; run_latent_swap that it is EGR
            model, processor = load_model(config, args.checkpoint, device="cuda")
            model.requires_grad_(False)
            base = manifest(config, model, processor, args.data_root) | {"checkpoint": str(args.checkpoint)}
            run_latent_swap(model, processor, ReferenceEvaluator(args.data_root), args.data_root, args.output, settings,
                            round2_settings, base, args.recorded_masked)
            return
        if args.stage == "tracking":  # the state probe of docs/PLANNING_PROBE_SPEC.md on round 2's 7 x 7 / 8 x 8 maps
            from .planning_probe import tracking
            model, processor = load_model(config, args.checkpoint, device="cuda")
            if not getattr(model.config, "livr_region_latents", None):
                parser.error("The tracking stage reads a region-latent checkpoint")
            model.requires_grad_(False)
            modes = [m for m in args.modes.split(",") if m]
            recorded = ({m: Path(args.recorded_run) / f"{m}.jsonl" for m in modes} if args.recorded_run else {})
            missing = [str(p) for p in recorded.values() if not p.exists()]
            if missing:
                parser.error("Missing recorded routes: " + ", ".join(missing))
            base = manifest(config, model, processor, args.data_root) | {"checkpoint": str(args.checkpoint)}
            tracking(model, processor, args.data_root, args.output, base, config["seed"],
                     round2_settings["heldout_split"], round2_settings["pool_split"], modes, recorded,
                     max_new_tokens=round2_settings["max_new_tokens"])
            return
        if args.stage == "dropout":  # from EGR's weights; run_global_dropout checks that they are EGR's
            if not args.round1_routes or not args.search_run:
                parser.error("Global dropout needs --round1-routes and --search-run")
            model, processor = load_model(config, args.checkpoint, device="cuda")
            base = manifest(config, model, processor, args.data_root) | {"checkpoint": str(args.checkpoint)}
            run_global_dropout(model, processor, ReferenceEvaluator(args.data_root), config, args.data_root, args.output,
                               settings, round2_settings, args.round1_routes, args.search_run, base)
            return
        missing = [name for name in ("round1_routes", "search_run", "gate_run") if not getattr(args, name)]
        if missing:
            parser.error("Training needs --" + ", --".join(m.replace("_", "-") for m in missing))
        round2.require_round1_checkpoint(args.checkpoint, config, round2_settings)
        fresh = {split: json.loads((Path(args.data_root) / split / "manifest.json").read_text(encoding="utf-8"))
                 for split in (round2_settings["pool_split"], round2_settings["heldout_split"])}
        model, processor = load_model(config, args.checkpoint, device="cuda")
        if model.config.livr_stage != 2:
            parser.error("The region-latent arm requires a Stage 2 checkpoint")
        base = manifest(config, model, processor, args.data_root) | {"checkpoint": str(args.checkpoint),
                                                                    "fresh_manifests": fresh,
                                                                    "gate_run": str(args.gate_run)}
        run_region_arm(model, processor, ReferenceEvaluator(args.data_root), config, args.data_root, args.output,
                       settings, round2_settings, args.gate_run, args.round1_routes, args.search_run, base)
    elif args.command == "detour-scaling":
        from .detour_scaling import run_arm
        from .evaluation import ReferenceEvaluator
        recipe = json.loads(Path(args.recipe).read_text(encoding="utf-8"))
        if args.arm not in recipe["arms"]:
            parser.error(f"--arm must be one of {sorted(recipe['arms'])}")
        if not args.checkpoint:
            parser.error("The scaling check starts from the final Stage 2 LIVR checkpoint")
        require_final_checkpoint(args.checkpoint, 2, config)
        model, processor = load_model(config, args.checkpoint, device="cuda")
        if model.config.livr_stage != 2:
            parser.error("The scaling check requires Stage 2")
        fresh = {split: json.loads((Path(args.data_root) / split / "manifest.json").read_text(encoding="utf-8"))
                 for split in (recipe["heldout_split"], recipe["pool_split"], *recipe["other_fresh"])}
        run_arm(model, processor, ReferenceEvaluator(args.data_root), config, args.data_root, args.output, args.arm,
                recipe, manifest(config, model, processor, args.data_root)
                | {"checkpoint": str(args.checkpoint), "fresh_manifests": fresh})
    elif args.command == "battery":
        from .latent_battery import run_battery
        if not args.checkpoint:
            parser.error("The latent battery requires a final Stage 2 LIVR checkpoint")
        require_final_checkpoint(args.checkpoint, 2, config)
        model, processor = load_model(config, args.checkpoint, device="cuda")
        run_battery(model, processor, config, args.data_root, args.output,
                    manifest(config, model, processor, args.data_root) | {"checkpoint": str(args.checkpoint)},
                    limit=args.limit, max_new_tokens=args.max_new_tokens)
    else:
        from .benchmark import run_benchmark
        if not args.checkpoint:
            parser.error("Evaluation requires a final Stage 2 or text-only RL checkpoint")
        from .runtime import checkpoint_phase, require_final_rl_checkpoint
        phase = checkpoint_phase(args.checkpoint)
        if phase in ("text-rl", "latent-rl"):
            rl = json.loads(Path(args.rl_config).read_text(encoding="utf-8"))
            if phase == "latent-rl":
                # The recipe's sigma may have been set at training time with --sigma-init.
                saved = json.loads((Path(args.checkpoint) / "complete.json").read_text())["progress"]["rl"]
                if {k: v for k, v in saved.items() if k != "latent"} != {k: v for k, v in rl.items() if k != "latent"}:
                    parser.error("RL recipe mismatch")
                rl = saved
            require_final_rl_checkpoint(args.checkpoint, config, rl, phase)
        else:
            require_final_checkpoint(args.checkpoint, 2, config)
        model, processor = load_model(config, args.checkpoint, device="cuda")
        if model.config.livr_stage != 2:
            parser.error("Benchmark evaluation requires Stage 2")
        context, mode = contextlib.nullcontext(), None
        if phase == "latent-rl":
            from .latent_rl import LatentAdjuster
            adjuster, mode = LatentAdjuster(model), args.latent_mode
            context = {"sample": adjuster.sample, "mean": adjuster.mean}.get(mode, contextlib.nullcontext)()
        with context:
            run_benchmark(model, processor, config, args.data_root, args.output,
                          manifest(config, model, processor, args.data_root)
                          | {"checkpoint": str(args.checkpoint)} | ({"latent_mode": mode} if mode else {}))


if __name__ == "__main__":
    main()
