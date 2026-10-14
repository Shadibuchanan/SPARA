"""Validate one vanilla SFT arm and export a compact evidence bundle like livr-6189457."""
import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path

from spara.data import layout_key, records
from spara.evaluation import ReferenceEvaluator, summarize

LIVR = Path(__file__).resolve().parents[1] / "docs/experiments/livr-6189457"
ARMS = {"a": {"config": "configs/sft.json", "segments": [("a-train", 2)], "evaluation": "a-evaluation"},
        "c": {"config": "configs/sft-staged.json", "segments": [("c-segment1", 1), ("c-segment2", 2)],
              "evaluation": "c-evaluation"}}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--run", type=Path, required=True, help="results/<run id> written by run-sft.sh")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Defaults to docs/experiments/sft-<arm>-<run id>")
    args = parser.parse_args()
    arm = ARMS[args.arm]
    output = args.output or project / f"docs/experiments/sft-{args.arm}-{args.run.name}"
    recipe = json.loads((project / arm["config"]).read_text())
    livr = json.loads((LIVR / "provenance.json").read_text())
    checks, artifacts = {}, {}
    output.mkdir(parents=True, exist_ok=True)
    epochs = recipe["epochs_per_stage"]
    for name, stage in arm["segments"]:
        progress = json.loads((args.run / name / "final/complete.json").read_text())["progress"]
        assert progress["stage"] == stage and not progress.get("pilot")
        assert progress["epoch"] == epochs and progress["next_row"] == 0 and progress["optimizer_steps"] == 125 * epochs
        assert all(progress["config"][key] == value for key, value in recipe.items())
        steps = [json.loads(line) for line in (args.run / name / "steps.jsonl").read_text().splitlines()]
        assert [row["step"] for row in steps] == list(range(1, 125 * epochs + 1))
        assert all(math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"]) for row in steps)
        assert steps[-1]["lr"] == 0
        manifest = json.loads((args.run / f"{name}-manifest.json").read_text())
        assert manifest["latent_ids"] == [] and manifest["config"]["latent_tokens"] == 0
        lineage = manifest["checkpoint_lineage"]
        assert not lineage["resume_optimizer"]
        if name == arm["segments"][0][0]:
            assert lineage["source"] is None
        else:
            assert Path(lineage["source"]).resolve() == (args.run / arm["segments"][0][0] / "final").resolve()
        target = f"{name.split('-', 1)[1]}-steps.jsonl"
        (output / target).write_text((args.run / name / "steps.jsonl").read_text())
        checks[name] = {"stage_label": stage, "epochs": epochs, "updates": len(steps), "final_lr": steps[-1]["lr"],
                        "finite_loss_and_gradients": True, "checkpoint_source": lineage["source"],
                        "fresh_optimizer": True}
    evaluation = args.run / arm["evaluation"]
    manifest = json.loads((evaluation / "manifest.json").read_text())
    assert manifest["evaluator"] == livr["evaluator"], "Evaluator provenance differs from 6189457"
    rows = records(args.data_root, "test")
    outputs = [json.loads(line) for line in (evaluation / "responses.jsonl").read_text().splitlines()]
    assert len(outputs) == len(rows) == 400 and [o["row_index"] for o in outputs] == list(range(400))
    assert [o["map_id"] for o in outputs] == [r["map_id"] for r in rows]
    assert len({(o["grid_size"], o["map_id"]) for o in outputs}) == 400
    evaluator = ReferenceEvaluator(args.data_root)
    for row, result in zip(rows, outputs):
        assert all(result[k] == v for k, v in evaluator.score(row, result["raw_response"]).items())
    summary = summarize(outputs)
    assert summary == json.loads((evaluation / "summary.json").read_text())
    # Zero-based test rows whose layout also appears in training (LIVR's 315/392 diagnostic).
    train_layouts = {layout_key(row) for row in records(args.data_root, "train")}
    overlap = [i for i, row in enumerate(rows) if layout_key(row) in train_layouts]
    assert len(overlap) == 8
    kept = [o for o in outputs if o["row_index"] not in overlap]
    summary["excluding_overlap"] = summarize(kept)["overall"] | {"excluded_rows": overlap}
    (output / "evaluation-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # Omit the repeated prompt and image-token prefix, as the LIVR export does.
    keys = ["row_index", "map_id", "grid_size", "answer", "extracted_answer", "success", "status", "invalid"]
    (output / "evaluation-outcomes.jsonl").write_text("".join(json.dumps({k: o[k] for k in keys}) + "\n" for o in outputs))
    budget = [json.loads(line) for line in (args.run / "budget.jsonl").read_text().splitlines()]
    phases = [b for b in budget if b["phase"].startswith(f"{args.arm}-")]
    for path in sorted(output.iterdir()):
        if path.name not in {"provenance.json", "README.md"}:
            artifacts[path.name] = sha256(path)
    provenance = {
        "run_id": f"sft-{args.arm}-{args.run.name}", "date": date.today().isoformat(),
        "model": recipe["model_id"], "model_revision": recipe["model_revision"], "config": recipe,
        "config_sha256": sha256(project / arm["config"]),
        "livr_config_sha256": sha256(project / "configs/livr.json"),
        "comparison": "docs/experiments/livr-6189457", "hardware": manifest["hardware"],
        "comparison_hardware": livr["hardware"], "cuda": manifest["cuda"], "python": manifest["python"],
        "torch": manifest["torch"], "evaluator": manifest["evaluator"],
        "effective_generation": manifest["effective_generation"], "data_manifest": manifest["data_manifest"],
        "latent_ids": manifest["latent_ids"],
        "trainable_parameters": sum(manifest["trainable_parameters"].values()),
        "frozen_parameters": sum(manifest["frozen_parameters"].values()),
        "gpu_seconds": {b["phase"]: b["gpu_seconds"] for b in phases},
        "completed_checks": checks | {"all_400_rescored": True, "evaluator_matches_6189457": True,
                                      "map_ids_in_original_order": True},
        "source_sha256": manifest["source_sha256"], "artifact_sha256": artifacts,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    overall, kept = summary["overall"], summary["excluding_overlap"]
    steps = ", ".join(f"`{n.split('-', 1)[1]}-steps.jsonl`" for n, _ in arm["segments"])
    (output / "README.md").write_text(f"""# Evidence for vanilla SFT arm {args.arm.upper()} ({args.run.name})

Vanilla SFT control for [LIVR run 6189457](../livr-6189457/): `{arm["config"]}`, no latent
tokens and no Stage-1 bottleneck, {125 * epochs * len(arm["segments"]):,} optimizer updates. See
[the plan](../../plans/VANILLA_SFT_CONTROL_PLAN.md) and [ADR-0004](../../adr/0004-reinstate-vanilla-sft-control.md).
Score: {overall["successes"]}/400 ({100 * overall["accuracy"]:.2f}%); {kept["successes"]}/{kept["count"]}
({100 * kept["accuracy"]:.2f}%) excluding the eight train/test overlapping rows. Hardware: {manifest["hardware"]}.

| File | Contents |
| --- | --- |
| evaluation-summary.json | Overall, per-grid and overlap-excluded reference success counts |
| evaluation-outcomes.jsonl | All 400 generated answers, extracted routes and outcomes in original test order; row_index is zero-based |
| {steps} | Optimizer-step logs in the LIVR format |
| provenance.json | Recipe, evaluator/data hashes, hardware, GPU seconds, completed checks, runtime source hashes and evidence checksums |

As in the LIVR bundle, raw responses omit the repeated prompt prefix, and no weights,
optimizer states, images or credentials are included. Regenerate the comparison chart with
`python scripts/plot_sft_comparison.py --sft-a <arm A bundle> --sft-c <arm C bundle>`.
""")
    print(json.dumps({"output": str(output), "summary": summary["overall"],
                      "excluding_overlap": summary["excluding_overlap"]}, indent=2))


if __name__ == "__main__":
    main()
