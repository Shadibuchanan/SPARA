"""Monitor one submitted baseline and export immutable final checkpoints locally.

Run on the workstation. This never submits jobs or changes cluster files.
Copies are restartable at file boundaries; SHA-256 verifies every exported file.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time


def validate_run(destination, job, status, data_root):
    """Audit saved artifacts; no model inference or training occurs here."""
    from spara.data import records
    from spara.evaluation import ReferenceEvaluator, summarize

    run = destination / "results" / f"baseline-{job}"
    recipe = json.loads((Path(__file__).resolve().parents[2] / "configs/livr.json").read_text())
    source_snapshot = destination / "source-snapshot-sha256.json"
    source_hashes = json.loads(source_snapshot.read_text()) if source_snapshot.exists() else None
    checks = {}
    for stage in (1, 2):
        progress = json.loads((run / f"stage{stage}/final/complete.json").read_text())["progress"]
        assert not progress.get("pilot") and progress["stage"] == stage
        assert progress["epoch"] == 10 and progress["next_row"] == 0 and progress["optimizer_steps"] == 1250
        assert all(progress["config"][key] == value for key, value in recipe.items())
        steps = [json.loads(line) for line in (run / f"stage{stage}/steps.jsonl").read_text().splitlines()]
        assert [row["step"] for row in steps] == list(range(1, 1251))
        assert all(math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"]) for row in steps)
        assert steps[-1]["lr"] == 0
        checks[f"stage{stage}"] = {"epochs": 10, "updates": len(steps), "finite_losses_and_gradients": True}
    manifests = [run / f"stage{i}-manifest.json" for i in (1, 2)] + [run / "evaluation/manifest.json"]
    for path in manifests:
        manifest = json.loads(path.read_text())
        assert all(manifest["config"][key] == value for key, value in recipe.items())
        assert str(manifest["slurm_job_id"]) == str(job)
        if source_hashes is not None:
            assert all(source_hashes.get(key) == value for key, value in manifest["source_sha256"].items())
    checks["source_snapshot_matches_manifests"] = source_hashes is not None
    rows = records(data_root, "test")
    outputs = [json.loads(line) for line in (run / "evaluation/responses.jsonl").read_text().splitlines()]
    assert len(rows) == len(outputs) == 400
    assert [result["row_index"] for result in outputs] == list(range(400))
    evaluator = ReferenceEvaluator(data_root)
    for row, result in zip(rows, outputs):
        rescored = evaluator.score(row, result["raw_response"])
        assert all(result[key] == value for key, value in rescored.items())
    summary = summarize(outputs)
    assert summary == json.loads((run / "evaluation/summary.json").read_text())
    accounting = status["accounting"].strip().splitlines()[0].split("|")
    assert accounting[1] == "COMPLETED" and accounting[4] == "0:0"
    elapsed = int(accounting[2])
    assert elapsed <= 14400 and "gres/gpu=1" in accounting[3].split(",")
    checks.update(reference_rescoring_matches=True, complete_test_set=True,
                  elapsed_gpu_seconds=elapsed, approved_gpu_seconds=14400, summary=summary)
    (destination / "run-validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    score = summary["overall"]["accuracy"] * 100
    lines = ["# LIVR full-run result", "", f"PACE job {job} completed both ten-epoch stages (1,250 updates each).",
             f"Reference success: **{summary['overall']['successes']}/400 ({score:.2f}%)**.",
             f"Difference from the published 3B result (66%): **{score-66:+.2f} percentage points**.", "",
             "| Grid size | Successes | Total | Accuracy |", "| --- | --- | --- | --- |"]
    for grid, result in summary["per_grid"].items():
        lines.append(f"| {grid} | {result['successes']} | {result['count']} | {result['accuracy']*100:.2f}% |")
    lines += ["", f"Allocated GPU time: {elapsed} seconds ({elapsed/3600:.3f} hours), within the four-hour ceiling.",
              "Both final resumable checkpoints and run evidence were copied locally and SHA-256 verified.",
              "All 400 saved responses were independently rescored using the pinned reference evaluator.",
              "This is one seeded run; test results were not used to select a checkpoint or tune the recipe.",
              "See docs/LIVR_IMPLEMENTATION_SPEC.md and docs/LIVR_PILOT_REPORT.md for assumptions and pilot evidence.", ""]
    (destination / "RUN_REPORT.md").write_text("\n".join(lines))
    print("Full-run validation passed:", summary, flush=True)


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, type=int)
    parser.add_argument("--target", default=os.environ.get("SPARA_SSH_TARGET", "pace-ice"))
    parser.add_argument("--root", default=os.environ.get("SPARA_REMOTE_ROOT", "~/scratch/SPARA"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[2] / "data/vsp"))
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    def remote(code):
        command = "python3 -c " + shlex.quote(code)
        return subprocess.check_output(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.target, command],
            text=True,
        )

    prefix = "import pathlib,json,subprocess,hashlib; root=pathlib.Path(" + repr(args.root) + ").expanduser().resolve(); "
    root = remote(prefix + "print(root)").strip()
    run = root + f"/results/baseline-{args.job}"
    verified = {}
    record = destination / "backup-verification.json"
    if record.exists():
        previous = json.loads(record.read_text())
        if previous["remote_run"] != run or previous["target"] != args.target:
            raise ValueError("Output belongs to another run or SSH target")

    def export(relative):
        source = root + "/" + relative
        inventory = json.loads(remote("""import pathlib,json,hashlib
p=pathlib.Path(SOURCE)
files=[p] if p.is_file() else sorted(f for f in p.rglob('*') if f.is_file())
result=[]
for f in files:
    h=hashlib.sha256()
    with f.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): h.update(block)
    result.append({'path':str(f.relative_to(pathlib.Path(ROOT))), 'bytes':f.stat().st_size,'sha256':h.hexdigest()})
print(json.dumps(result))
""".replace("SOURCE", repr(source)).replace("ROOT", repr(root))))
        for item in inventory:
            output = (destination / item["path"]).resolve()
            if not output.is_relative_to(destination):
                raise ValueError("Unsafe remote filename")
            output.parent.mkdir(parents=True, exist_ok=True)
            expected = item["sha256"]
            if not (output.is_file() and output.stat().st_size == item["bytes"] and digest(output) == expected):
                partial = output.with_name(output.name + ".download-partial")
                print("Copying", item["path"], item["bytes"], "bytes", flush=True)
                subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                                args.target + ":" + root + "/" + item["path"], str(partial)], check=True)
                if partial.stat().st_size != item["bytes"] or digest(partial) != expected:
                    raise ValueError("Backup checksum mismatch: " + item["path"])
                partial.replace(output)
            verified[item["path"]] = item
            record.write_text(json.dumps({"target": args.target, "remote_run": run,
                                          "files": verified}, indent=2) + "\n")
        print("Verified export:", relative, flush=True)

    completed_stages = set()
    while True:
        status = json.loads(remote(prefix + """
run=pathlib.Path(RUN)
accounting=subprocess.check_output(['sacct','-n','-P','-X','-j',JOB,'--format=JobID,State,ElapsedRaw,AllocTRES,ExitCode'],text=True)
stages={str(i):(run/f'stage{i}/final/complete.json').is_file() for i in (1,2)}
progress={}
for i in (1,2):
    p=run/f'stage{i}/steps.jsonl'
    if p.exists():
        lines=p.read_text().splitlines()
        if lines: progress[str(i)]=json.loads(lines[-1])
p=run/'evaluation/summary.json'
print(json.dumps({'accounting':accounting,'stages':stages,'progress':progress,'evaluation':json.loads(p.read_text()) if p.exists() else None}))
""".replace("RUN", repr(run)).replace("JOB", repr(str(args.job)))))
        (destination / "latest-status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status), flush=True)
        for stage, complete in status["stages"].items():
            if complete and stage not in completed_stages:
                export(f"results/baseline-{args.job}/stage{stage}/final")
                completed_stages.add(stage)
        lines = status["accounting"].strip().splitlines()
        state = lines[0].split("|")[1].split()[0] if lines else "UNKNOWN"
        if state in {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}:
            # All writers have stopped. Preserve logs and evidence, including failed runs.
            export(f"results/baseline-{args.job}")
            export(f"logs/baseline-{args.job}.log")
            export("logs/baseline-budget.json")
            if state == "COMPLETED":
                validate_run(destination, args.job, status, args.data_root)
            print("Finished collection; Slurm state:", state, flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
