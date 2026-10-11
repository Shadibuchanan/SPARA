"""Render the committed baseline evidence without data downloads or model loading."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=project / "docs/experiments/livr-6189457")
    parser.add_argument("--output", type=Path, default=project / "docs/figures")
    args = parser.parse_args()
    provenance = json.loads((args.evidence / "provenance.json").read_text())
    for name, expected in provenance["artifact_sha256"].items():
        if hashlib.sha256((args.evidence / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Evidence hash mismatch: {name}")
    summary = json.loads((args.evidence / "evaluation-summary.json").read_text())
    outcomes = [json.loads(line) for line in (args.evidence / "evaluation-outcomes.jsonl").read_text().splitlines()]
    assert [row["row_index"] for row in outcomes] == list(range(400))
    assert sum(row["success"] for row in outcomes) == summary["overall"]["successes"]
    stages = [[json.loads(line) for line in (args.evidence / f"stage{i}-steps.jsonl").read_text().splitlines()]
              for i in (1, 2)]
    for index, rows in enumerate(stages, 1):
        assert len(rows) == 1250 and [r["step"] for r in rows] == list(range(1, 1251))
        assert all(r["stage"] == index for r in rows)
    assert sum(r["successes"] for r in summary["per_grid"].values()) == summary["overall"]["successes"]
    assert sum(r["count"] for r in summary["per_grid"].values()) == summary["overall"]["count"]
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#334155",
                         "text.color": "#0f172a", "axes.edgecolor": "#94a3b8",
                         "svg.fonttype": "none", "svg.hashsalt": "spara-livr-6189457"})

    def save(fig, name):
        fig.savefig(args.output / f"{name}.png", dpi=180, facecolor="white")
        vector = args.output / f"{name}.svg"
        fig.savefig(vector, metadata={"Date": None}, facecolor="white")
        # Matplotlib emits trailing spaces in SVG path data; normalize for Git.
        vector.write_text("\n".join(line.rstrip() for line in vector.read_text(encoding="utf-8").splitlines()) + "\n",
                          encoding="utf-8", newline="\n")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.5), gridspec_kw={"width_ratios": [1, 1.6]})
    fig.subplots_adjust(left=.075, right=.98, top=.78, bottom=.22, wspace=.26)
    fig.suptitle("LIVR planning baseline: evaluation", x=.075, y=.96, ha="left", fontsize=19, fontweight="bold")
    overall = summary["overall"]
    fig.text(.075, .87, f"{overall['successes']}/{overall['count']} successful routes  |  Final Stage 2 checkpoint  |  Seed 42", color="#475569")
    scores = [provenance["reference_score"]["accuracy"] * 100, overall["accuracy"] * 100]
    axes[0].bar([0, 1], scores, width=.58, color=["#94a3b8", "#087f8c"])
    axes[0].set_xticks([0, 1], ["Published LIVR\n3B result", "Our LIVR\nreimplementation"])
    axes[0].set_title("Overall comparison", loc="left", pad=15)
    for i, score in enumerate(scores):
        axes[0].text(i, score + 2, f"{score:g}%", ha="center", fontweight="bold")
    grids = sorted(summary["per_grid"], key=int)
    values = [summary["per_grid"][grid]["accuracy"] * 100 for grid in grids]
    axes[1].bar(range(len(grids)), values, width=.6, color="#087f8c")
    axes[1].set_xticks(range(len(grids)), [f"{grid} × {grid}" for grid in grids])
    axes[1].set_title("Our result by grid size", loc="left", pad=15)
    for i, (grid, score) in enumerate(zip(grids, values)):
        row = summary["per_grid"][grid]
        axes[1].text(i, score + 2, f"{row['successes']}/{row['count']}", ha="center", fontweight="bold")
    for ax in axes:
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Reference route success")
    fig.text(.075, .065, "Single seeded run. The published protocol is not fully recovered; this is not a controlled improvement claim.", fontsize=9, color="#475569")
    fig.text(.075, .025, "Sources: PACE job 6189457; LIVR v2 §4.2. Original 400-example Mirage split retained, including known overlap.", fontsize=9, color="#475569")
    save(fig, "livr-evaluation")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.1), sharey=True)
    fig.subplots_adjust(left=.085, right=.98, top=.76, bottom=.2, wspace=.17)
    fig.suptitle("LIVR planning baseline: training loss", x=.085, y=.96, ha="left", fontsize=19, fontweight="bold")
    fig.text(.085, .865, "Ten epochs per stage  |  1,250 updates per stage  |  Effective batch size 8", color="#475569")
    for ax, rows, stage in zip(axes, stages, (1, 2)):
        by_epoch = defaultdict(list)
        for row in rows:
            by_epoch[row["epoch"] + 1].append(row["loss"])
        assert sorted(by_epoch) == list(range(1, 11)) and all(len(v) == 125 for v in by_epoch.values())
        means = [sum(by_epoch[e]) / len(by_epoch[e]) for e in range(1, 11)]
        ax.plot([r["step"] / 125 for r in rows], [r["loss"] for r in rows], color="#087f8c", alpha=.18, linewidth=.65, label="Optimizer-step mean")
        ax.plot(range(1, 11), means, color="#075985", linewidth=2.2, marker="o", markersize=4, label="Epoch mean")
        ax.set_title(f"Stage {stage}: " + ("visual bottleneck" if stage == 1 else "ordinary attention"), loc="left", pad=15)
        ax.set_xlabel("Epoch within stage")
        ax.set_xlim(0, 10.3)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_yscale("log")
        ax.grid(axis="y", which="major", alpha=.18)
    axes[0].set_ylabel("Answer-token training loss (log scale)")
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.text(.085, .055, "Training loss measures fit to training answers; it is not validation accuracy or evidence of test generalization.", fontsize=9, color="#475569")
    save(fig, "livr-training-loss")
    print(f"Rendered PNG and SVG charts in {args.output}")


if __name__ == "__main__":
    main()
