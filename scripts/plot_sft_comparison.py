"""Compare committed LIVR and vanilla SFT evidence without data downloads or model loading."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # fixed categorical order: LIVR, Vanilla SFT Continuous (arm A), Vanilla SFT Staged (arm C)


def load(evidence, step_files):
    provenance = json.loads((evidence / "provenance.json").read_text())
    for name, expected in provenance["artifact_sha256"].items():
        if hashlib.sha256((evidence / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Evidence hash mismatch: {evidence.name}/{name}")
    outcomes = [json.loads(line) for line in (evidence / "evaluation-outcomes.jsonl").read_text().splitlines()]
    assert [row["row_index"] for row in outcomes] == list(range(400))
    steps = [json.loads(line) for name in step_files for line in (evidence / name).read_text().splitlines()]
    assert len(steps) == 2500
    return outcomes, steps


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-a", type=Path, required=True)
    parser.add_argument("--sft-c", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=project / "docs/figures")
    args = parser.parse_args()
    runs = [("LIVR (K=4, 10+10)", *load(project / "docs/experiments/livr-6189457", ["stage1-steps.jsonl", "stage2-steps.jsonl"])),
            ("Vanilla SFT Continuous (20 ep)", *load(args.sft_a, ["train-steps.jsonl"])),
            ("Vanilla SFT Staged (10+10 ep)", *load(args.sft_c, ["segment1-steps.jsonl", "segment2-steps.jsonl"]))]
    # Zero-based overlap rows recorded by the exporter; the test set is shared by all three runs.
    overlap, other = (set(json.loads((path / "evaluation-summary.json").read_text())["excluding_overlap"]["excluded_rows"])
                      for path in (args.sft_a, args.sft_c))
    assert overlap == other and len(overlap) == 8
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#334155",
                         "text.color": "#0f172a", "axes.edgecolor": "#94a3b8",
                         "svg.fonttype": "none", "svg.hashsalt": "spara-sft-control"})

    def save(fig, name):
        fig.savefig(args.output / f"{name}.png", dpi=180, facecolor="white")
        vector = args.output / f"{name}.svg"
        fig.savefig(vector, metadata={"Date": None}, facecolor="white")
        vector.write_text("\n".join(line.rstrip() for line in vector.read_text(encoding="utf-8").splitlines()) + "\n",
                          encoding="utf-8", newline="\n")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1, 1.5]})
    fig.subplots_adjust(left=.07, right=.98, top=.76, bottom=.22, wspace=.22)
    fig.suptitle("Vanilla SFT control vs LIVR: evaluation", x=.07, y=.96, ha="left", fontsize=19, fontweight="bold")
    fig.text(.07, .87, "Same split, recipe, seed, data order and evaluator  |  2,500 optimizer updates each  |  One seed per arm", color="#475569")
    width = .26
    for i, (label, outcomes, _) in enumerate(runs):
        kept = [o for o in outcomes if o["row_index"] not in overlap]
        scores = [100 * sum(o["success"] for o in rows) / len(rows) for rows in (outcomes, kept)]
        bars = axes[0].bar([x + (i - 1) * width for x in (0, 1)], scores, width=width - .02, color=COLORS[i], label=label)
        for bar, score in zip(bars, scores):
            axes[0].text(bar.get_x() + bar.get_width() / 2, score + 1.5, f"{score:.1f}", ha="center", fontsize=8.5)
        grids = sorted({o["grid_size"] for o in outcomes})
        per_grid = [100 * sum(o["success"] for o in outcomes if o["grid_size"] == g) / sum(o["grid_size"] == g for o in outcomes) for g in grids]
        axes[1].bar([x + (i - 1) * width for x in range(len(grids))], per_grid, width=width - .02, color=COLORS[i], label=label)
    axes[0].axhline(66, color="#64748b", linestyle="--", linewidth=1, label="LIVR published: 66%")
    axes[0].legend(handles=axes[0].get_lines(), loc="lower right", bbox_to_anchor=(1, 1.0), frameon=False,
                   fontsize=8.5)
    axes[0].set_xticks([0, 1], ["All 400 rows", "392 rows\n(overlap excluded)"])
    axes[0].set_title("Overall route success", loc="left", pad=15)
    axes[1].set_xticks(range(len(grids)), [f"{g} × {g}" for g in grids])
    axes[1].set_title("By grid size (400 rows)", loc="left", pad=15)
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    for ax in axes:
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    fig.text(.07, .065, "n=1 seed per arm; the binomial 95% interval at 80.75% on 400 rows is about ±3.9 points. Smaller differences are not read as real.", fontsize=9, color="#475569")
    fig.text(.07, .025, "Sources: LIVR job 6189457 evidence; vanilla SFT evidence bundles; LIVR v2 §4.2. LIVR ran on an H200, SFT on an H100.", fontsize=9, color="#475569")
    save(fig, "sft-control-evaluation")

    fig, ax = plt.subplots(figsize=(11.4, 5.1))
    fig.subplots_adjust(left=.085, right=.98, top=.76, bottom=.2)
    fig.suptitle("Vanilla SFT control vs LIVR: training loss", x=.085, y=.96, ha="left", fontsize=19, fontweight="bold")
    fig.text(.085, .865, "Epoch-mean answer-token loss over 20 epochs (125 updates each)  |  Dotted line: LIVR stage / Staged restart", color="#475569")
    for color, (label, _, steps) in zip(COLORS, runs):
        by_epoch = defaultdict(list)
        for index, row in enumerate(steps):
            by_epoch[index // 125 + 1].append(row["loss"])
        assert all(len(v) == 125 for v in by_epoch.values())
        means = [sum(by_epoch[e]) / 125 for e in range(1, 21)]
        ax.plot(range(1, 21), means, color=color, linewidth=2, marker="o", markersize=4, label=label)
    ax.axvline(10.5, color="#94a3b8", linestyle=":", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_xlim(0.5, 20.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_yscale("log")
    ax.set_ylabel("Training loss (log scale)")
    ax.grid(axis="y", alpha=.18)
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    fig.text(.085, .055, "Training loss measures fit to training answers; it is not validation accuracy.", fontsize=9, color="#475569")
    save(fig, "sft-control-training-loss")
    print(f"Rendered PNG and SVG charts in {args.output}")


if __name__ == "__main__":
    main()
