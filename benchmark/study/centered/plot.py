"""Plot held-out cotangent errors for two seed distributions."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    timings = read_csv(args.results / "timings.csv")
    shifted = [
        (r["schedule"], int(r["cols"]), float(r["common_mode_error"])) for r in timings
        if r["phase"] == "holdout" and r["mode"] == "ReverseModeCombined" and
        r["trial"] == "0" and r["warps"] == "4"]
    cases = [("All seeds shifted by 65,536", shifted)]
    schedules = [("tensor-control", "Uncentered", "#2176a8"), ("tensor-centered", "First column", "#b54825")]
    heldout = read_csv(args.results / "seed-regimes.csv")
    if any(r["schedule"] == "tensor-pivot" for r in heldout):
        schedules.append(("tensor-pivot", "Max-logit column", "#498444"))
    for regime, title in (("first-column-outlier", "Rare column has seed 2²⁴"),
                          ("max-logit-outlier", "Dominant column has seed 2²⁴")):
        outlier = [(r["schedule"], int(r["cols"]), float(r["max_abs_error"])) for r in heldout
                   if r["mode"] == "ReverseModeCombined" and r["regime"] == regime]
        if outlier:
            cases.append((title, outlier))
    fig, axes = plt.subplots(1, len(cases), figsize=(3.8 * len(cases), 3.5), sharey=True, layout="constrained")
    for ax, (title, data) in zip(axes, cases):
        for schedule, name, color in schedules:
            values = sorted((cols, error) for variant, cols, error in data if schedule == variant)
            ax.plot([str(n) for n, _ in values], [e for _, e in values], marker="o", color=color, label=name)
        ax.axhline(1e-5, color=".4", linewidth=0.8, linestyle="--", label="Calibration budget")
        ax.set_yscale("log")
        ax.set_ylim(1e-9, 10)
        ax.set_xlabel("Columns per row")
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Worst absolute VJP error vs. FP64")
    axes[0].legend(fontsize=8)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
