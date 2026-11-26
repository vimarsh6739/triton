"""Plot the recurrence and softmax studies from their individual timing samples."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def recurrence(source, output):
    groups = defaultdict(dict)
    for row in rows(source):
        key = row["schedule"], int(row["width"]), int(row["steps"])
        groups[key][int(row["sample"])] = float(row["us"])
    pairs = sorted({(d, t) for _, d, t in groups})
    fig, ax = plt.subplots(figsize=(7.5, 4.1), layout="constrained")
    for steps in sorted({t for _, t in pairs}):
        widths, centers, errors = [], [], []
        for d, t in pairs:
            if t != steps:
                continue
            post, both = groups["post", d, t], groups["both", d, t]
            samples = [us / both[i] for i, us in post.items()]
            lo, mid, hi = np.percentile(samples, [25, 50, 75])
            widths.append(d)
            centers.append(mid)
            errors.append((mid - lo, hi - mid))
        ax.errorbar(widths, centers, yerr=np.array(errors).T, label=f"{steps} steps",
                    marker="o", capsize=3)
    ax.axhline(1, color="0.55", linewidth=1)
    ax.set(xscale="log", xlabel="Feature width D", ylabel="Speedup over post-AD LICM (×)",
           title="Hoisting the projection before reverse AD")
    ticks = sorted({d for d, _ in pairs})
    ax.set_xticks(ticks, labels=[f"{d:,}" for d in ticks])
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output / "recurrent-projection.png", dpi=180)
    plt.close(fig)


def softmax(source, output):
    times = defaultdict(list)
    error = {}
    for row in rows(source):
        if row["phase"] != "holdout" or row["warps"] != "4":
            continue
        key = row["mode"], row["schedule"], int(row["cols"])
        times[key].append(float(row["us"]))
        error[key] = float(row["common_mode_error"])
    widths = sorted({w for _, _, w in times})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for mode, label, marker in [("ForwardMode", "JVP", "o"), ("ReverseModeCombined", "VJP", "s")]:
        speedup = [median(times[mode, "baseline", w]) / median(times[mode, "detach", w]) for w in widths]
        left.plot(widths, speedup, marker=marker, label=label)
        for schedule, style in [("baseline", ":"), ("detach", "-")]:
            right.plot(widths, [error[mode, schedule, w] for w in widths], marker=marker,
                       linestyle=style, label=f"{label}, {schedule}")
    left.axhline(1, color="0.55", linewidth=1)
    left.set(title="Runtime at four warps", ylabel="Speedup from detaching the shift (×)")
    right.set(title="Directional seed offset by 65,536", yscale="log",
              ylabel="Maximum absolute gradient error")
    for ax in (left, right):
        ax.set(xscale="log", xlabel="Softmax row width")
        ax.set_xticks(widths, labels=[f"{w:,}" for w in widths])
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(output / "softmax-shift.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--softmax", type=Path, required=True)
    parser.add_argument("--recurrence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    recurrence(args.recurrence, args.output)
    softmax(args.softmax, args.output)
