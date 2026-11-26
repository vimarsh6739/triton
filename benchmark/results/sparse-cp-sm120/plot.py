"""Plot generated and hand-coded VJPs at the largest tested observation count."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    directory = Path(__file__).parent
    with (directory / "timings.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    n = max(int(row["observations"]) for row in rows)
    groups = defaultdict(list)
    for row in rows:
        if int(row["observations"]) == n:
            key = row["loss"], int(row["rank"]), bool(int(row["hot_row"])), row["method"]
            groups[key].append(float(row["us"]))

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True, layout="constrained")
    ranks = sorted({rank for _, rank, _, _ in groups})
    for ax, loss in zip(axes, ["squared", "huber"]):
        categories = [(rank, hot) for rank in ranks for hot in (False, True)]
        for i, method in enumerate(("generated", "hand")):
            quantiles = np.array([np.percentile(groups[loss, rank, hot, method], [25, 50, 75])
                                  for rank, hot in categories]).T
            low, med, high = quantiles
            ax.bar(np.arange(len(med)) + (i - .5) * .36, med, width=.36, label=method,
                   yerr=[med-low, high-med], capsize=3)
        ax.set_xticks(range(len(categories)),
                      [f"Rank {r}\n{'hot row' if hot else 'uniform'}" for r, hot in categories])
        ax.set_title(f"{loss.title()} loss" if loss == "huber" else "Squared loss")
        ax.spines[["right", "top"]].set_visible(False)
    axes[0].set_ylabel("Combined primal and VJP (µs)")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Sparse CP at {n:,} observations")
    fig.savefig(directory / "sparse-cp.png", dpi=180)


if __name__ == "__main__":
    main()
