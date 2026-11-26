"""Plot full and backward-only medians at the Liger benchmark's row widths."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    folder = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=folder)
    parser.add_argument("--output", type=Path, default=folder / "groupnorm.png")
    args = parser.parse_args()
    groups = defaultdict(list)
    for file in args.samples.glob("*-seed*.csv"):
        scope = file.name.split("-")[0]
        with file.open() as stream:
            for row in csv.DictReader(stream):
                if row["channels"] == "4096" and row["dtype"] == "torch.bfloat16":
                    groups[scope, int(row["batch"]), row["method"], int(row["seed"])].append(float(row["us"]))
    methods = [("liger", "Liger"), ("generated", "Pointer VJP"), ("values", "Tile VJP"),
               ("scheduled", "Tiled manual"), ("torch", "PyTorch")]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), layout="constrained")
    for ax, batch in zip(axes, [2, 8]):
        for shift, scope, color in [(-0.18, "full", "#2d5f88"), (0.18, "backward", "#d78436")]:
            x = [i + shift for i in range(len(methods))]
            times = []
            for method, _ in methods:
                runs = [median(v) for (s, n, m, _), v in groups.items() if (s, n, m) == (scope, batch, method)]
                times.append(median(runs))
            ax.bar(x, times, 0.35, label="Forward + backward" if scope == "full" else "Backward only", color=color)
        ax.set_xticks(range(len(methods)), [label for _, label in methods], rotation=25, ha="right")
        ax.set_title(f"Batch {batch} · BF16")
        ax.set_ylabel("CUDA graph latency (µs)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.savefig(args.output, dpi=170)


if __name__ == "__main__":
    main()
