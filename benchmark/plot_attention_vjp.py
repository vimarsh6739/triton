"""Plot median latency across seeds of the attention forward and VJP."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timing_csv", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    latencies = defaultdict(list)
    with args.timing_csv.open() as f:
        for row in csv.DictReader(f):
            latencies[row["implementation"], int(row["n"])].append(float(row["median_us"]))
    sizes = sorted({n for _, n in latencies})
    names = {"generated": "TTIR AD", "torch_flash": "PyTorch Flash SDPA", "torch_math": "PyTorch math SDPA"}
    fig, ax = plt.subplots(figsize=(6.5, 3.5), layout="constrained")
    for impl, label in names.items():
        data = [latencies[impl, n] for n in sizes]
        centers = [median(values) for values in data]
        ax.plot(sizes, centers, marker="o", label=label)
        ax.fill_between(sizes, [min(x) for x in data], [max(x) for x in data], alpha=.18)
    ax.set(xlabel="Sequence length", ylabel="Forward + Q, K, V pullback (µs)", xticks=sizes,
           title="Causal FP16 attention on RTX 5090, 2 batches × 8 heads × 64 features")
    ax.grid(axis="y", alpha=.25)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(args.output)


if __name__ == "__main__":
    main()
