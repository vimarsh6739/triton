"""Summarize CUDA event samples and plot AdamW at a fixed launch configuration."""

import argparse
import csv
import gzip
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def summarize(folder):
    rows = []
    for path in sorted(folder.glob("*.csv.gz")):
        groups = defaultdict(list)
        with gzip.open(path, "rt", newline="") as f:
            for raw in csv.DictReader(f):
                groups[tuple(raw[key] for key in ("n", "mode", "implementation", "block", "warps"))].append(raw)
        for (n, mode, impl, block, warps), group in groups.items():
            times = [float(r["us"]) for r in group]
            qs = statistics.quantiles(times, n=4, method="inclusive") if len(times) > 1 else [times[0]] * 3
            rows.append(dict(run=path.name.split(".")[0], n=int(n), mode=mode, impl=impl, block=int(block),
                             warps=int(warps), samples=len(times), median_us=statistics.median(times),
                             q25_us=qs[0], q75_us=qs[2], atomics=int(group[0]["ttir_atomics"]),
                             registers=int(group[0]["registers"]), max_derivative_abs=float(group[0]["max_derivative_abs"])))
    return sorted(rows, key=lambda r: (r["run"], r["n"], r["mode"], r["impl"], r["block"], r["warps"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    rows = summarize(args.directory)
    if not rows:
        parser.error("no timing CSVs found")
    with (args.directory / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fig, (full, projected) = plt.subplots(1, 2, figsize=(9, 4.25), layout="constrained")
    config = [r for r in rows if r["block"] == 256 and r["warps"] == 4]
    for ax, run, series in [
        (full, "sweep", [("jvp", "generated", "Enzyme JVP", "o", "-"), ("jvp", "hand", "Hand JVP", "x", "--"),
                         ("vjp", "generated", "Enzyme VJP", "s", "-"), ("vjp", "hand", "Hand VJP", "+", "--")]),
        (projected, "projection", [("vjp_g", "generated", "Activity-specialized", "o", "-"),
                                    ("vjp_g", "full_projected", "Full VJP projected to dg", "s", "--")])
    ]:
        for mode, impl, label, marker, line in series:
            points = [r for r in config if r["run"] == run and r["mode"] == mode and r["impl"] == impl]
            x = [r["n"] for r in points]
            ax.plot(x, [r["median_us"] for r in points], marker=marker, ls=line, label=label)
            ax.fill_between(x, [r["q25_us"] for r in points], [r["q75_us"] for r in points], alpha=.14)
        ax.set(xscale="log", yscale="log", xlabel="Parameter elements", ylabel="Median kernel time (µs)")
        ax.grid(axis="y", alpha=.25)
        ax.legend(fontsize=8)
    full.set_title("All optimizer state active")
    projected.set_title("Training-gradient pullback")
    fig.suptitle("AdamW on RTX 5090 · FP32 · block 256, four warps", fontsize=12)
    fig.savefig(args.directory / "adamw.png", dpi=200)


if __name__ == "__main__":
    main()
