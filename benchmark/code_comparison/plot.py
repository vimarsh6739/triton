"""Render the counted source beside the runtime of the same two implementations."""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from source_lines import read_source, statement_count


ROOT = Path(__file__).resolve().parent
WIDTH = 15.5
INK, MUTED = "#162d39", "#48616d"
AD, HAND = "#006b62", "#8d4928"
AD_BG, HAND_BG = "#e9f5ef", "#fcf1e9"
METHODS = {"generated": "TTIR AD", "handwritten": "Handwritten Triton", "liger": "Upstream Liger"}


@dataclass(frozen=True)
class Example:
    directory: str
    title: str
    subtitle: str
    shared: str
    handwritten: str
    common_description: str
    scopes: tuple
    sizes: tuple
    size_label: str
    timing_note: str


EXAMPLES = [
    Example("movielens", "Three factor gradients from one MovieLens loss",
            "Rank-16 recommender  ·  user × film × week  ·  squared loss on MovieLens 100K ratings",
            "shared.py", "handwritten.py",
            "Both versions call the same loss and launch helper. Every additional line for each version appears below.",
            (("kernel-timings.csv", "Fused loss + gradients", None), ("timings.csv", "With batch preparation", None)),
            (128, 2048), "ratings",
            "Batch preparation gathers ratings and indices, clears gradients and sets the seed. Optimizer updates are outside both measurements."),
    Example("groupnorm", "One GroupNorm forward, two backward implementations",
            "Liger benchmark shape  ·  4,096 channels × 512 positions  ·  four channels per group  ·  BF16",
            "common.py", "manual.py",
            "Both versions share the forward kernel and PyTorch autograd wrapper. Every additional line appears below.",
            (("timings.csv", "Forward + backward", None),), (2, 8), "batch",
            "The hand-tiled backward reuses saved statistics. The generated pointer pullback recomputes them. Upstream Liger supports additional shapes."),
    Example("adamw", "Forward and reverse AD for one AdamW update",
            "Fused optimizer update  ·  parameter, minibatch gradient and two moments  ·  FP32",
            "shared.py", "handwritten.py",
            "Both versions provide the primal update and both derivative modes. Every additional line appears below.",
            (("timings.csv", "Update + forward derivative", "jvp"), ("timings.csv", "Update + reverse derivative", "vjp")),
            (4194307,), "elements",
            "Each timing covers one mode, including updated optimizer state. Both modes contribute to the source totals."),
]


def put(fig, x, y, string, *, size=10, color=INK, **kwargs):
    fig.text(x / WIDTH, 1 - y / fig.get_figheight(), string, fontsize=size,
             va="top", color=color, **kwargs)


def block(fig, x, y, width, height, color):
    h = fig.get_figheight()
    fig.add_artist(Rectangle((x / WIDTH, 1 - (y + height) / h), width / WIDTH,
                             height / h, transform=fig.transFigure, color=color, linewidth=0, zorder=0))


def listing(path):
    """Lay out physical source lines, keeping their file line numbers when wrapped."""
    rows = []
    previous = 0
    for number, line in read_source(path):
        if previous and number - previous > 1:
            rows.append(("", ""))
        previous = number
        while len(line) > 90:
            split = line.rfind(" ", 35, 90)
            split = split if split > 0 else 90
            rows.append((str(number) if number else "", line[:split]))
            line = "    " + line[split:].lstrip()
            number = None
        rows.append((str(number) if number else "", line))
    return rows


def print_listing(fig, x, y, rows, color):
    for i, (number, line) in enumerate(rows):
        position = y + 0.151 * i
        put(fig, x, position, number, size=7.2, color=MUTED, family="DejaVu Sans Mono", ha="right")
        put(fig, x + 0.11, position - 0.009, line, size=8.45, color=color, family="DejaVu Sans Mono")


def read_timings(example):
    """Summarize seed medians; keep independent input seeds visible in the CSV."""
    summaries = []
    for file, scope, mode in example.scopes:
        groups = defaultdict(list)
        with (ROOT / example.directory / file).open() as handle:
            for row in csv.DictReader(handle):
                if mode and row["mode"] != mode:
                    continue
                batch = int(row["n"] if example.size_label == "elements" else row["batch"])
                if batch in example.sizes:
                    groups[batch, row["method"], int(row["seed"])].append(float(row["us"]))
        for (batch, method, seed), samples in sorted(groups.items()):
            summaries.append(dict(scope=scope, batch=batch, method=method, seed=seed,
                                  samples=len(samples), median_us=median(samples)))
    return summaries


def timing_table(fig, y, example, summaries):
    put(fig, 0.40, y, "GPU time per call", size=16, weight="bold")
    put(fig, 0.40, y + 0.32, "Microseconds  ·  median of two seed medians  ·  CUDA graphs on RTX 5090", color=MUTED)
    columns = ["generated", "handwritten"]
    if example.directory == "groupnorm":
        columns.append("liger")
    x_values = [7.7, 10.9, 14.75]
    y += 0.72
    for x, method in zip(x_values, columns):
        put(fig, x, y, METHODS[method], size=10, weight="bold", ha="right")
    y += 0.31
    for _, scope, _ in example.scopes:
        for batch in example.sizes:
            label = f"Batch {batch}" if example.size_label == "batch" else f"{batch:,} {example.size_label}"
            put(fig, 0.4, y, scope, size=11)
            put(fig, 3.9, y, label, size=11, color=MUTED)
            for x, method in zip(x_values, columns):
                rows = [r for r in summaries if (r["scope"], r["batch"], r["method"]) == (scope, batch, method)]
                if len(rows) != 2 or len({r["samples"] for r in rows}) != 1:
                    raise ValueError(f"Expected equally sampled results for two seeds: {scope}, {batch}, {method}")
                center = median(r["median_us"] for r in rows)
                put(fig, x, y, f"{center:.2f}", size=12, weight="bold", ha="right")
            y += 0.28
    put(fig, 0.4, y + 0.12, example.timing_note, size=9.7, color=MUTED)


def draw(example, output):
    base = ROOT / example.directory
    common = base / example.shared
    methods = [("generated", base / "generated.py", AD, AD_BG),
               ("handwritten", base / example.handwritten, HAND, HAND_BG)]
    shared_count = len(read_source(common))
    summaries = read_timings(example)
    code_top = 2.12
    longest = max(len(listing(path)) for _, path, _, _ in methods)
    timing_top = code_top + 0.151 * longest + 0.32
    height = timing_top + 1.68 + 0.28 * len(example.scopes) * len(example.sizes)
    fig = plt.figure(figsize=(WIDTH, height), facecolor="white")
    put(fig, 0.4, 0.24, example.title, size=25, weight="bold")
    put(fig, 0.41, 0.76, example.subtitle, size=12, color=MUTED)
    put(fig, 0.41, 1.10, example.common_description, size=10.2)
    counts = []
    for x, (method, path, color, bg) in zip((0.40, 7.91), methods):
        variant = len(read_source(path))
        total = shared_count + variant
        counts.append(dict(example=example.directory, method=method, shared_lines=shared_count,
                           method_lines=variant, total_lines=total,
                           total_statements=statement_count(common) + statement_count(path)))
        put(fig, x, 1.49, METHODS[method], size=16, weight="bold", color=color)
        put(fig, x, 1.82, f"{total} source lines   ·   {shared_count} shared + {variant} method-specific", size=11)
        rows = listing(path)
        block(fig, x, code_top, 7.18, len(rows) * 0.151 + 0.14, bg)
        print_listing(fig, x + 0.32, code_top + 0.11, rows, INK)
    timing_table(fig, timing_top, example, summaries)
    fig.savefig(output / f"{example.directory}.png", dpi=160)
    with PdfPages(output / f"{example.directory}.pdf", metadata={"CreationDate": None}) as pdf:
        pdf.savefig(fig)
        # Print the shared source once, at the same type size as the method-specific code.
        rows = listing(common)
        split = (len(rows) + 1) // 2
        appendix = plt.figure(figsize=(WIDTH, max(4, 1.95 + 0.151 * split)), facecolor="white")
        put(appendix, 0.4, 0.28, "Source counted on both sides", size=22, weight="bold")
        put(appendix, 0.4, 0.83, f"{example.title}   ·   {example.shared}   ·   {shared_count} source lines",
            size=12, color=MUTED)
        print_listing(appendix, 0.72, 1.46, rows[:split], INK)
        print_listing(appendix, 8.23, 1.46, rows[split:], INK)
        pdf.savefig(appendix)
        plt.close(appendix)
    plt.close(fig)
    return counts, [dict(example=example.directory, **row) for row in summaries]


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    mpl.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    counts, timings = [], []
    for example in EXAMPLES:
        sources, medians = draw(example, args.output)
        counts.extend(sources)
        timings.extend(medians)
    write_csv(args.output / "source-counts.csv", counts)
    write_csv(args.output / "seed-medians.csv", timings)


if __name__ == "__main__":
    main()
