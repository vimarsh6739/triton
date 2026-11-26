"""Compare source size and full GPU training calls for HDRNet bilateral slicing."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import textwrap

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parent
INK = "#192d37"
GRAY = "#54656d"
COLORS = {"torch": "#5b718f", "cuda": "#a35432", "ttir": "#087563"}
FILES = {
    "torch": ["torch_impl.py"],
    "fused_cuda": ["cuda_fused/kernel.cu", "cuda_fused/__init__.py"],
    "warp_cuda": ["cuda_warp/kernel.cu", "cuda_warp/__init__.py"],
    "ttir": ["triton_ad.py"],
}
NAMES = {"torch": "PyTorch eager", "compiled": "PyTorch compiled", "cuda_strict": "HDRNet CUDA¹",
         "fused_cuda": "CUDA, one thread per pixel", "warp_cuda": "CUDA, warp-cooperative", "ttir": "Triton + TTIR AD"}
ORDER = list(NAMES)


def sources(strict):
    groups = {k: [ROOT / p for p in paths] for k, paths in FILES.items()}
    groups["cuda_strict"] = [strict / p for p in ("hdrnet_kernels.cuh", "launch.cu", "__init__.py")]
    cloc = subprocess.check_output(["cloc", "--csv", "--by-file", "--quiet",
                                    *[str(p) for paths in groups.values() for p in paths]], text=True)
    counts = {r["filename"]: int(r["code"]) for r in csv.DictReader(cloc.splitlines()) if r["language"] != "SUM"}
    programs = {}
    for method, paths in groups.items():
        code = []
        for path in paths:
            # The counted modules use only single-line comments and one-line Python docstrings.
            prefix = ("#", '"""') if path.suffix == ".py" else ("//",)
            lines = [line for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith(prefix)]
            if len(lines) != counts[str(path)]:
                raise ValueError(f"Source listing for {path} disagrees with cloc")
            code.extend(lines)
        programs[method] = code
    compile_call = next(l.strip() for l in (ROOT / "bench.py").read_text().splitlines() if " = torch.compile(" in l)
    programs["compiled"] = programs["torch"] + [compile_call]
    return programs


def medians(path):
    seeds = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            seeds[row["method"], int(row["height"]), int(row["seed"])].append(float(row["ms"]))
    result = {}
    for m in ORDER:
        for h in (1024, 2048):
            runs = [statistics.median(seeds[m, h, s]) for s in (17, 43)]
            result[m, h] = statistics.mean(runs)
    return result


def color_of(method):
    if method in ("torch", "compiled"):
        return COLORS["torch"]
    return COLORS["ttir" if method == "ttir" else "cuda"]


def source_panel(fig, box, name, program, caption, shade):
    ax = fig.add_axes(box)
    ax.set_axis_off()
    ax.text(0, 1, name, color=shade, size=16, weight="bold", va="top", transform=ax.transAxes)
    ax.text(0, .83, f"{len(program)}", size=31, color=INK, weight="bold", va="top", transform=ax.transAxes)
    ax.text(.265, .745, "source lines", size=10.8, color=GRAY, transform=ax.transAxes)
    # File order is preserved. Each stroke is one counted line, grouped into columns of 24.
    segments = []
    for i, line in enumerate(program):
        col, row = divmod(i, 24)
        x, y = col * .131, .60 - row * .016
        indent = len(line) - len(line.lstrip())
        segments.append(((x + indent * .00072, y), (x + min(len(line), 125) * .00089 + .010, y)))
    ax.add_collection(LineCollection(segments, colors=shade, linewidths=.74, transform=ax.transAxes))
    ax.text(0, .09, caption, size=10, color=INK, va="top", transform=ax.transAxes)


def first_page(programs, times):
    fig = plt.figure(figsize=(13.8, 9.1), facecolor="white")
    fig.text(.043, .948, "Differentiating HDRNet’s bilateral color transform", fontsize=23, weight="bold", color=INK)
    fig.text(.044, .900, "Pretrained Local Laplacian model   ·   RGB output and gradients for the grid, guide and input photo",
             fontsize=12, color=GRAY)
    for x, key, title, desc in [
        (.044, "compiled", "PyTorch", "Native 3D grid sampling and autograd.\nThe count includes the compile call."),
        (.374, "warp_cuda", "Handwritten CUDA", "16 lanes share a pixel. The backward\ncomputes and accumulates its gradients."),
        (.704, "ttir", "Triton + Enzyme", "The author writes the forward stencil.\nEnzyme differentiates its Triton IR."),
    ]:
        source_panel(fig, (x, .618, .29, .260), title, programs[key], desc, color_of(key))
        if key == "compiled":
            excerpt = [next(line.strip() for line in programs[key] if " = F.grid_sample(" in line)]
        elif key == "warp_cuda":
            lines = programs[key]
            first = next(i for i, line in enumerate(lines) if "atomicAdd(dgrid" in line)
            excerpt = [line.strip() for line in lines[first:first + 3]]
        else:
            lines = programs[key]
            first = next(i for i, line in enumerate(lines) if "autodiff(_slice)" in line)
            excerpt = [line.strip() for line in lines[first:first + 3]]
        display = []
        for line in excerpt:
            display.extend(textwrap.wrap(line, width=52, subsequent_indent="    ", break_long_words=False))
        fig.text(x, .579, "\n".join(display), fontsize=7.9, color=INK, va="top",
                 family="DejaVu Sans Mono", linespacing=1.2)
    fig.text(.044, .470, "Each stroke is one counted line. Excerpts show the sampling call, gradient updates and differentiation request. Full listings are in the PDF.",
             fontsize=9, color=GRAY)
    fig.text(.044, .425, "Forward + backward", fontsize=16, weight="bold", color=INK)
    fig.text(.044, .393, "GPU milliseconds per call  ·  lower is faster", fontsize=10.3, color=GRAY)
    for i, (h, label) in enumerate([(1024, "1.05 MP"), (2048, "4.19 MP")]):
        ax = fig.add_axes((.272 + i * .37, .118, .286, .223))
        values = [times[m, h] for m in ORDER]
        ax.barh(range(len(ORDER)), values, height=.53, color=[color_of(m) for m in ORDER])
        ax.set_title(label + f"   ({h:,} × {h:,})", loc="left", fontsize=11, color=INK, weight="bold", pad=7)
        ax.set_xlim(0, max(values) * 1.23)
        ax.set_ylim(len(ORDER) - .45, -.55)
        ax.set_yticks(range(len(ORDER)), [NAMES[m] for m in ORDER] if i == 0 else [""] * len(ORDER))
        ax.tick_params(axis="y", labelcolor=INK, length=0, pad=9, labelsize=10)
        ax.tick_params(axis="x", labelsize=8, width=.5, length=3, colors=GRAY)
        for y, value in enumerate(values):
            ax.text(value + max(values) * .023, y, f"{value:.2f}", va="center", fontsize=10, color=INK)
        for spine in ("top", "left", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#b9c5c9")
    fig.text(.044, .045,
             "RTX 5090, FP32. Midpoints of two seed medians, 25 shuffled samples each.  ¹Published HDRNet CUDA with the depth-gradient boundary correction.",
             size=9.15, color=GRAY)
    return fig


def listing_pages(pdf, name, lines):
    # Split into equal-width columns on letter landscape with soft wrapping for long CUDA signatures.
    wrapped = []
    for number, line in enumerate(lines, 1):
        parts = textwrap.wrap(line, width=94, replace_whitespace=False, drop_whitespace=False,
                              subsequent_indent="    ", break_on_hyphens=False)
        for k, part in enumerate(parts):
            wrapped.append((str(number) if k == 0 else "", part))
    for start in range(0, len(wrapped), 94):
        fig = plt.figure(figsize=(15.5, 8.3), facecolor="white")
        fig.text(.035, .94, f"{name}  /  {len(lines)} counted source lines", size=17, weight="bold", color=INK)
        for i, (number, part) in enumerate(wrapped[start:start + 94]):
            col, row = divmod(i, 47)
            x, y = .046 + col * .49, .875 - row * .0176
            fig.text(x, y, number, ha="right", family="DejaVu Sans Mono", size=6.6, color=GRAY)
            fig.text(x + .006, y, part, family="DejaVu Sans Mono", size=7.35, color=INK)
        pdf.savefig(fig)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "results")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        strict = Path(temp) / "cuda_strict"
        subprocess.run([sys.executable, str(ROOT / "cuda/prepare_strict.py"), str(strict)], check=True)
        programs = sources(strict)
    times = medians(ROOT / "results/trained/timings.csv")
    mpl.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig = first_page(programs, times)
    fig.savefig(args.output / "bilateral.png", dpi=170)
    with PdfPages(args.output / "bilateral.pdf", metadata={"CreationDate": None}) as pdf:
        pdf.savefig(fig)
        for m in ["compiled", "warp_cuda", "ttir", "cuda_strict", "fused_cuda"]:
            listing_pages(pdf, NAMES[m], programs[m])
    plt.close(fig)
    with (args.output / "comparison.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["method", "source_lines", "1024_ms", "2048_ms"])
        for m in ORDER:
            writer.writerow([m, len(programs[m]), times[m, 1024], times[m, 2048]])


if __name__ == "__main__":
    main()
