"""Choose a launch size for the original CUDA kernels on a separate image shape."""

import argparse
import csv
from pathlib import Path
import random
from statistics import median
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuda import slice_apply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=11)
    args = parser.parse_args()
    torch.manual_seed(73)
    height, width = 768, 1024
    inputs = [torch.rand(shape, device="cuda").requires_grad_()
              for shape in ((1, 16, 16, 8, 3, 4), (1, height, width), (1, height, width, 3))]
    seed = torch.randn_like(inputs[-1])
    graphs, held, samples = {}, {}, []
    for threads in (64, 128, 256, 512, 1024):
        def call():
            value = slice_apply(*inputs, threads)
            return (value, *torch.autograd.grad(value, inputs, seed))
        try:
            call()
            torch.cuda.synchronize()
        except RuntimeError as exc:
            if "too many resources requested" not in str(exc):
                raise
            print(f"{threads} threads exceeds kernel resources")
            continue
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            held[threads] = call()
        graphs[threads] = graph
    order = list(graphs)
    rng = random.Random(73)
    begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for trial in range(-3, args.trials):
        rng.shuffle(order)
        for threads in order:
            begin.record()
            graphs[threads].replay()
            end.record()
            end.synchronize()
            if trial >= 0:
                samples.append((height, width, threads, trial, 1000 * begin.elapsed_time(end)))
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("height", "width", "threads", "trial", "us"))
        writer.writerows(samples)
    for threads in sorted(graphs):
        print(threads, median(s[-1] for s in samples if s[2] == threads))


if __name__ == "__main__":
    main()
