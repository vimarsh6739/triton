"""Select a fixed CUDA pixel schedule on an HDRNet photo at a held-out resolution."""

import argparse
import csv
from pathlib import Path
import random
import statistics
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import reference, verify
from inputs import make_inputs
from cuda_fused import slice_apply as scalar
from cuda_warp import slice_apply as cooperative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    tensors = make_inputs(args.image, 768, 1024, 101)
    inputs, seed = tensors[:3], tensors[3]
    truth = reference(*inputs, seed)
    candidates = [(1, 128)] + [(lanes, threads) for lanes in (4, 16) for threads in (64, 128, 256, 512)]
    graphs, results, samples = {}, {}, {key: [] for key in candidates}
    for key in candidates:
        lanes, threads = key
        def call():
            y = scalar(*inputs) if lanes == 1 else cooperative(*inputs, lanes=lanes, threads=threads)
            return (y.detach(), *torch.autograd.grad(y, inputs, seed))
        verify(call(), truth)
        graphs[key] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[key]):
            results[key] = call()
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    rng = random.Random(101)
    with args.output.open("x", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("lanes", "threads", "seed", "height", "width", "trial", "ms"))
        for trial in range(-5, 31):
            rng.shuffle(candidates)
            for key in candidates:
                start.record()
                graphs[key].replay()
                stop.record()
                stop.synchronize()
                if trial >= 0:
                    ms = start.elapsed_time(stop)
                    writer.writerow((*key, 101, 768, 1024, trial, ms))
                    samples[key].append(ms)
        for key in candidates:
            verify(results[key], truth)
    print(sorted((statistics.median(v), k) for k, v in samples.items()))


if __name__ == "__main__":
    main()
