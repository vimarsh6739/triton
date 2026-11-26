"""Measure steady-state eager autograd, including Python dispatch and GPU work."""
import argparse
import csv
from pathlib import Path
import random
import time

import torch

from groupnorm_pullback import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--calls", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=15)
    args = parser.parse_args()
    if min(args.calls, args.rounds) < 1:
        parser.error("counts must be positive")
    with args.output.open("x", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["seed", "batch", "channels", "spatial", "dtype", "method", "round", "calls", "wall_us"])
        for seed in (51, 137):
            torch.manual_seed(seed)
            rng = random.Random(seed)
            for n in (2, 8):
                shape = (n, 4096, 512)
                x, w, b = [torch.randn(s, device="cuda", dtype=torch.bfloat16).requires_grad_()
                           for s in (shape, (4096,), (4096,))]
                dy = torch.randn_like(x)
                ref = evaluate(*(t.detach().double().requires_grad_() for t in (x, w, b)), dy.double(), 1024, "torch")
                methods = ["liger", "scheduled", "generated", "values", "torch"]
                for name in methods:
                    for _ in range(4):
                        values = evaluate(x, w, b, dy, 1024, name)
                    for val, expected in zip(values, ref):
                        torch.testing.assert_close(val.double(), expected, atol=.04 * expected.abs().max().item(), rtol=.03)
                for i in range(args.rounds):
                    rng.shuffle(methods)
                    for name in methods:
                        torch.cuda.synchronize()
                        start = time.perf_counter_ns()
                        for _ in range(args.calls):
                            values = evaluate(x, w, b, dy, 1024, name)
                        torch.cuda.synchronize()
                        elapsed = (time.perf_counter_ns() - start) / (1000 * args.calls)
                        writer.writerow([seed, n, 4096, 512, "bfloat16", name, i, args.calls, elapsed])
                f.flush()
    from groupnorm_values import value_kernel
    value_kernel.cache_clear()
