"""Time full PyTorch forward and backward calls against independent FP64 GroupNorm."""

import argparse
import csv
from pathlib import Path
import random

import torch
from liger_kernel.ops.group_norm import LigerGroupNormFunction

import generated
import manual


def liger(x, w, b, groups, eps):
    return LigerGroupNormFunction.apply(x, w, b, len(w), groups, eps)


def run(shape, k, dtype, seed_id, repeats, samples, errors):
    _, channels, _ = shape
    groups, eps = channels // k, 1e-6
    torch.manual_seed(seed_id)
    tensors = tuple(torch.randn(s, dtype=dtype, device="cuda").requires_grad_()
                    for s in (shape, (channels,), (channels,)))
    seed = torch.randn_like(tensors[0])
    original_seed = seed.clone()
    reference_inputs = tuple(t.detach().double().requires_grad_() for t in tensors)
    output = torch.nn.functional.group_norm(reference_inputs[0], groups, *reference_inputs[1:], eps=eps)
    reference = (output.detach(), *torch.autograd.grad(output, reference_inputs, seed.double()))
    implementations = {"handwritten": manual.groupnorm, "generated": generated.groupnorm, "liger": liger}
    held, graphs = {}, {}
    common = (seed_id, *shape, k, str(dtype).removeprefix("torch."))

    def evaluate(fn):
        y = fn(*tensors, groups, eps)
        return (y.detach(), *torch.autograd.grad(y, tensors, seed))

    def check(name, actual, phase):
        for field, y, ref in zip(("y", "dx", "dw", "db"), actual, reference):
            diff = y.double() - ref
            rel = (diff.norm() / ref.norm()).item()
            tol = 0.025 if dtype == torch.bfloat16 else 2.5e-5
            torch.testing.assert_close(y.double(), ref, atol=tol * ref.abs().max().item(), rtol=tol)
            if rel > tol:
                raise ValueError(f"{field} of {name} failed FP64 check ({rel})")
            errors.writerow((*common, name, phase, field, rel, diff.abs().max().item()))
        torch.testing.assert_close(seed, original_seed, rtol=0, atol=0)

    for name, fn in implementations.items():
        for _ in range(3):
            values = evaluate(fn)
        check(name, values, "warm")
        graphs[name] = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graphs[name]):
            held[name] = evaluate(fn)

    methods = list(implementations)
    rng = random.Random(seed_id)
    for trial in range(-5, repeats):
        rng.shuffle(methods)
        for name in methods:
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            stop.record()
            stop.synchronize()
            if trial >= 0:
                samples.writerow((*common, name, trial, 1000 * start.elapsed_time(stop)))
    for name, values in held.items():
        check(name, values, "replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=51)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "timings.csv").open("w", newline="") as raw, \
            (args.output / "accuracy.csv").open("w", newline="") as check:
        samples = csv.writer(raw, lineterminator="\n")
        errors = csv.writer(check, lineterminator="\n")
        fields = ("seed", "batch", "channels", "spatial", "channels_per_group", "dtype", "method")
        samples.writerow((*fields, "trial", "us"))
        errors.writerow((*fields, "phase", "field", "relative_l2", "max_abs"))
        for seed in (51, 137):
            for shape, k, dtype in [((3, 30, 47), 3, torch.float32),
                                   ((3, 30, 47), 3, torch.bfloat16),
                                   ((2, 4096, 512), 4, torch.bfloat16),
                                   ((8, 4096, 512), 4, torch.bfloat16)]:
                run(shape, k, dtype, seed, args.repeats, samples, errors)


if __name__ == "__main__":
    main()
