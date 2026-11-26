"""Measure the counted AdamW programs, checking their buffer contracts against FP64."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

import torch
import triton
from enzyme_ad.jax import enzyme_call

from . import generated, handwritten


def reference(p, g, m, v):
    m1 = torch.lerp(g, m, 0.9)
    v1 = 0.999 * v + 0.001 * g.square()
    denom = (v1 / (1 - 0.999**32)).sqrt() + 1e-8
    p1 = (1 - 3e-3 * 1e-2) * p - (3e-3 / (1 - 0.9**32)) * m1 / denom
    return p1, m1, v1


def run(n, seed, rounds):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    p, g, m = [torch.randn(n, device="cuda") for _ in range(3)]
    v = torch.rand_like(p) + 1e-3
    inputs = (p, g, m, v)
    tangents = tuple(torch.randn_like(p) for _ in inputs)
    initial = tuple(torch.randn_like(p) for _ in inputs)
    seeds = tuple(torch.randn_like(p) for _ in range(3))
    outputs = tuple(torch.empty_like(p) for _ in seeds)
    shadows = tuple(torch.empty_like(p) for _ in seeds)
    grads = tuple(torch.empty_like(p) for _ in inputs)
    xd = tuple(x.double() for x in inputs)
    expected_y, expected_jvp = torch.func.jvp(reference, xd, tuple(x.double() for x in tangents))
    _, pullback = torch.func.vjp(reference, *xd)
    expected_vjp = tuple(dx + buf for dx, buf in zip(pullback(tuple(x.double() for x in seeds)), initial))
    zeros = tuple(torch.zeros_like(p) for _ in seeds)

    checks = []
    samples = []
    graphs = {}

    def check(actual, expected, label, stage):
        for index, (x, ref) in enumerate(zip(actual, expected, strict=True)):
            target = ref.to(x.dtype)
            torch.testing.assert_close(x, target, atol=8e-6, rtol=2e-5)
            checks.append(dict(seed=seed, n=n, method=name, mode=mode, stage=stage, output=label,
                               index=index, max_abs=float((x - target).abs().max())))

    def verify(stage):
        check(outputs, expected_y, "primal", stage)
        if mode == "jvp":
            check(shadows, expected_jvp, "tangent", stage)
        if mode == "vjp":
            check(grads, expected_vjp, "adjoint", stage)
            check(shadows, zeros, "consumed_seed", stage)

    def reset():
        if mode == "vjp":
            for dst, src in zip(grads + shadows, initial + seeds, strict=True):
                dst.copy_(src)

    for name, method in (("generated", generated), ("handwritten", handwritten)):
        for mode in ("primal", "jvp", "vjp"):
            launch = getattr(method, mode)
            args = (inputs, outputs)
            if mode != "primal":
                args += ((grads if mode == "vjp" else tangents), shadows)
            for dest in outputs + (shadows if mode == "jvp" else ()):
                dest.fill_(float("nan"))
            reset()
            compiled = launch(*args)
            verify("before")
            start = torch.cuda.Event(enable_timing=True, external=True)
            end = torch.cuda.Event(enable_timing=True, external=True)
            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                reset()
                start.record()
                launch(*args)
                end.record()
            digest = hashlib.sha256(compiled.asm["ttir"].encode()).hexdigest()
            graphs[name, mode] = (graph, start, end, digest)

    keys = list(graphs)
    for sample in range(-10, rounds):
        rng.shuffle(keys)
        for name, mode in keys:
            graph, start, end, digest = graphs[name, mode]
            graph.replay()
            torch.cuda.synchronize()
            if sample >= 0:
                samples.append(dict(seed=seed, n=n, mode=mode, method=name, sample=sample,
                                    us=1000 * start.elapsed_time(end), ttir_sha256=digest))
    for (name, mode), (graph, *_) in graphs.items():
        graph.replay()
        verify("after")
    return samples, checks


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=4_194_307)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 67])
    parser.add_argument("--rounds", type=int, default=41)
    args = parser.parse_args()
    if min(args.n, args.rounds) < 1:
        parser.error("n and rounds must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    timings, checks = [], []
    for seed in args.seeds:
        ts, cs = run(args.n, seed, args.rounds)
        timings.extend(ts)
        checks.extend(cs)
    write_rows(args.output / "timings.csv", timings)
    write_rows(args.output / "checks.csv", checks)
    with open(enzyme_call.__file__, "rb") as binary:
        digest = hashlib.file_digest(binary, "sha256").hexdigest()
    device = torch.cuda.get_device_properties(0)
    metadata = {"gpu": device.name, "uuid": str(device.uuid), "torch": torch.__version__,
                "cuda": torch.version.cuda, "triton": triton.__version__, "enzyme_sha256": digest}
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
