"""Compare one training minibatch's squared loss and all three factor gradients."""

import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import random
import statistics
import subprocess
import zipfile

import numpy as np
import torch

ARCHIVE_SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
WEEK = 7 * 24 * 60 * 60


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_training(archive):
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Expected the unmodified MovieLens 100K archive")
    with zipfile.ZipFile(archive) as z, z.open("ml-100k/u1.base") as f:
        data = torch.from_numpy(np.loadtxt(f, dtype=np.int64)).to("cuda")
    week = data[:, 3] // WEEK
    indices = [data[:, 0] - 1, data[:, 1] - 1, week - week.min()]
    ratings = data[:, 2].double()
    return [idx.int().contiguous() for idx in indices], (ratings - ratings.mean()).float()


def oracle(factors, indices, ratings, seed, initial):
    tables = [x.double().requires_grad_() for x in factors]
    u, v, w = [table[idx.long()] for table, idx in zip(tables, indices)]
    residual = (u * v * w).sum(-1) - ratings.double()
    losses = 0.5 * residual.square()
    derivatives = torch.autograd.grad(losses, tables, seed.double())
    return [losses, *(d + x.double() for d, x in zip(derivatives, initial))]


def verify(actual, expected, seed):
    if seed.count_nonzero().item():
        raise AssertionError("Output seed was not consumed")
    errors = {}
    for name, measured, ref in zip(("loss", "users", "items", "weeks"), actual, expected):
        diff = measured.double() - ref
        torch.testing.assert_close(measured.double(), ref, atol=5e-6, rtol=4e-5)
        errors[name] = (diff.norm() / ref.norm()).item()
    return errors


def measure(data, factors, ids, seed_value, args, methods):
    indices, ratings = data
    n = ids.numel()
    batch_indices = [idx[ids] for idx in indices]
    batch_ratings = ratings[ids]
    output = torch.empty_like(batch_ratings)
    initial = [torch.randn_like(table) * 0.01 for table in factors]
    gradients = [torch.empty_like(table) for table in factors]
    trial_seed = torch.randn_like(batch_ratings)
    seed = torch.empty_like(batch_ratings)
    trial_truth = oracle(factors, batch_indices, batch_ratings, trial_seed, initial)
    mean_seed = torch.full_like(seed, 1 / n)
    train_truth = oracle(factors, batch_indices, batch_ratings, mean_seed,
                         [torch.zeros_like(table) for table in factors])

    def check(method, phase, truth):
        errors = verify([output, *gradients], truth, seed)
        return dict(seed=seed_value, batch=n, method=method, phase=phase, **errors)

    def train_call(fn):
        idx = [col[ids] for col in indices]
        y = ratings[ids]
        for gradient in gradients:
            gradient.zero_()
        seed.fill_(1 / n)
        fn(factors, idx, y, output, gradients, seed)

    slots = []
    if args.scope == "kernel":
        for _ in range(args.calls):
            slots.append((torch.empty_like(output), [torch.empty_like(t) for t in factors], torch.empty_like(seed)))

    graphs, checks = {}, []
    for method, fn in methods.items():
        seed.copy_(trial_seed)
        for gradient, start in zip(gradients, initial):
            gradient.copy_(start)
        compiled = fn(factors, batch_indices, batch_ratings, output, gradients, seed)
        if method == "generated" and "@revdiffesquared_loss" not in compiled.asm["ttir"]:
            raise AssertionError("Missing generated reverse kernel")
        checks.append(check(method, "signed-seed-nonzero-initial", trial_truth))
        train_call(fn)
        checks.append(check(method, "training-mean", train_truth))
        graph = torch.cuda.CUDAGraph()
        start, stop = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)]
        with torch.cuda.graph(graph):
            if args.scope == "kernel":
                for _, adjoints, cotangent in slots:
                    for gradient in adjoints:
                        gradient.zero_()
                    cotangent.fill_(1 / n)
                start.record()
                for out, adjoints, cotangent in slots:
                    fn(factors, batch_indices, batch_ratings, out, adjoints, cotangent)
                stop.record()
            else:
                for _ in range(args.calls):
                    train_call(fn)
        graphs[method] = graph, start, stop

    order = list(graphs)
    rng = random.Random(seed_value + n)
    samples = []
    for round_ in range(-5, args.rounds):
        rng.shuffle(order)
        for method in order:
            graph, start, stop = graphs[method]
            if args.scope == "batch":
                start.record()
            graph.replay()
            if args.scope == "batch":
                stop.record()
            stop.synchronize()
            if round_ >= 0:
                us = start.elapsed_time(stop) * 1000 / args.calls
                samples.append(dict(seed=seed_value, batch=n, method=method, round=round_, us=us))
    for method, (graph, _, _) in graphs.items():
        graph.replay()
        if args.scope == "kernel":
            for slot, (out, adjoints, cotangent) in enumerate(slots):
                errors = verify([out, *adjoints], train_truth, cotangent)
                checks.append(dict(seed=seed_value, batch=n, method=method, phase=f"after-slot-{slot}", **errors))
        else:
            checks.append(check(method, "after-replay", train_truth))
    for method in methods:
        us = [r["us"] for r in samples if r["method"] == method]
        print(seed_value, n, method, statistics.median(us), flush=True)
    return samples, checks


def save_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 43])
    p.add_argument("--scope", choices=["batch", "kernel"], default="batch")
    p.add_argument("--calls", type=int, default=32)
    p.add_argument("--rounds", type=int, default=51)
    args = p.parse_args()
    if min(args.calls, args.rounds) < 1:
        p.error("calls and rounds must both be positive")
    data = load_training(args.archive)
    import triton
    from triton.backends.nvidia.compiler import get_ptxas_version
    from triton.runtime.cache import triton_key
    import generated
    import handwritten

    methods = {"generated": generated.differentiate, "handwritten": handwritten.differentiate}
    print(torch.cuda.get_device_properties(0), flush=True)
    sizes = [col.max().item() + 1 for col in data[0]]
    rank = 16
    samples, checks = [], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        factors = [torch.randn(size, rank, device="cuda") * 0.05 for size in sizes]
        factors[2].add_(1)
        permutation = torch.randperm(len(data[1]), device="cuda")
        batches = permutation.split(2048)
        for ids in (batches[0], batches[-1]):
            timings, errors = measure(data, factors, ids, seed, args, methods)
            samples.extend(timings)
            checks.extend(errors)
    args.out.mkdir(parents=True, exist_ok=False)
    prefix = "kernel-" if args.scope == "kernel" else ""
    save_csv(args.out / f"{prefix}timings.csv", samples)
    save_csv(args.out / f"{prefix}checks.csv", checks)
    from enzyme_ad.jax import enzyme_call

    config = dict(scope=args.scope, seeds=args.seeds, calls=args.calls, rounds=args.rounds, rank=rank,
                  training_rows=len(data[1]), table_rows=sizes, archive_sha256=ARCHIVE_SHA256,
                  gpu=torch.cuda.get_device_name(), gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                  torch=torch.__version__, triton=triton.__version__,
                  cuda=torch.version.cuda, ptxas=get_ptxas_version(),
                  enzyme_sha256=sha256(inspect.getfile(enzyme_call)),
                  triton_key=hashlib.sha256(triton_key().encode()).hexdigest(),
                  driver=subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version",
                                                  "--format=csv,noheader"], text=True).splitlines()[0])
    (args.out / f"{prefix}environment.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
