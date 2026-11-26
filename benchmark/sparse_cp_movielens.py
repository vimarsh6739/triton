"""Train a user × item × week CP model with a generated Triton pullback on MovieLens 100K."""

import argparse
import csv
import hashlib
import math
import zipfile
from pathlib import Path

import numpy as np
import torch
import triton
from triton.experimental import Const, Duplicated, autodiff

from sparse_cp import sampled_loss


ARCHIVE_SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
WEEK = 7 * 24 * 60 * 60


def load_ratings(archive):
    with archive.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != ARCHIVE_SHA256:
            raise ValueError("Expected the GroupLens ml-100k.zip archive")
    with zipfile.ZipFile(archive) as z:
        splits = []
        for name in ("u1.base", "u1.test"):
            with z.open("ml-100k/" + name) as f:
                splits.append(torch.from_numpy(np.loadtxt(f, dtype=np.int64)).to("cuda"))
    train, test = splits
    train_mean = train[:, 2].double().mean().item()
    first_week = (train[:, 3] // WEEK).min().item()

    def encode(split):
        indices = [split[:, 0] - 1, split[:, 1] - 1, split[:, 3] // WEEK - first_week]
        return [a.int().contiguous() for a in indices], (split[:, 2] - train_mean).float()

    return encode(train), encode(test), train_mean


def differentiate(kernel, factors, indices, target, adjoints, output, seed):
    n = len(target)
    rank = factors[0].shape[1]
    active = [Duplicated(x, dx) for x, dx in zip(factors, adjoints)]
    return kernel[(triton.cdiv(n, 16),)](*active, *map(Const, indices), Const(target),
                                         Duplicated(output, seed), N=n, R=rank, BATCH=16,
                                         WIDTH=triton.next_power_of_2(rank), HUBER=False, num_warps=1)


def check_gradient(kernel, factors, train):
    indices, target = train
    # An irregular subset exercises the partial observation tile and repeated
    # users/items/weeks. Nonuniform seeds check the full vector-output pullback.
    ids = torch.arange(0, 1023 * 71, 71, device=target.device)
    idx = [a[ids] for a in indices]
    truth = target[ids]
    seeds = torch.linspace(-0.7, 1.1, len(ids), device=target.device)
    refs = [f.detach().double().requires_grad_() for f in factors]
    a, b, c = [f[i.long()] for i, f in zip(idx, refs)]
    losses = 0.5 * ((a * b * c).sum(-1) - truth.double()).square()
    expected = torch.autograd.grad(losses, refs, seeds.double())
    output = torch.full_like(truth, float("nan"))
    adjoints = [torch.zeros_like(f) for f in factors]
    seed = seeds.clone()
    compiled = differentiate(kernel, factors, idx, truth, adjoints, output, seed)
    assert "@revdiffesampled_loss(" in compiled.asm["ttir"]
    torch.testing.assert_close(output.double(), losses, atol=2e-6, rtol=2e-5)
    assert seed.count_nonzero() == 0
    relative_errors = []
    for grad, ref in zip(adjoints, expected):
        torch.testing.assert_close(grad.double(), ref, atol=5e-6, rtol=5e-5)
        relative_errors.append(((grad.double() - ref).norm() / ref.norm()).item())
    return max(relative_errors)


def rmse(factors, split, seen):
    indices, target = split
    known = torch.ones_like(target, dtype=torch.bool)
    rows = []
    for factor, idx, train_rows in zip(factors, indices, seen):
        safe_idx = idx.long().clamp(0, len(factor) - 1)
        known &= (idx >= 0) & (idx < len(factor)) & train_rows[safe_idx]
        rows.append(factor[safe_idx].double())
    prediction = (rows[0] * rows[1] * rows[2]).sum(-1)
    residual = torch.where(known, prediction, 0.0) - target.double()
    return residual.square().mean().sqrt().item(), (~known).sum().item()


def fit(train, test, train_mean, args, seed_value, writer):
    torch.manual_seed(seed_value)
    indices, target = train
    n = len(target)
    sizes = [int(i.max()) + 1 for i in indices]
    seen = [torch.bincount(i.long(), minlength=s) > 0 for i, s in zip(indices, sizes)]
    factors = [torch.randn(s, args.rank, device="cuda") * 0.05 for s in sizes]
    # Start the time factor near one so every week initially participates in
    # the same low-rank user-item model. All three factors are trainable.
    factors[2].add_(1)
    params = [torch.nn.Parameter(x) for x in factors]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.decay)
    kernel = autodiff(sampled_loss)
    grad_error = check_gradient(kernel, params, train)
    adjoints = [torch.zeros_like(p) for p in params]
    for p, grad in zip(params, adjoints):
        p.grad = grad

    def record(epoch):
        with torch.no_grad():
            train_rmse, _ = rmse(params, train, seen)
            test_rmse, cold, baseline = "", "", ""
            if epoch == args.epochs:
                test_rmse, cold = rmse(params, test, seen)
                baseline = test[1].double().square().mean().sqrt().item()
        writer.writerow([seed_value, epoch, n, len(test[1]), args.rank, train_mean, train_rmse, test_rmse, baseline,
                         cold, grad_error])
        print(f"seed={seed_value} epoch={epoch} train_rmse={train_rmse:.5f} test_rmse={test_rmse}", flush=True)

    record(0)
    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(n, device="cuda")
        for batch in permutation.split(args.batch_size):
            y = target[batch]
            output = torch.empty_like(y)
            seed = torch.full_like(y, 1 / len(y))
            for grad in adjoints:
                grad.zero_()
            differentiate(kernel, params, [i[batch] for i in indices], y, adjoints, output, seed)
            optimizer.step()
        record(epoch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 43, 97])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--decay", type=float, default=0.01)
    args = parser.parse_args()
    if min(args.rank, args.epochs, args.batch_size) < 1 or not (math.isfinite(args.lr) and args.lr > 0):
        parser.error("rank, epochs, minibatch size and learning rate must be positive")
    if not math.isfinite(args.decay) or args.decay < 0:
        parser.error("weight decay must be finite and nonnegative")
    train, test, mean = load_ratings(args.archive)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["seed", "epoch", "train_count", "test_count", "rank", "train_mean", "train_rmse", "test_rmse",
                         "train_mean_test_rmse", "cold_test_count", "max_factor_relative_gradient_error"])
        for seed in args.seeds:
            fit(train, test, mean, args, seed, writer)
            f.flush()


if __name__ == "__main__":
    main()
