"""Compare group-parallel pullbacks with the pinned Liger GroupNorm autograd function."""

import argparse
import csv
import random
from pathlib import Path

import torch
import triton
import triton.language as tl
from liger_kernel.ops.group_norm import LigerGroupNormFunction, group_norm_forward
from triton.experimental import Duplicated, autodiff


@triton.jit
def normalized_affine(x, w, b, N: tl.constexpr, EPS: tl.constexpr):
    mean = tl.sum(tl.sum(x, 1), 0) / N
    var = tl.sum(tl.sum(x * x, 1), 0) / N - mean * mean
    xhat = (x - mean) * tl.rsqrt(var + EPS)
    return xhat * w[:, None] + b[:, None]


@triton.jit
def group_primal(X, W, B, Y, GROUPS: tl.constexpr, K: tl.constexpr, H: tl.constexpr,
                 CK: tl.constexpr, SH: tl.constexpr, EPS: tl.constexpr):
    group = tl.program_id(0)
    channel = tl.arange(0, CK)
    space = tl.arange(0, SH)
    valid = (channel[:, None] < K) & (space[None, :] < H)
    offset = group * K * H + channel[:, None] * H + space[None, :]
    x = tl.load(X + offset, valid, 0).to(tl.float32)
    ci = (group % GROUPS) * K + channel
    w = tl.load(W + ci, channel < K, 0).to(tl.float32)
    b = tl.load(B + ci, channel < K, 0).to(tl.float32)
    tl.store(Y + offset, normalized_affine(x, w, b, K * H, EPS), valid)


@triton.jit
def group_pullback(X, W, Mean, InvStd, DY, DX, DW, DB, GROUPS: tl.constexpr, K: tl.constexpr,
                   H: tl.constexpr, CK: tl.constexpr, SH: tl.constexpr):
    group = tl.program_id(0)
    channel = tl.arange(0, CK)
    space = tl.arange(0, SH)
    valid = (channel[:, None] < K) & (space[None, :] < H)
    offset = group * K * H + channel[:, None] * H + space[None, :]
    ci = (group % GROUPS) * K + channel
    x = tl.load(X + offset, valid, 0).to(tl.float32)
    dy = tl.load(DY + offset, valid, 0).to(tl.float32)
    w = tl.load(W + ci, channel < K, 0).to(tl.float32)
    mean = tl.load(Mean + group).to(tl.float32)
    inv = tl.load(InvStd + group).to(tl.float32)
    xhat = (x - mean) * inv
    # The per-channel sums needed for dW and dB also give the group sums for dX.
    dw = tl.sum(dy * xhat, 1)
    db = tl.sum(dy, 1)
    c1 = tl.sum(dw * w, 0) / (K * H)
    c2 = tl.sum(db * w, 0) / (K * H)
    dx = (w[:, None] * dy - c2 - xhat * c1) * inv
    tl.store(DX + offset, dx, valid)
    tl.atomic_add(DW + ci, dw, channel < K, sem="relaxed")
    tl.atomic_add(DB + ci, db, channel < K, sem="relaxed")


class TiledGroupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, groups, eps, method):
        n, c, h = x.shape
        k = c // groups
        ctx.grid = (n * groups,)
        ctx.opts = dict(GROUPS=groups, K=k, H=h, CK=triton.next_power_of_2(k),
                        SH=triton.next_power_of_2(h))
        ctx.method, ctx.eps = method, eps
        if method in ("generated", "values"):
            y = torch.empty_like(x)
            group_primal[ctx.grid](x, w, b, y, EPS=eps, **ctx.opts)
            ctx.save_for_backward(x, w, b)
        else:
            y, _, mean, rstd, _ = group_norm_forward(x, c, groups, w, b, eps)
            ctx.save_for_backward(x, w, mean, rstd)
        return y

    @staticmethod
    def backward(ctx, dy):
        dy = dy.contiguous()
        x, w, *rest = ctx.saved_tensors
        dw, db = torch.zeros_like(w), torch.zeros_like(w)
        if ctx.method == "values":
            from groupnorm_values import value_kernel

            dx = torch.empty_like(x)
            kernel = value_kernel(x.dtype, ctx.opts["GROUPS"], ctx.opts["K"], ctx.opts["H"], ctx.eps,
                                  x.device.index)
            kernel[ctx.grid + (1, 1)](x, w, rest[0], dy, dx, dw, db)
        elif ctx.method == "generated":
            dx = torch.zeros_like(x)
            tmp = torch.empty_like(x)
            autodiff(group_primal)[ctx.grid](
                Duplicated(x, dx), Duplicated(w, dw), Duplicated(rest[0], db),
                Duplicated(tmp, dy.clone()), EPS=ctx.eps, **ctx.opts)
        else:
            dx = torch.empty_like(x)
            group_pullback[ctx.grid](x, w, *rest, dy, dx, dw, db, **ctx.opts)
        return dx, dw, db, None, None, None


def forward(x, w, b, groups, method):
    eps = 1e-6
    if method == "liger":
        return LigerGroupNormFunction.apply(x, w, b, len(w), groups, eps)
    if method == "torch":
        return torch.nn.functional.group_norm(x, groups, w, b, eps)
    return TiledGroupNorm.apply(x, w, b, groups, eps, method)


def evaluate(x, w, b, seed, groups, method):
    y = forward(x, w, b, groups, method)
    grads = torch.autograd.grad(y, (x, w, b), seed)
    return (y.detach(), *grads)


def run(shape, k, dtype, repeats, selected, seed_id, scope, writer):
    n, c, h = shape
    torch.manual_seed(seed_id)
    groups = c // k
    x, w, b = [torch.randn(s, device="cuda", dtype=dtype).requires_grad_() for s in (shape, (c,), (c,))]
    seed = torch.randn_like(x)
    truth = evaluate(*(t.detach().double().requires_grad_() for t in (x, w, b)), seed.double(), groups, "torch")
    methods = list(selected)
    graphs, held, primals = {}, {}, {}

    def verify(method, values):
        relative = []
        for field, a, e in zip(("y", "dx", "dw", "db"), values, truth):
            # Compare every element against a scale set by the FP64 vector, then
            # check relative L2. Near-zero parameter gradients need absolute slack.
            maximum = e.abs().max().item()
            tolerance = .05 if dtype == torch.bfloat16 else 5e-5
            torch.testing.assert_close(a.double(), e, atol=tolerance * maximum, rtol=tolerance)
            err = ((a.double() - e).norm() / e.norm()).item()
            if err > tolerance:
                raise ValueError(f"{method} {field}: relative L2 {err}")
            relative.append(err)
        return relative

    for name in methods:
        for _ in range(3):
            values = evaluate(x, w, b, seed, groups, name)
        verify(name, values)
        graph = torch.cuda.CUDAGraph()
        leaves = tuple(t.detach().requires_grad_() for t in (x, w, b))
        if scope == "backward":
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                y = forward(*leaves, groups, name)
            capture_stream.synchronize()
            primals[name] = y  # Preserve the saved forward tensors during replay.
            with torch.cuda.graph(graph, stream=capture_stream):
                grads = torch.autograd.grad(y, leaves, seed, retain_graph=True)
                held[name] = (y.detach(), *grads)
        else:
            with torch.cuda.graph(graph):
                held[name] = evaluate(*leaves, seed, groups, name)
        graphs[name] = graph
    rng = random.Random(seed_id)
    samples = {name: [] for name in methods}
    for i in range(repeats + 5):
        rng.shuffle(methods)
        for name in methods:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            if i >= 5:
                samples[name].append(start.elapsed_time(end) * 1000)
    for name in methods:
        errors = verify(name, held[name])
        for trial, us in enumerate(samples[name]):
            writer.writerow([seed_id, n, c, h, k, str(dtype), name, trial, us, *errors])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=51)
    parser.add_argument("--scope", choices=["full", "backward"], default="full")
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--methods", nargs="+", choices=["torch", "liger", "scheduled", "generated", "values"],
                        default=["torch", "liger", "scheduled", "generated", "values"])
    args = parser.parse_args()
    with args.out.open("x", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["seed", "batch", "channels", "spatial", "channels_per_group", "dtype", "method",
                         "trial", "us", "y_rel", "dx_rel", "dw_rel", "db_rel"])
        for shape, k in [((3, 30, 47), 3), ((2, 4096, 512), 4), ((8, 4096, 512), 4)]:
            for dtype in (torch.float32, torch.bfloat16):
                run(shape, k, dtype, args.repeats, args.methods, args.seed, args.scope, writer)
                stream.flush()

    if "values" in args.methods:
        from groupnorm_values import value_kernel

        value_kernel.cache_clear()
