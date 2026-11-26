"""Differentiate a sparse three-factor tensor reconstruction loss."""

import argparse
import csv
import hashlib
import inspect
import json
import subprocess
import random
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.experimental import Const, Duplicated, autodiff


@triton.jit
def _factor_row(factor, index, obs, lane, valid, R: tl.constexpr):
    row = tl.load(index + obs, valid, other=0)
    offset = row[:, None] * R + lane[None, :]
    values = tl.load(factor + offset, valid[:, None] & (lane[None, :] < R), other=0)
    return offset, values


@triton.jit
def sampled_loss(A, B, C, I, J, K, target, loss, N: tl.constexpr, R: tl.constexpr,
                 BATCH: tl.constexpr, WIDTH: tl.constexpr, HUBER: tl.constexpr):
    obs = tl.program_id(0) * BATCH + tl.arange(0, BATCH)
    lane = tl.arange(0, WIDTH)
    _, a = _factor_row(A, I, obs, lane, obs < N, R)
    _, b = _factor_row(B, J, obs, lane, obs < N, R)
    _, c = _factor_row(C, K, obs, lane, obs < N, R)
    r = tl.sum(a * b * c, axis=1) - tl.load(target + obs, obs < N, other=0)
    value = 0.5 * r * r
    if HUBER:
        value = tl.where(tl.abs(r) <= 1, value, tl.abs(r) - 0.5)
    tl.store(loss + obs, value, obs < N)


@triton.jit
def hand_vjp(A, DA, B, DB, C, DC, I, J, K, target, loss, seed, N: tl.constexpr,
             R: tl.constexpr, BATCH: tl.constexpr, WIDTH: tl.constexpr, HUBER: tl.constexpr):
    obs = tl.program_id(0) * BATCH + tl.arange(0, BATCH)
    lane = tl.arange(0, WIDTH)
    ia, a = _factor_row(A, I, obs, lane, obs < N, R)
    ib, b = _factor_row(B, J, obs, lane, obs < N, R)
    ic, c = _factor_row(C, K, obs, lane, obs < N, R)
    r = tl.sum(a * b * c, axis=1) - tl.load(target + obs, obs < N, other=0)
    value = 0.5 * r * r
    slope = r
    if HUBER:
        value = tl.where(tl.abs(r) <= 1, value, tl.abs(r) - 0.5)
        slope = tl.minimum(tl.maximum(r, -1.), 1.)
    tl.store(loss + obs, value, obs < N)
    weight = (slope * tl.load(seed + obs, obs < N, other=0))[:, None]
    mask = (obs < N)[:, None] & (lane < R)[None, :]
    tl.atomic_add(DA + ia, weight * b * c, mask, sem="relaxed")
    tl.atomic_add(DB + ib, weight * a * c, mask, sem="relaxed")
    tl.atomic_add(DC + ic, weight * a * b, mask, sem="relaxed")
    # The output seed can be replicated across warps in the factor layout.
    # All readers must finish before the representative store clears it.
    tl.debug_barrier()
    tl.store(seed + obs, 0, obs < N)


def case(n, rank, hot, huber, args, writer):
    torch.manual_seed(71 + n + rank)
    shape = (args.factor_rows, rank)
    factors = [torch.randn(shape, device="cuda") / rank**(1/6) for _ in range(3)]
    indices = [torch.randint(args.factor_rows, (n,), device="cuda", dtype=torch.int32) for _ in range(3)]
    if hot:
        # Half of the observations share one row of the first factor.
        indices[0][::2] = 0
    target, weights = [torch.randn(n, device="cuda") for _ in range(2)]
    initial = [torch.randn_like(x) * .01 for x in factors]
    derivatives = [torch.empty_like(x) for x in factors]
    output, seed = torch.empty_like(target), torch.empty_like(weights)

    refs = [x.double().requires_grad_() for x in factors]
    a, b, c = [f[i.long()] for f, i in zip(refs, indices)]
    residual = (a * b * c).sum(-1) - target.double()
    expected = 0.5 * residual.square()
    if huber:
        expected = torch.where(residual.abs() <= 1, expected, residual.abs() - .5)
    grads = torch.autograd.grad(expected, refs, weights.double())
    reference = [g + d for g, d in zip(grads, initial)]

    def prepare():
        seed.copy_(weights)
        for dest, buf in zip(derivatives, initial):
            dest.copy_(buf)

    active = [Duplicated(x, g) for x, g in zip(factors, derivatives)]
    pairs = [v for pair in zip(factors, derivatives) for v in pair]
    grid = (triton.cdiv(n, args.batch),)
    kwargs = dict(N=n, R=rank, WIDTH=triton.next_power_of_2(rank), BATCH=args.batch,
                  HUBER=huber, num_warps=args.warps)
    graphs, stats = {}, {}
    for method in ("generated", "hand"):
        if method == "generated":
            kernel = autodiff(sampled_loss)
            buffers = [*active, *map(Const, (*indices, target)), Duplicated(output, seed)]
        else:
            kernel = hand_vjp
            buffers = [*pairs, *indices, target, output, seed]
        def launch():
            return kernel[grid](*buffers, **kwargs)

        prepare()
        output.fill_(float("nan"))
        compiled = launch()
        ttir = compiled.asm["ttir"]
        if method == "generated":
            assert "@revdiffesampled_loss" in ttir
        (args.output / f"rank{rank}-n{n}-huber{int(huber)}-{method}.ttir").write_text(ttir)
        torch.testing.assert_close(output.double(), expected, atol=1e-5, rtol=1e-5)
        assert not seed.count_nonzero().item()
        rel = [((v.double()-r).norm()/r.norm()).item() for v, r in zip(derivatives, reference)]
        if max(rel) > 2e-4 or not all(torch.isfinite(v).all() for v in derivatives):
            raise AssertionError(f"{method}: relative gradient errors {rel}")
        for i, (actual, ref) in enumerate(zip(derivatives, reference)):
            # Heavily contended FP32 atomics accumulate in scheduler order.
            # Check the hot row with its own norm; test every other entry.
            first = 1 if hot and i == 0 else 0
            torch.testing.assert_close(actual[first:].double(), ref[first:], atol=1e-4, rtol=3e-5)
            if first:
                hot_error = (actual[0].double()-ref[0]).norm() / ref[0].norm()
                if hot_error > 2e-4:
                    raise AssertionError(f"{method}: hot row relative error {hot_error.item():g}")
        stats[method] = [ttir.count("tt.atomic_rmw"), compiled.n_regs, max(rel)]
        start, end = (torch.cuda.Event(enable_timing=True, external=True) for _ in range(2))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            prepare()
            start.record()
            launch()
            end.record()
        graphs[method] = (graph, start, end)
        for _ in range(5):
            graph.replay()
    torch.cuda.synchronize()
    order = list(graphs)
    rng = random.Random(21)
    for trial in range(args.repeats):
        rng.shuffle(order)
        for method in order:
            graph, start, end = graphs[method]
            graph.replay()
            end.synchronize()
            writer.writerow([n, rank, int(hot), "huber" if huber else "squared", method,
                             *stats[method], trial, 1000 * start.elapsed_time(end)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observations", type=int, nargs="+", default=[8191, 65539])
    parser.add_argument("--ranks", type=int, nargs="+", default=[24, 64])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--warps", type=int, default=1)
    parser.add_argument("--factor-rows", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from enzyme_ad.jax import enzyme_call
    from triton.backends.nvidia.compiler import get_ptxas_version
    from triton.runtime.cache import triton_key

    with Path(inspect.getfile(enzyme_call)).open("rb") as lib:
        enzyme_sha = hashlib.file_digest(lib, "sha256").hexdigest()
    config = dict(vars(args), output=args.output.name, gpu=torch.cuda.get_device_name(),
                  triton=triton.__version__, torch=torch.__version__, cuda=torch.version.cuda,
                  triton_key=hashlib.sha256(triton_key().encode()).hexdigest(), enzyme_sha256=enzyme_sha,
                  ptxas=get_ptxas_version(), driver=subprocess.check_output(
                      ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                      text=True).splitlines()[0])
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    with (args.output / "timings.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["observations", "rank", "hot_row", "loss", "method", "atomics",
                         "registers", "relative_l2_gradient_error", "trial", "us"])
        for n in args.observations:
            for rank in args.ranks:
                for hot in (False, True):
                    for huber in (False, True):
                        case(n, rank, hot, huber, args, writer)
                        handle.flush()


if __name__ == "__main__":
    main()
