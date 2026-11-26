"""Check combined reverse TTIR for tutorial and bounded-length attention."""

import argparse
import csv
import importlib
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.experimental as ad
import triton.language as tl
from torch.nn.attention import SDPBackend, sdpa_kernel


@triton.jit
def attention_tile(Q, K, V, O, N: tl.constexpr, D: tl.constexpr, M: tl.constexpr, S: tl.constexpr,
                   CAUSAL: tl.constexpr):
    """Each CTA processes M query positions and all S padded key positions."""
    h = tl.program_id(1)
    rows = tl.program_id(0) * M + tl.arange(0, M)
    cols = tl.arange(0, S)
    dim = tl.arange(0, D)
    q_addr = (h * N + rows[:, None]) * D + dim[None, :]
    kv_addr = (h * N + cols[:, None]) * D + dim[None, :]
    q = tl.load(Q + q_addr, rows[:, None] < N, 0)
    k = tl.load(K + kv_addr, cols[:, None] < N, 0)
    v = tl.load(V + kv_addr, cols[:, None] < N, 0)
    scores = tl.dot(q, tl.trans(k)) * (D**-0.5)
    visible = cols[None, :] < N
    if CAUSAL:
        visible = visible & (cols[None, :] <= rows[:, None])
    scores = tl.where(visible, scores, -float("inf"))
    numer = tl.exp(scores - tl.max(scores, 1)[:, None])
    prob = numer / tl.sum(numer, 1)[:, None]
    out = tl.dot(prob.to(v.dtype), v)
    tl.store(O + q_addr, out.to(q.dtype), rows[:, None] < N)


@triton.jit
def _stream_step(q, K, V, h, rows, dim, start, hi, mass, acc, N: tl.constexpr, D: tl.constexpr,
                 S: tl.constexpr, CAUSAL: tl.constexpr):
    cols = start + tl.arange(0, S)
    addr = (h * N + cols[:, None]) * D + dim[None, :]
    k = tl.load(K + addr, cols[:, None] < N, 0)
    v = tl.load(V + addr, cols[:, None] < N, 0)
    scores = tl.dot(q, tl.trans(k)) * (D**-0.5)
    visible = cols[None, :] < N
    if CAUSAL:
        visible = visible & (cols[None, :] <= rows[:, None])
    scores = tl.where(visible, scores, -float("inf"))
    new_hi = tl.maximum(hi, tl.max(scores, 1))
    prob = tl.exp(scores - new_hi[:, None])
    scale = tl.exp(hi - new_hi)
    mass = mass * scale + tl.sum(prob, 1)
    acc = tl.dot(prob.to(v.dtype), v, acc * scale[:, None])
    return new_hi, mass, acc


@triton.jit
def attention_stream(Q, K, V, O, N: tl.constexpr, D: tl.constexpr, M: tl.constexpr, S: tl.constexpr,
                     CAUSAL: tl.constexpr, UNROLL: tl.constexpr):
    h = tl.program_id(1)
    rows = tl.program_id(0) * M + tl.arange(0, M)
    dim = tl.arange(0, D)
    addr = (h * N + rows[:, None]) * D + dim[None, :]
    q = tl.load(Q + addr, rows[:, None] < N, 0)
    hi = tl.full((M,), -float("inf"), tl.float32)
    mass = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, D), tl.float32)
    if UNROLL:
        for block in tl.static_range(triton.cdiv(N, S)):
            hi, mass, acc = _stream_step(q, K, V, h, rows, dim, block * S, hi, mass, acc, N, D, S, CAUSAL)
    else:
        for block in range(triton.cdiv(N, S)):
            hi, mass, acc = _stream_step(q, K, V, h, rows, dim, block * S, hi, mass, acc, N, D, S, CAUSAL)
    tl.store(O + addr, (acc / mass[:, None]).to(q.dtype), rows[:, None] < N)


def launch(case, tensors, out, gradients, seed, causal, keep_temps=False):
    q, k, v = tensors
    b, h, n, d = q.shape
    active = [ad.Duplicated(x, dx) for x, dx in zip(tensors, gradients)]
    if case == "tutorial":
        if n % 64 or d < 32:
            raise ValueError("the tutorial has an unmasked statistic store and requires full 64-row tiles")
        kernel = importlib.import_module("06-fused-attention")._attn_fwd
        m = torch.empty((b, h, n), device=q.device)
        dm = torch.zeros_like(m)
        args = (ad.Const(d**-0.5), ad.Duplicated(m, dm), ad.Const(b), ad.Const(h), *active,
                ad.Duplicated(out, seed))
        return ad.autodiff(kernel, keep_temps=keep_temps)[(n // 64, b * h, 1)](
            *args, N_CTX=n, HEAD_DIM=d, BLOCK_M=64, BLOCK_N=32, FP8_OUTPUT=False,
            STAGE=3 if causal else 1, warp_specialize=False, IS_HOPPER=False, num_warps=4, num_stages=2)
    if case.startswith("stream"):
        return ad.autodiff(attention_stream, keep_temps=keep_temps)[(triton.cdiv(n, 16), b * h)](
            *active, ad.Duplicated(out, seed), N=n, D=d, M=16, S=32, CAUSAL=causal,
            UNROLL=case == "stream-unrolled", num_warps=4)
    return ad.autodiff(attention_tile)[(triton.cdiv(n, 16), b * h)](
        *active, ad.Duplicated(out, seed), N=n, D=d, M=16, S=max(16, triton.next_power_of_2(n)),
        CAUSAL=causal, num_warps=4)


def sdpa(tensors, seed, causal, backend):
    with sdpa_kernel(backend):
        out = F.scaled_dot_product_attention(*tensors, is_causal=causal)
        return (out.detach(), *torch.autograd.grad(out, tensors, seed))


def error(x, reference):
    diff = x.double() - reference
    return diff.abs().max().item(), (diff.norm() / reference.norm().clamp_min(1e-30)).item()


def verify(outputs, truth):
    for x, ref in zip(outputs, truth, strict=True):
        torch.testing.assert_close(x.double(), ref, atol=.006, rtol=.006)
        rel = error(x, ref)[1]
        if not math.isfinite(rel) or rel >= .003:
            raise AssertionError(f"relative L2 error {rel} exceeds tolerance")


def measure(fn, truth):
    # Keep inputs and outputs alive across capture and replay. Every replay
    # must compute the same four results, including any seed initialization.
    for _ in range(3):
        outputs = fn()
    torch.cuda.synchronize()
    verify(outputs, truth)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn()
    for _ in range(10):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(25):
        start.record()
        for _ in range(40):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / 40)
    verify(outputs, truth)
    return statistics.median(samples), sorted(samples)[6], sorted(samples)[18]


def append_csv(path, rows):
    if path:
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            if f.tell() == 0:
                writer.writeheader()
            writer.writerows(rows)


def run(args):
    torch.manual_seed(args.seed)
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    tensors = [torch.randn((args.batch, args.heads, args.n, args.dim), dtype=torch.float16, device="cuda")
               for _ in range(3)]
    seed = torch.randn_like(tensors[0])
    if args.prefix_seed:
        seed[..., args.n // 2:, :] = 0
    truth = sdpa([x.double().requires_grad_() for x in tensors], seed.double(), args.causal, SDPBackend.MATH)
    out = torch.empty_like(seed)
    grads = [torch.zeros_like(x) for x in tensors]
    shadow = seed.clone()
    compiled = launch(args.case, tensors, out, grads, shadow, args.causal, args.keep_temps)
    torch.cuda.synchronize()
    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
        for stage in ("ttir", "ttgir", "llir", "ptx"):
            (args.dump / (args.case + "." + stage)).write_text(compiled.asm[stage])
    rows = []
    for name, x, ref in zip(("O", "dQ", "dK", "dV"), (out, *grads), truth):
        abs_max, rel = error(x, ref)
        print(f"{name}: max={abs_max:.7g}, relative L2={rel:.7g}", flush=True)
        rows.append(dict(value=name, max_abs=abs_max, rel_l2=rel))
    print(f"nonzero output cotangent after launch: {torch.count_nonzero(shadow).item()}")
    labels = dict(case=args.case, causal=args.causal, n=args.n, d=args.dim, batch=args.batch,
                  heads=args.heads, seed=args.seed, prefix_seed=args.prefix_seed)
    if args.resources:
        ir = compiled.asm["ttir"]
        append_csv(args.resources, [dict(**labels, registers=compiled.n_regs, local_bytes_thread=4 * compiled.n_spills,
                                        shared_bytes_cta=compiled.metadata.shared, tt_dots=ir.count("tt.dot "),
                                        tt_atomics=ir.count("tt.atomic_rmw"), scf_loops=ir.count("scf.for"))])
    append_csv(args.csv, [dict(**labels, **row) for row in rows])
    verify((out, *grads), truth)
    assert not torch.count_nonzero(shadow)
    if args.prefix_seed and args.causal:
        for grad in grads:
            assert not torch.count_nonzero(grad[..., args.n // 2:, :])

    if args.timings:
        def generated():
            for dx in grads:
                dx.zero_()
            shadow.copy_(seed)
            launch(args.case, tensors, out, grads, shadow, args.causal, args.keep_temps)
            return (out, *grads)

        trainable = tuple(x.detach().requires_grad_() for x in tensors)
        runs = {
            "generated": generated,
            "torch_flash": lambda: sdpa(trainable, seed, args.causal, SDPBackend.FLASH_ATTENTION),
            "torch_math": lambda: sdpa(trainable, seed, args.causal, SDPBackend.MATH),
        }
        order = list(runs) if args.seed % 2 else list(reversed(runs))
        for name in order:
            fn = runs[name]
            median, q25, q75 = measure(fn, truth)
            append_csv(args.timings, [dict(**labels, implementation=name, median_us=median, q25_us=q25, q75_us=q75)])
            print(f"{name}: {median:.3f} us (IQR {q25:.3f} to {q75:.3f})", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case", choices=("tutorial", "tile", "stream-rolled", "stream-unrolled"))
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--prefix-seed", action="store_true")
    p.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--csv", type=Path)
    p.add_argument("--resources", type=Path)
    p.add_argument("--timings", type=Path)
    p.add_argument("--keep-temps", action="store_true", help="Retain Enzyme input and derivative MLIR")
    p.add_argument("--dump", type=Path)
    run(p.parse_args())
