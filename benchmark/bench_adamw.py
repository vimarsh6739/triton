"""Compare generated and hand-written state derivatives for an AdamW update."""

import argparse
import csv
import hashlib
import random
import statistics
from pathlib import Path

import torch
import triton
from triton.experimental import Const, Duplicated, autodiff, fwddiff

from adamw import step, step_jvp, step_vjp


def reference(p, g, m, v):
    m1 = torch.lerp(g, m, 0.9)
    v1 = v * 0.999 + g.square() * 0.001
    denom = (v1 / (1 - 0.999**32)).sqrt() + 1e-8
    p1 = p * (1 - 3e-3 * 1e-2) - 3e-3 * (m1 / (1 - 0.9**32)) / denom
    return p1, m1, v1


def check(actual, expected):
    errors = []
    for result, target in zip(actual, expected, strict=True):
        target = target.to(result.dtype)
        torch.testing.assert_close(result, target, atol=8e-6, rtol=2e-5)
        errors.append((result - target).abs().max().item())
    return max(errors)


def event_samples(launch, prepare, repeats):
    # External events become timestamp nodes inside the graph. This lets each
    # VJP see fresh nonzero seeds while excluding the reset and host dispatch.
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        prepare()
        start.record()
        launch()
        end.record()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        graph.replay()
        torch.cuda.current_stream().synchronize()
        samples.append(1000 * start.elapsed_time(end))
    return samples


def run_size(n, opts, writer):
    torch.manual_seed(opts.seed)
    p, g, m = (torch.randn(n, device="cuda") for _ in range(3))
    v = torch.rand_like(p) + 1e-3
    inputs = p, g, m, v
    tangents = tuple(torch.randn_like(p) for _ in inputs)
    outputs = tuple(torch.empty_like(p) for _ in range(3))
    seeds = tuple(torch.randn_like(p) for _ in outputs)
    shadows = tuple(torch.empty_like(p) for _ in outputs)
    grads = tuple(torch.empty_like(p) for _ in inputs)
    # Nonzero initial adjoints test the accumulation part of the reverse ABI.
    initial_grads = tuple(torch.randn_like(p) for _ in inputs)

    ref_inputs = tuple(x.double() for x in inputs)
    ref_y, ref_jvp = torch.func.jvp(reference, ref_inputs, tuple(x.double() for x in tangents))
    _, pullback = torch.func.vjp(reference, *ref_inputs)
    ref_vjp = tuple(d + initial for d, initial in zip(pullback(tuple(s.double() for s in seeds)), initial_grads))
    zero_shadows = tuple(torch.zeros_like(p) for _ in outputs)

    selected_seeds = (seeds[0], *zero_shadows[1:])
    ref_dg = pullback(tuple(s.double() for s in selected_seeds))[1] + initial_grads[1]

    modes = (("primal", "triton", step), ("jvp", "generated", fwddiff(step)), ("jvp", "hand", step_jvp),
             ("vjp", "generated", autodiff(step)), ("vjp", "hand", step_vjp), ("vjp", "hand_atomic", step_vjp),
             ("vjp_g", "generated", autodiff(step)), ("vjp_g", "full_projected", autodiff(step)))
    jobs = [(b, w, mode, impl, kernel) for b in opts.blocks for w in opts.warps for mode, impl, kernel in modes
            if mode in opts.modes]
    random.Random(opts.seed + n).shuffle(jobs)
    for block, warps, mode, impl, kernel in jobs:
        buffers = inputs + outputs
        args = (*buffers, n)
        if mode != "primal":
            pairs = list(zip(buffers, (tangents if mode == "jvp" else grads) + shadows))
            if mode == "vjp_g" and impl == "generated":
                # Example reweighting differentiates the parameter update w.r.t.
                # the training gradient. Only G and P1 need shadow storage.
                args = (*[Duplicated(x, dx) if i in (1, 4) else Const(x)
                          for i, (x, dx) in enumerate(pairs)], Const(n))
            elif impl in ("generated", "full_projected"):
                args = (*[Duplicated(x, dx) for x, dx in pairs], Const(n))
            else:
                args = (*[arg for pair in pairs for arg in pair], n)
        grid = (triton.cdiv(n, block),)

        extra = {"ATOMIC": impl == "hand_atomic"} if kernel is step_vjp else {}

        def launch():
            return kernel[grid](*args, BLOCK=block, num_warps=warps, **extra)

        def prepare():
            if mode in ("vjp", "vjp_g"):
                # Use the same copies before both projected variants so cache
                # conditioning and the observed input adjoint match.
                out_seeds = selected_seeds if mode == "vjp_g" else seeds
                for buf, initial in zip(grads + shadows, initial_grads + out_seeds):
                    buf.copy_(initial)

        # Poison outputs so a preceding implementation can't hide a missing store.
        for dest in outputs + (shadows if mode == "jvp" else ()):
            dest.fill_(float("nan"))
        prepare()
        compiled = launch()
        primal_error = check(outputs, ref_y)
        derivative_error = 0.0
        if mode == "jvp":
            derivative_error = check(shadows, ref_jvp)
        if mode == "vjp":
            derivative_error = check(grads, ref_vjp)
            check(shadows, zero_shadows)
        if mode == "vjp_g":
            derivative_error = check([grads[1], shadows[0]], [ref_dg, zero_shadows[0]])
        ttir = compiled.asm["ttir"]
        if impl in ("generated", "full_projected"):
            prefix = "fwddiffe" if mode == "jvp" else "revdiffe"
            assert f"@{prefix}step(" in ttir
        digest = hashlib.sha256(ttir.encode()).hexdigest()
        if opts.ir_dir:
            opts.ir_dir.mkdir(exist_ok=True, parents=True)
            stem = f"{mode}-{impl}-b{block}-w{warps}-{digest[:12]}"
            (opts.ir_dir / f"{stem}.ttir").write_text(ttir)
            if mode == "primal":
                (opts.ir_dir / f"{stem}.ll").write_text(compiled.asm["llir"])
        times = event_samples(launch, prepare, opts.repeats)
        row = [torch.cuda.get_device_name(), n, "f32", mode, impl, block, warps, opts.seed, primal_error,
               derivative_error, compiled.n_regs, compiled.n_spills, ttir.count("tt.atomic_rmw"), digest]
        for sample, us in enumerate(times):
            writer.writerow([*row, sample, us])
        print(f"{n:8} {mode:6} {impl:10} block={block:4} warps={warps} {statistics.median(times):8.3f} us",
              flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1023, 65536, 1048576, 4194307])
    parser.add_argument("--blocks", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--warps", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--modes", nargs="+", choices=["primal", "jvp", "vjp", "vjp_g"],
                        default=["primal", "jvp", "vjp"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--ir-dir", type=Path)
    opts = parser.parse_args()
    if opts.repeats < 1 or min(opts.sizes) < 1:
        parser.error("sizes and repeats must be positive")
    opts.csv.parent.mkdir(parents=True, exist_ok=True)
    with opts.csv.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["gpu", "n", "dtype", "mode", "implementation", "block", "warps", "seed", "max_primal_abs",
                         "max_derivative_abs", "registers", "spills", "ttir_atomics", "ttir_sha256", "sample", "us"])
        for n in opts.sizes:
            run_size(n, opts, writer)
            stream.flush()


if __name__ == "__main__":
    main()
