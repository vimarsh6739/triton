"""Compare the TTIR pullback of Liger's SwiGLU with its training implementation."""

import argparse
import copy
import csv
import random
import sys
from statistics import median
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton
from liger_kernel.ops import swiglu
from triton.experimental import Const, Duplicated, autodiff


class GeneratedSwiGLU(torch.autograd.Function):
    """Dense local tensors, default down multiplier, scalar gate multiplier."""

    @staticmethod
    def forward(ctx, a, b, scale):
        ctx.shape = a.shape
        a, b, out = swiglu.swiglu_forward(a, b, scale)
        ctx.save_for_backward(a, b)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, seed):
        a, b = ctx.saved_tensors
        rows, width = a.shape
        if swiglu._should_tile(width):
            kernel = swiglu._swiglu_forward_kernel_tiled
            block, warps = swiglu._swiglu_tile_settings(width)
            grid = (rows, triton.cdiv(width, block))
        else:
            kernel = swiglu._swiglu_forward_kernel
            block, warps = swiglu.calculate_settings(width)
            grid = (rows,)
        da, db = torch.zeros_like(a), torch.zeros_like(b)
        autodiff(kernel)[grid](
            Duplicated(a, da), Duplicated(b, db),
            Duplicated(torch.empty_like(a), seed.clone(memory_format=torch.contiguous_format)),
            Const(a.stride(0)), Const(ctx.scale), width, BLOCK_SIZE=block, num_warps=warps,
        )
        return da.view(ctx.shape), db.view(ctx.shape), None


def evaluate(fn, a, b, seed, scale):
    out = fn(a, b, scale)
    return (out, *torch.autograd.grad(out, (a, b), seed))


def generated_swiglu(mode):
    if mode == "generated":
        return GeneratedSwiGLU.apply
    from liger_writeback import SpecializedSwiGLU

    return lambda a, b, scale: SpecializedSwiGLU.apply(a, b, scale, mode)


def check_mlp(mode="generated", include_mode=False, header=True):
    from liger_kernel.transformers import swiglu as module

    torch.manual_seed(173)
    torch.set_float32_matmul_precision("highest")
    config = SimpleNamespace(hidden_act="silu", hidden_size=128, intermediate_size=384)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    if header:
        writer.writerow((["mode"] if include_mode else []) +
                        ["dtype", "output", "generated_vs_liger_rel_l2", "generated_vs_fp64_rel_l2"])
    for dtype in [torch.float32, torch.bfloat16]:
        model = module.LigerSwiGLUMLP(config).to(device="cuda", dtype=dtype)
        x = torch.randn((2, 17, config.hidden_size), device="cuda").to(dtype).requires_grad_()
        seed = torch.randn_like(x)

        def compute(net, features):
            y = net(features)
            return (y, *torch.autograd.grad(y, (features, *net.parameters()), seed.to(y.dtype)))

        hand = compute(model, x)
        with patch.object(module, "_swiglu_dispatch", lambda a, b: generated_swiglu(mode)(a, b, 1.0)):
            generated = compute(model, x)
        with patch.object(module, "_swiglu_dispatch", lambda a, b: torch.nn.functional.silu(a) * b):
            reference = compute(copy.deepcopy(model).double(), x.detach().double().requires_grad_())
        names = ["out", "dx", *dict(model.named_parameters())]
        tolerance = 2e-5 if dtype == torch.float32 else .05
        for name, new, old, ref in zip(names, generated, hand, reference):
            torch.testing.assert_close(new, old, atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(new.double(), ref, atol=tolerance, rtol=tolerance)
            writer.writerow(([mode] if include_mode else []) + [dtype, name, ((new.double() - old).norm() / old.double().norm()).item(),
                             ((new.double() - ref).norm() / ref.norm()).item()])


def run(rows, cols, dtype, scale, repeats, modes=(), random_seed=173):
    torch.manual_seed(random_seed)
    shape = (rows, cols)
    a, b, seed = [torch.randn(shape, device="cuda").to(dtype) for _ in range(3)]
    a64 = a.double().requires_grad_()
    b64 = b.double().requires_grad_()
    reference = evaluate(lambda x, y, s: torch.nn.functional.silu(s * x) * y, a64, b64, seed.double(), scale)
    impls = {"liger": swiglu.LigerSiLUMulFunction.apply, "generated": GeneratedSwiGLU.apply}
    impls.update({mode: generated_swiglu(mode) for mode in modes})
    errors, outputs, graphs, inputs = {}, {}, {}, {}
    atol, rtol = {torch.float32: (1e-4, 1e-4), torch.float16: (.05, .05), torch.bfloat16: (.1, .05)}[dtype]
    for name, fn in impls.items():
        x, y = (value.clone().requires_grad_() for value in (a, b))
        values = evaluate(fn, x, y, seed, scale)
        # The handwritten backward reuses its two input buffers for outputs.
        outputs[name] = tuple(value.detach().clone() for value in values)
        for value, expected in zip(values, reference):
            torch.testing.assert_close(value.double(), expected, atol=atol, rtol=rtol)
        errors[name] = [((v.double() - r).norm() / r.norm()).item() for v, r in zip(values, reference)]
    for generated, hand in zip(outputs["generated"], outputs["liger"]):
        torch.testing.assert_close(generated, hand, atol=atol, rtol=rtol)

    # Both replays start with identical restored inputs. Enqueue restorations
    # before the timing events. Keep all graph outputs alive for every replay.
    captured = {}
    for name, fn in impls.items():
        x, y = (value.clone().requires_grad_() for value in (a, b))
        inputs[name] = x, y
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured[name] = tuple(t.detach() for t in evaluate(fn, x, y, seed, scale))
        graphs[name] = graph
    samples = {name: [] for name in impls}
    rng = random.Random(random_seed)
    for trial in range(repeats + 5):
        order = list(impls)
        rng.shuffle(order)
        for name in order:
            x, y = inputs[name]
            with torch.no_grad():
                x.copy_(a)
                y.copy_(b)
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            if trial >= 5:
                samples[name].append(start.elapsed_time(end) * 1000)
    for name in impls:
        for value, expected in zip(captured[name], reference):
            torch.testing.assert_close(value.double(), expected, atol=atol, rtol=rtol)
    return errors, samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=41)
    parser.add_argument("--seed", type=int, default=173, help="Seed for activation data and timing order")
    parser.add_argument("--production-only", action="store_true", help="Only the 3,072 × 12,288 BF16 case")
    parser.add_argument("--samples", help="CSV output for individual timed replays")
    parser.add_argument("--mlp", action="store_true", help="Run the upstream MLP forward and parameter pullback check")
    parser.add_argument("--modes", nargs="+", default=[], choices=["discard", "read_seed", "overwrite", "donate", "stable"],
                        help="SwiGLU ownership specializations to check and time alongside the baselines")
    args = parser.parse_args()
    if args.mlp:
        for index, mode in enumerate(args.modes or ["generated"]):
            check_mlp(mode, include_mode=bool(args.modes), header=index == 0)
        sys.exit(0)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["rows", "width", "dtype", "gate", "impl", "out_rel_l2", "da_rel_l2", "db_rel_l2", "median_us"])
    timings = []
    cases = [(32, 256, 1.0), (128, 1024, 1.0), (256, 14336, 1.0),
             (37, 781, .625), (3072, 12288, 1.0)]
    for rows, cols, scale in cases[-1:] if args.production_only else cases:
        for dtype in ((torch.bfloat16,) if rows == 3072 else (torch.float32, torch.bfloat16, torch.float16)):
            errors, samples = run(rows, cols, dtype, scale, args.repeats, args.modes, args.seed)
            for name, values in samples.items():
                case = [rows, cols, str(dtype), scale, name]
                writer.writerow([*case, *errors[name], median(values)])
                timings.extend([*case, i, t] for i, t in enumerate(values))
            sys.stdout.flush()
    if args.samples:
        with open(args.samples, "w", newline="") as stream:
            raw = csv.writer(stream, lineterminator="\n")
            raw.writerow(["rows", "width", "dtype", "gate", "impl", "trial", "us"])
            raw.writerows(timings)
