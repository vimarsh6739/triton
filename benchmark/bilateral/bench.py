"""Time bilateral slicing and all three input gradients on the same image."""

import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import random
import subprocess

import numpy as np
import torch

from inputs import make_inputs, oracle
from torch_impl import slice_apply as torch_slice


def reference(grid, guide, image, seed, strip_height=32):
    """Differentiate a separate FP64 interpolation, keeping memory bounded."""
    coefficients = grid.detach().double().requires_grad_()
    dgrid = torch.zeros_like(coefficients)
    y = torch.empty_like(image, dtype=torch.float64)
    dg = torch.empty_like(guide, dtype=torch.float64)
    di = torch.empty_like(image, dtype=torch.float64)
    height = image.shape[1]
    for first in range(0, height, strip_height):
        rows = slice(first, first + strip_height)
        g = guide[:, rows].detach().double().requires_grad_()
        rgb = image[:, rows].detach().double().requires_grad_()
        output = oracle(coefficients, g, rgb, row_offset=first, full_height=height)
        dcoeff, dguide, dimage = torch.autograd.grad(output, (coefficients, g, rgb), seed[:, rows].double())
        y[:, rows] = output.detach()
        dgrid.add_(dcoeff)
        dg[:, rows] = dguide
        di[:, rows] = dimage
    return y, dgrid, dg, di


def verify(values, truth):
    results = []
    for field, actual, expected in zip(("rgb", "grid", "guide", "input"), values, truth, strict=True):
        diff = actual.detach().double() - expected
        max_abs = diff.abs().max().item()
        rel = (diff.norm() / expected.norm().clamp_min(1e-20)).item()
        scaled_max = max_abs / max(expected.abs().max().item(), 1e-20)
        results.append((field, rel, max_abs, scaled_max))
        if not torch.isfinite(diff).all() or rel > 5e-4 or scaled_max > 1e-2:
            raise ValueError(f"{field} failed FP64 check: rel_l2={rel:g}, scaled_max={scaled_max:g}")
    return results


def run_case(args, methods, height, width, seed_id, samples, errors):
    grid, guide, image, cotangent = make_inputs(args.image, height, width, seed_id,
                                                grid_shape=tuple(args.grid), batch=args.batch, device="cuda")
    if args.trained_inputs:
        if args.batch != 1 or height != width:
            raise ValueError("the saved HDRNet activations are square batch-one images")
        with np.load(args.trained_inputs / f"{height}.npz") as data:
            grid, guide = (torch.from_numpy(data[k]).to(image.device).contiguous().requires_grad_()
                           for k in ("grid", "guide"))
        if grid.shape != (1, *args.grid, 3, 4) or guide.shape != image.shape[:-1]:
            raise ValueError("saved activations disagree with the requested dimensions")
    truth = reference(grid, guide, image, cotangent)
    original_seed = cotangent.clone()

    def train(fn):
        y = fn(grid, guide, image)
        return y.detach(), *torch.autograd.grad(y, (grid, guide, image), cotangent)

    def check(name, stage, values):
        try:
            rows = verify(values, truth)
        except ValueError as error:
            raise ValueError(f"{name} at {height}x{width}, seed {seed_id}, {stage}: {error}") from error
        for row in rows:
            errors.writerow((seed_id, args.batch, height, width, *args.grid, name, stage, *row))
        torch.testing.assert_close(cotangent, original_seed, rtol=0, atol=0)

    graphs, outputs = {}, {}
    for name, fn in methods.items():
        for _ in range(2):
            values = train(fn)
        check(name, "warm", values)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs[name] = train(fn)
        graphs[name] = graph
    methods_order = list(graphs)
    rng = random.Random(seed_id)
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for trial in range(-5, args.rounds):
        rng.shuffle(methods_order)
        for name in methods_order:
            start.record()
            graphs[name].replay()
            stop.record()
            stop.synchronize()
            if trial >= 0:
                samples.writerow((seed_id, args.batch, height, width, *args.grid, name, trial, start.elapsed_time(stop)))
    for name, values in outputs.items():
        check(name, "replay", values)
    print(f"checked and timed {height}x{width}, batch {args.batch}, seed {seed_id}", flush=True)


def get_methods(names):
    modules = {"ttir": "triton_ad", "fused_cuda": "cuda_fused", "warp_cuda": "cuda_warp"}
    result = {}
    for name in names:
        if name == "torch":
            result[name] = torch_slice
        elif name == "compiled":
            # Select the visible NVIDIA device for Inductor's Triton helpers.
            from triton.backends.nvidia.driver import CudaDriver
            from triton.runtime import driver
            driver.set_active(CudaDriver())
            result[name] = torch.compile(torch_slice, fullgraph=True, dynamic=False)
        else:
            result[name] = importlib.import_module(modules.get(name, name)).slice_apply
    return result


def record_environment(args):
    import triton
    from enzyme_ad.jax import enzyme_call

    def sha256(path):
        with Path(path).open("rb") as f:
            return hashlib.file_digest(f, "sha256").hexdigest()

    device = torch.cuda.get_device_properties(0)
    frontend = Path(triton.__file__).resolve().parent
    inputs = {"photo": sha256(args.image)}
    if args.trained_inputs:
        for size in args.sizes:
            height = size.split("x")[0]
            inputs[height] = sha256(args.trained_inputs / f"{height}.npz")
    config = {"gpu": device.name, "gpu_uuid": str(device.uuid), "torch": torch.__version__,
              "cuda": torch.version.cuda, "triton": triton.__version__,
              "enzyme_sha256": sha256(enzyme_call.__file__), "input_sha256": inputs,
              "methods": args.methods, "seeds": args.seeds, "sizes": args.sizes, "batch": args.batch,
              "grid": args.grid, "rounds": args.rounds}
    # Installed wheels may have no Git checkout.
    revision = subprocess.run(["git", "-C", str(frontend), "rev-parse", "HEAD"], text=True, capture_output=True)
    if revision.returncode == 0:
        config["frontend_commit"] = revision.stdout.strip()
    config["driver"] = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                               text=True).splitlines()[0]
    (args.out / "environment.json").write_text(json.dumps(config, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--methods", nargs="+", choices=["torch", "compiled", "cuda", "cuda_strict", "cuda_rgb", "fused_cuda", "warp_cuda", "ttir"],
                        default=["torch", "compiled", "cuda_strict", "fused_cuda", "warp_cuda", "ttir"])
    parser.add_argument("--sizes", nargs="+", default=["1024x1024", "2048x2048"])
    parser.add_argument("--grid", nargs=3, type=int, default=[16, 16, 8], metavar=("GH", "GW", "DEPTH"))
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--trained-inputs", type=Path, help="directory of size.npz files from pretrained/extract.py")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 43])
    parser.add_argument("--rounds", type=int, default=21)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.cuda.init()
    record_environment(args)
    methods = get_methods(args.methods)
    with (args.out / "timings.csv").open("w", newline="") as raw, (args.out / "accuracy.csv").open("w", newline="") as acc:
        samples, errors = (csv.writer(f, lineterminator="\n") for f in (raw, acc))
        dims = ("seed", "batch", "height", "width", "grid_h", "grid_w", "depth", "method")
        samples.writerow((*dims, "trial", "ms"))
        errors.writerow((*dims, "stage", "field", "relative_l2", "max_abs", "max_error_over_peak"))
        for seed in args.seeds:
            for size in args.sizes:
                height, width = map(int, size.split("x"))
                run_case(args, methods, height, width, seed, samples, errors)
                raw.flush()
                acc.flush()


if __name__ == "__main__":
    main()
