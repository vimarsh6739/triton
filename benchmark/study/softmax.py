"""Time the same TTIR softmax with and without differentiating its common shift."""

import argparse
import csv
import hashlib
import json
import random
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.experimental.autodiff import _find_tt_func, _make_public
from centered_softmax import TEMPLATES, compile_tensor_derivative


@triton.jit
def softmax(x, y, COLS: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, TILE)
    offset = row * COLS + lane
    value = tl.load(x + offset, lane < COLS, other=-float("inf"))
    numerator = tl.exp(value - tl.max(value, 0))
    tl.store(y + offset, numerator / tl.sum(numerator, 0), lane < COLS)


def compile_derivative(opt, primal, mode, schedule, directory):
    wrap = (f"infn=softmax outfn=gradient argTys=enzyme_dup,enzyme_dup mode={mode}")
    prefix = ["--detach-triton-softmax-shift"] if schedule == "detach" else []
    output = subprocess.check_output(
        [opt, *prefix, f"--enzyme-wrap={wrap}", "--canonicalize", "--remove-unnecessary-enzyme-ops",
         "--canonicalize", "--cse"], input=primal, text=True)
    fn = _make_public(_find_tt_func(output, "gradient"))
    ir = directory / f"{mode}-{schedule}.ttir"
    ir.write_text(f"module {{\n{fn}\n}}\n")
    return ir, fn.count('"tt.reduce"')


def fingerprint(opt, center_seed):
    """Pin the measurement cache to the code generator, benchmark and device."""
    def digest(path):
        with Path(path).open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest()

    from triton.runtime.cache import triton_key
    from triton.backends.nvidia.compiler import get_ptxas_version

    target = {
        "benchmark": digest(__file__), "optimizer": digest(opt), "triton": hashlib.sha256(triton_key().encode()).hexdigest(),
        "ptxas": get_ptxas_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "driver": torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version")
        else subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                     text=True).splitlines()[0],
        "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
    }
    if center_seed:
        files = [Path(__file__).with_name("centered_softmax.py"), *sorted(TEMPLATES.glob("*.mlir.in"))]
        target["tensor_wrapper"] = {path.name: digest(path) for path in files}
    return target


def check(kernel, mode, x, vector):
    rows = x.shape[0]
    y = torch.full_like(x, float("nan"))
    if mode == "ForwardMode":
        grad = torch.full_like(x, float("nan"))
        kernel[(rows, 1, 1)](x, vector, y, grad)
    else:
        grad = torch.zeros_like(x)
        seed = vector.clone()
        kernel[(rows, 1, 1)](x, grad, y, seed)
        assert torch.count_nonzero(seed) == 0
    # Softmax's Jacobian is symmetric, so both AD modes have this reference.
    p = x.double().softmax(-1)
    u = vector.double()
    reference = p * (u - (p * u).sum(dim=-1, keepdim=True))
    torch.testing.assert_close(y, p.float(), atol=2e-6, rtol=1e-5)
    if not torch.isfinite(grad).all():
        raise ValueError("nonfinite softmax derivative")
    return (grad.double() - reference).abs().max().item()


def measure(kernels, x, vector, mode, slots, repeats):
    # A reverse launch consumes its output seed. Each graph node gets a
    # different seed and input adjoint; restore them before starting the clock.
    rows, cols = x.shape
    # The primal specializes its base pointers to 16-byte alignment.
    stride = triton.cdiv(rows * cols, 4) * 4
    buffers = [torch.empty((slots, stride), device=x.device)[:, :rows * cols].view(slots, rows, cols)
               for _ in range(3)]
    dx, y, dy = buffers
    def reset():
        if mode == "ForwardMode":
            dx.copy_(vector.expand_as(dx))
        else:
            dx.zero_()
            dy.copy_(vector.expand_as(dy))

    reset()
    graphs = {}
    for key, compiled in kernels.items():
        graph = torch.cuda.CUDAGraph()
        calls = [(x, dx[i], y[i], dy[i]) for i in range(slots)]
        with torch.cuda.graph(graph):
            for args in calls:
                compiled[(rows, 1, 1)](*args)
        graphs[key] = graph
        reset()
        graph.replay()
    torch.cuda.synchronize()
    order = list(kernels)
    rng = random.Random(537)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for trial in range(repeats):
        rng.shuffle(order)
        for key in order:
            reset()
            start.record()
            graphs[key].replay()
            end.record()
            end.synchronize()
            yield key, trial, start.elapsed_time(end) * 1000 / slots


def experiment(args, phase, rows, cols, mode, cache, writer, checks):
    device = "cuda"
    torch.manual_seed(912 + cols + rows)
    x = torch.randn(rows, cols, device=device)
    vector = torch.randn_like(x)
    # This pins a tie at the maximum and tests the symmetric max subgradient.
    x[0, :2] = 20
    out = torch.empty_like(x)
    directory = args.output / f"cols-{cols}"
    directory.mkdir(exist_ok=True)
    primal = softmax[(rows,)](x, out, COLS=cols, TILE=triton.next_power_of_2(cols)).asm["ttir"]
    # Strip source locations through MLIR before extracting the standalone TTIR.
    primal = subprocess.check_output([args.opt, "--strip-debuginfo"], input=primal, text=True)
    (directory / "primal.ttir").write_text(primal)
    signature = f"{mode}:{cols}:" + hashlib.sha256(primal.encode()).hexdigest()
    if phase != "calibrate" and signature not in cache:
        raise ValueError("No matching calibration for this TTIR and mode")
    kernels, metadata = {}, {}
    schedules = ["baseline", "detach"]
    if args.center_seed or args.pivot_seed:
        schedules.extend(["tensor-control", "tensor-centered"])
    if args.pivot_seed:
        schedules.append("tensor-pivot")
    for schedule in schedules:
        if args.replay and not cache[signature].startswith(schedule + "-"):
            continue
        if schedule.startswith("tensor-"):
            ir, reductions = compile_tensor_derivative(args.opt, cols, mode, schedule, directory)
        else:
            ir, reductions = compile_derivative(args.opt, primal, mode, schedule, directory)
        for warps in args.warps:
            key = f"{schedule}-w{warps}"
            if args.replay and key != cache[signature]:
                continue
            kernel = triton.compile(str(ir), options={"num_warps": warps})
            errors = [check(kernel, mode, x, vector) for _ in range(args.verify_repeats)]
            error = max(errors)
            for trial, obs in enumerate(errors):
                checks.writerow([phase, rows, cols, mode, schedule, warps, trial, obs, obs <= 3e-6])
            if error > 3e-6:
                if phase == "holdout" and key == cache[signature]:
                    raise ValueError(f"Cached {key} failed held-out validation (max error {error:g})")
                continue
            # Large common-mode directions expose cancellation in the JVP.
            # The reverse case uses the same vector as an output cotangent.
            common_error = check(kernel, mode, x, vector + 2**16)
            if (phase == "holdout" and key == cache[signature] and args.max_common_error is not None
                    and common_error > args.max_common_error):
                raise ValueError(f"Cached {key} exceeds the held-out common-mode budget ({common_error:g})")
            kernels[key] = kernel
            metadata[key] = [schedule, warps, reductions, kernel.n_regs, error, common_error]

    if not kernels:
        raise ValueError(f"No correct candidates for {signature}")
    samples = defaultdict(list)
    for key, trial, us in measure(kernels, x, vector, mode, args.slots, args.repeats):
        samples[key].append((trial, us))
    if phase == "calibrate":
        eligible = [k for k in samples if args.max_common_error is None or
                    metadata[k][-1] <= args.max_common_error]
        if not eligible:
            raise ValueError(f"No schedule meets the common-mode error budget for {signature}")
        cache[signature] = min(eligible, key=lambda k: statistics.median(us for _, us in samples[k]))
    for key, timings in samples.items():
        for trial, us in timings:
            writer.writerow([phase, rows, cols, mode, *metadata[key], key == cache[signature], trial, us])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibrate-rows", type=int, default=512)
    parser.add_argument("--holdout-rows", type=int, default=2048)
    parser.add_argument("--cols", type=int, nargs="+", default=[127, 512, 1023, 4096])
    parser.add_argument("--warps", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--verify-repeats", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--max-common-error", type=float, help="Also gate calibration on the large-offset seed")
    parser.add_argument("--center-seed", action="store_true",
                        help="Include tensor softmax derivatives with and without first-column seed centering")
    parser.add_argument("--pivot-seed", action="store_true",
                        help="Also try centering at the first valid column with the largest logit")
    parser.add_argument("--replay", type=Path, help="Use a calibration from an earlier run")
    args = parser.parse_args()
    args.opt = args.opt.resolve()
    if min(args.cols + args.warps + [args.repeats, args.verify_repeats, args.slots, args.calibrate_rows, args.holdout_rows]) <= 0:
        parser.error("Rows, columns, repeats, slots and warp counts must be positive")
    if args.calibrate_rows == args.holdout_rows:
        parser.error("Calibration and holdout must use different row counts")
    args.output.mkdir(parents=True, exist_ok=False)
    target = fingerprint(args.opt, args.center_seed or args.pivot_seed)
    cache = {}
    phases = [("calibrate", args.calibrate_rows), ("holdout", args.holdout_rows)]
    config = {"target": target, "calibrate_rows": args.calibrate_rows, "warps": args.warps,
              "slots": args.slots, "dtype": "float32", "max_common_error": args.max_common_error,
              "verify_repeats": args.verify_repeats}
    if args.pivot_seed:
        config["pivot_seed"] = True
    if args.replay:
        record = json.loads(args.replay.read_text())
        if record["config"] != config:
            raise ValueError("The saved compiler, device or calibration settings differ")
        cache = record["schedules"]
        phases = phases[1:]
    with ((args.output / "timings.csv").open("w", newline="") as handle,
          (args.output / "checks.csv").open("w", newline="") as validation):
        checks = csv.writer(validation, lineterminator="\n")
        checks.writerow(["phase", "rows", "cols", "mode", "schedule", "warps", "trial", "max_abs_error", "passed"])
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["phase", "rows", "cols", "mode", "schedule", "warps", "reductions",
                         "registers", "max_abs_error", "common_mode_error", "selected", "trial", "us"])
        for phase, rows in phases:
            for cols in args.cols:
                for mode in ("ForwardMode", "ReverseModeCombined"):
                    experiment(args, phase, rows, cols, mode, cache, writer, checks)
                    handle.flush()
                    validation.flush()
            if phase == "calibrate":
                (args.output / "cache.json").write_text(json.dumps({"config": config, "schedules": cache}, indent=2)+"\n")


if __name__ == "__main__":
    main()
