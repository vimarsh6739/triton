"""Attribute one warmed bilateral-slice call to CUPTI GPU activities, phase by phase."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from bench import get_methods
from inputs import make_inputs


def trace(fn, path):
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        output = fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(path))
    with path.open() as stream:
        events = json.load(stream)["traceEvents"]
    activities = [ev for ev in events if ev.get("ph") == "X"
                  and ev.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    if not any(ev["cat"] == "kernel" for ev in activities):
        raise RuntimeError("CUPTI recorded no device kernels")
    return output, sorted(activities, key=lambda ev: ev["ts"])


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("activations", type=Path, help="2048.npz from pretrained/extract.py")
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--traces", type=int, default=3)
    args = parser.parse_args()
    if args.traces < 1:
        parser.error("traces must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.empty(1, device="cuda")
    with np.load(args.activations) as arrays:
        grid, guide = (torch.from_numpy(arrays[key]).cuda().contiguous().requires_grad_()
                       for key in ("grid", "guide"))
    _, height, width = guide.shape
    _, _, image, seed = make_inputs(args.image, height, width, args.seed)
    inputs = grid, guide, image
    names = {}
    records = []
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "activities.json"
        for method, fn in get_methods(("torch", "compiled", "ttir")).items():
            for _ in range(3):
                torch.autograd.grad(fn(*inputs), inputs, seed)
            for sample in range(args.traces):
                output, forward = trace(lambda: fn(*inputs), path)
                _, backward = trace(lambda: torch.autograd.grad(output, inputs, seed), path)
                for phase, events in (("forward", forward), ("backward", backward)):
                    first = events[0]["ts"]
                    for index, ev in enumerate(events):
                        symbol = ev["name"]
                        key = hashlib.sha256(symbol.encode()).hexdigest()[:16]
                        names[key] = symbol
                        launch = ev.get("args", {})
                        records.append(dict(method=method, phase=phase, sample=sample, index=index,
                                            kind=ev["cat"], symbol=key,
                                            start_us=round(ev["ts"] - first, 3), duration_us=ev["dur"],
                                            grid=str(launch.get("grid", "")), block=str(launch.get("block", ""))))
    write_rows(args.output / "activities.csv", records)
    write_rows(args.output / "symbols.csv", [dict(symbol=k, name=v) for k, v in sorted(names.items())])
    device = torch.cuda.get_device_properties(0)
    metadata = dict(gpu=device.name, uuid=str(device.uuid), torch=torch.__version__, cuda=torch.version.cuda,
                    seed=args.seed, warmup_calls=3, traces=args.traces, height=height, width=width,
                    activations_sha256=hashlib.sha256(args.activations.read_bytes()).hexdigest())
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
