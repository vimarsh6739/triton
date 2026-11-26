"""Evaluate the known limit of first-column centering using a low-probability outlier."""

import argparse
import csv
from pathlib import Path

import torch
import triton


def measure(compiled, mode, x, u):
    y = torch.full_like(x, float("nan"))
    # Start at nonzero input adjoints to check that the VJP accumulates them.
    initial = torch.full_like(x, 0.125)
    if mode == "ForwardMode":
        dx, dy = u.clone(), torch.full_like(y, float("nan"))
    else:
        dx, dy = initial.clone(), u.clone()
    compiled[(x.shape[0], 1, 1)](x, dx, y, dy)
    p = x.double().softmax(-1)
    v = u.double()
    grad = p * (v - (p * v).sum(-1, keepdim=True))
    if mode == "ReverseModeCombined":
        assert torch.count_nonzero(dy) == 0
        grad += initial.double()
        actual = dx
    else:
        torch.testing.assert_close(dx, u, atol=0, rtol=0)
        actual = dy
    torch.testing.assert_close(y, p.float(), atol=2e-6, rtol=1e-5)
    return (actual.double() - grad).abs().max().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Directory of TTIR emitted by softmax.py --center-seed")
    parser.add_argument("output", type=Path)
    parser.add_argument("--pivot-seed", action="store_true")
    args = parser.parse_args()
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["cols", "mode", "schedule", "regime", "max_abs_error"])
        for directory in sorted(args.run.glob("cols-*")):
            cols = int(directory.name.removeprefix("cols-"))
            torch.manual_seed(13752 + cols)
            x = torch.randn((64, cols), device="cuda")
            u = torch.randn_like(x)
            signs = torch.ones_like(u[:, :1])
            signs[::2] = -1
            outlier_x, outlier_u = x.clone(), u.clone()
            outlier_x[:, 0] = -100
            outlier_u[:, 0] = 2**24
            cases = {"constant": (x, signs.expand_as(u).contiguous() * 65536),
                     "per-row-offset": (x, u + signs * 65536),
                     "first-column-outlier": (outlier_x, outlier_u)}
            schedules = ["tensor-control", "tensor-centered"]
            if args.pivot_seed:
                schedules.append("tensor-pivot")
                high_x, high_u = x.clone(), u.clone()
                largest = min(1, cols - 1)
                high_x[:, largest] = 20
                high_u[:, largest] = 2**24
                cases["max-logit-outlier"] = (high_x, high_u)
                if cols >= 2:
                    ties = x.clone()
                    first = min(1, cols - 2)
                    ties[:, first:first + 2] = 20
                    cases["tied-max-offset"] = (ties, u + signs * 65536)
            for mode in ("ForwardMode", "ReverseModeCombined"):
                for schedule in schedules:
                    kernel = triton.compile(str(directory / f"{mode}-{schedule}.ttir"), options={"num_warps": 4})
                    for label, (values, seed) in cases.items():
                        error = measure(kernel, mode, values, seed)
                        if schedule != "tensor-control" and label in ("constant", "per-row-offset", "tied-max-offset"):
                            assert error <= 1e-5, (cols, mode, label, error)
                        writer.writerow([cols, mode, schedule, label, error])


if __name__ == "__main__":
    main()
