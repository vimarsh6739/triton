"""Measure the guide pullback at learned bin centers in FP32 and FP64."""

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inputs import make_inputs


def guide_vjp(grid, guide, pixel, dy, y, x, size, dtype, positive_part):
    """Evaluate a single pixel directly using the published eight-corner stencil."""
    grid, pixel, dy = (v.to(dtype) for v in (grid, pixel, dy))
    g = guide.to(dtype).requires_grad_()
    gh, gw, depth = grid.shape[:3]
    gy = (y + 0.5) * gh / size - 0.5
    gx = (x + 0.5) * gw / size - 0.5
    z = g * depth - 0.5
    z0 = math.floor(z.item())
    output = 0
    for j in (math.floor(gy), math.floor(gy) + 1):
        for i in (math.floor(gx), math.floor(gx) + 1):
            for k in (z0, z0 + 1):
                coeff = grid[min(max(j, 0), gh-1), min(max(i, 0), gw-1), min(max(k, 0), depth-1)]
                rgb = coeff[:, :3] @ pixel + coeff[:, 3]
                wz = positive_part(1 - ((z - k).square() + 1e-8).sqrt())
                output = output + (1 - abs(gx-i)) * (1 - abs(gy-j)) * wz * rgb
    return torch.autograd.grad(output, g, dy)[0].item()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", type=Path)
    ap.add_argument("arrays", type=Path)
    ap.add_argument("--seed", type=int, default=51)
    args = ap.parse_args()
    with np.load(args.arrays) as archive:
        grid, guide = (torch.from_numpy(archive[k])[0] for k in ("grid", "guide"))
    size = guide.shape[0]
    _, _, image, dy = make_inputs(args.image, size, size, args.seed, device="cpu")
    z = guide * grid.shape[2] - 0.5
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["size", "seed", "y", "x", "guide", "fp32_inclusive", "fp32_strict", "fp64"])
    for y, x in (z == z.round()).nonzero().tolist():
        values = [guide_vjp(grid, guide[y, x], image[0, y, x].detach(), dy[0, y, x], y, x, size, dtype, fn)
                  for dtype, fn in [(torch.float32, lambda v: v.clamp_min(0)), (torch.float32, torch.relu),
                                    (torch.float64, torch.relu)]]
        writer.writerow([size, args.seed, y, x, guide[y, x].item(), *values])


if __name__ == "__main__":
    main()
