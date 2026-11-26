"""Measure HDRNet's clamped depth-gradient convention on one pixel and one affine coefficient."""

import math

import torch
from cuda import slice_apply


def main():
    grid = torch.ones((1, 1, 1, 2, 3, 4), device="cuda", requires_grad=True)
    guide = torch.tensor([[[0.25 - 1e-7]]], device="cuda")
    image = torch.zeros((1, 1, 1, 3), device="cuda")
    y = slice_apply(grid, guide, image)
    dy = torch.zeros_like(y)
    dy[..., 0] = 1
    measured = torch.autograd.grad(y, grid, dy)[0][0, 0, 0, 0, 0, 3].item()
    z = guide.item() * 2
    # Both depth neighbors query the same clamped grid element.
    expected = sum(max(1 - math.sqrt((zz + 0.5 - z)**2 + 1e-8), 0) for zz in (-1, 0))
    print(f"CUDA {measured:.9f}, FP64 {expected:.9f}, absolute difference {abs(measured-expected):.3g}")
    assert measured == 1.0 and 9e-5 < measured - expected < 1.1e-4


if __name__ == "__main__":
    main()
