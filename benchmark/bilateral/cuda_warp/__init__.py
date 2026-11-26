"""CUDA control with a group of four or sixteen cooperating lanes per pixel."""

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache
def kernels():
    load(name="bilateral_warp_cuda", sources=[str(Path(__file__).with_name("kernel.cu"))],
         extra_cuda_cflags=["-O3"], is_python_module=False)
    return torch.ops.bilateral_warp


class _Bilateral(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, guide, image, lanes, threads):
        n, gh, gw, depth, cout, cin = grid.shape
        _, h, w = guide.shape
        if (cout, cin) != (3, 4) or guide.shape != (n, h, w) or image.shape != (n, h, w, 3):
            raise ValueError("expected a 3-by-4 affine grid, scalar guide and RGB image")
        if not grid.is_cuda or min(n, gh, gw, depth, h, w) <= 0:
            raise ValueError("expected nonempty CUDA tensors")
        if any(t.dtype != torch.float32 or not t.is_contiguous() or t.device != grid.device
               for t in (grid, guide, image)):
            raise ValueError("expected contiguous FP32 tensors on one GPU")
        if max(grid.numel(), guide.numel() * lanes) > 2**31 - 1:
            raise ValueError("grid and thread indices must fit signed 32-bit indexing")
        if lanes not in (4, 16) or threads not in (64, 128, 256, 512):
            raise ValueError("unsupported cooperative launch configuration")
        ctx.save_for_backward(grid, guide, image)
        ctx.lanes, ctx.threads = lanes, threads
        forward = getattr(kernels(), f"forward{lanes}")
        return forward(grid, guide, image, threads)

    @staticmethod
    def backward(ctx, seed):
        backward = getattr(kernels(), f"backward{ctx.lanes}")
        return (*backward(*ctx.saved_tensors, seed, ctx.threads), None, None)


def slice_apply(grid, guide, image, lanes=16, threads=64):
    return _Bilateral.apply(grid, guide, image, lanes, threads)
