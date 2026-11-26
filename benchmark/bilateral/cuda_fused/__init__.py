"""Handwritten CUDA control using one thread per pixel and a fused scatter backward."""

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache
def kernels():
    load(name="bilateral_fused_cuda", sources=[str(Path(__file__).with_name("kernel.cu"))],
         extra_cuda_cflags=["-O3"], is_python_module=False)
    return torch.ops.bilateral_fused


class _Bilateral(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, guide, image):
        n, gh, gw, depth, cout, cin = grid.shape
        _, h, w = guide.shape
        if (cout, cin) != (3, 4) or guide.shape != (n, h, w) or image.shape != (n, h, w, 3):
            raise ValueError("expected a 3-by-4 affine grid, scalar guide and RGB image")
        if not grid.is_cuda or min(n, gh, gw, depth, h, w) <= 0:
            raise ValueError("expected nonempty CUDA tensors")
        if any(t.dtype != torch.float32 or not t.is_contiguous() or t.device != grid.device
               for t in (grid, guide, image)):
            raise ValueError("expected contiguous FP32 tensors on one GPU")
        if max(grid.numel(), image.numel()) > 2**31 - 1:
            raise ValueError("grid and image must fit signed 32-bit indexing")
        ctx.save_for_backward(grid, guide, image)
        return kernels().forward(grid, guide, image)

    @staticmethod
    def backward(ctx, seed):
        return tuple(kernels().backward(*ctx.saved_tensors, seed))


slice_apply = _Bilateral.apply
