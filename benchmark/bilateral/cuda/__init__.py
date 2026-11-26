"""PyTorch binding for HDRNet's handwritten CUDA bilateral slice-and-apply kernels."""

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache
def kernels():
    load(name="hdrnet_bilateral_bench", sources=[str(Path(__file__).with_name("launch.cu"))],
         extra_cuda_cflags=["-O3", "-ftz=true"], is_python_module=False)
    return torch.ops.hdrnet_bench


class SliceApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, guide, image, threads):
        ctx.save_for_backward(grid, guide, image)
        ctx.threads = threads
        return kernels().forward(grid, guide, image, threads)

    @staticmethod
    def backward(ctx, grad):
        return (*kernels().backward(*ctx.saved_tensors, grad, ctx.threads), None)


def slice_apply(grid, guide, image, threads=64):
    return SliceApply.apply(grid, guide, image, threads)
