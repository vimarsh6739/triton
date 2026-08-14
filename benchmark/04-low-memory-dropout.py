#!/usr/bin/env python3
"""Forward-mode autodiff runtime benchmark for low-memory dropout."""

import torch

import triton
import triton.experimental as texp
import triton.language as tl

from _common import assert_close, check_ir, run


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _dropout(x_ptr, x_keep_ptr, output_ptr, n_elements, p, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def _seeded_dropout(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def benchmark():
    torch.manual_seed(103)
    n_elements, block_size, p = 777, 256, 0.35
    x = torch.randn(n_elements, device=DEVICE, dtype=torch.float32)
    dx = torch.randn_like(x)
    keep = (torch.rand(n_elements, device=DEVICE) > p).to(torch.int32)
    y = torch.empty_like(x)
    dy = torch.zeros_like(x)
    grid = (triton.cdiv(n_elements, block_size),)
    compiled = texp.fwddiff(_dropout)[grid](
        texp.Duplicated(x, dx),
        texp.Const(keep),
        texp.Duplicated(y, dy),
        texp.Const(n_elements),
        texp.Const(p),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    torch.cuda.synchronize()

    ref_y = torch.where(keep.bool(), x / (1 - p), 0.0)
    ref_dy = torch.where(keep.bool(), dx / (1 - p), 0.0)
    check_ir(compiled, "_dropout", ("arith.select",))
    assert_close("mask primal", y, ref_y, atol=0, rtol=0)
    assert_close("mask tangent", dy, ref_dy, atol=5e-7, rtol=1e-6)

    seed = 123
    seeded_primal = torch.empty_like(x)
    _seeded_dropout[grid](x, seeded_primal, n_elements, p, seed, BLOCK_SIZE=block_size, num_warps=4)
    seeded_y = torch.empty_like(x)
    seeded_dy = torch.zeros_like(x)
    seeded_compiled = texp.fwddiff(_seeded_dropout)[grid](
        texp.Duplicated(x, dx),
        texp.Duplicated(seeded_y, seeded_dy),
        texp.Const(n_elements),
        texp.Const(p),
        texp.Const(seed),
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    torch.cuda.synchronize()

    seeded_ref_dy = torch.where(seeded_primal != 0, dx / (1 - p), 0.0)
    check_ir(seeded_compiled, "_seeded_dropout")
    assert_close("seeded primal", seeded_y, seeded_primal, atol=0, rtol=0)
    assert_close("seeded tangent", seeded_dy, seeded_ref_dy, atol=5e-7, rtol=1e-6)


if __name__ == "__main__":
    run("dropout", benchmark)
