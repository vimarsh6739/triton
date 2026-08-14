#!/usr/bin/env python3
"""Forward-mode autodiff runtime benchmark for vector addition."""

import torch

import triton
import triton.experimental as texp
import triton.language as tl

from _common import assert_close, check_ir, run


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def benchmark():
    torch.manual_seed(100)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    dx = torch.full_like(x, 2.0)
    dy = torch.full_like(y, 3.0)
    output = torch.empty_like(x)
    doutput = torch.empty_like(x)
    grid = (triton.cdiv(size, 1024),)
    compiled = texp.fwddiff(add_kernel)[grid](
        texp.Duplicated(x, dx),
        texp.Duplicated(y, dy),
        texp.Duplicated(output, doutput),
        texp.Const(size),
        BLOCK_SIZE=1024,
    )
    torch.cuda.synchronize()

    check_ir(compiled, "add_kernel", ("arith.addf",))
    assert_close("primal", output, x + y, atol=0, rtol=0)
    assert_close("tangent", doutput, dx + dy, atol=0, rtol=0)


if __name__ == "__main__":
    run("vector_add", benchmark)
