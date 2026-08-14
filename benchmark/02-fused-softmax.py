#!/usr/bin/env python3
"""Forward-mode autodiff runtime benchmark for fused softmax."""

import torch

import triton
import triton.experimental as texp
import triton.language as tl

from _common import assert_close, check_ir, run


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols,
                   BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


def benchmark():
    torch.manual_seed(101)
    rows, cols, block_size = 13, 781, 1024
    x = torch.randn((rows, cols), device=DEVICE, dtype=torch.float32)
    dx = torch.randn_like(x)
    y = torch.empty_like(x)
    dy = torch.zeros_like(x)
    compiled = texp.fwddiff(softmax_kernel, keep_temps=True)[(3,)](
        texp.Duplicated(y, dy),
        texp.Duplicated(x, dx),
        texp.Const(x.stride(0)),
        texp.Const(y.stride(0)),
        texp.Const(rows),
        texp.Const(cols),
        BLOCK_SIZE=block_size,
        num_stages=2,
        num_warps=4,
    )
    torch.cuda.synchronize()

    ref_y = torch.softmax(x, dim=1)
    ref_dy = ref_y * (dx - (ref_y * dx).sum(dim=1, keepdim=True))
    check_ir(compiled, "softmax_kernel", ("scf.for", "tt.reduce"))
    assert_close("primal", y, ref_y, atol=2e-6, rtol=1e-5)
    assert_close("tangent", dy, ref_dy, atol=3e-6, rtol=1e-5)


if __name__ == "__main__":
    run("softmax", benchmark)
