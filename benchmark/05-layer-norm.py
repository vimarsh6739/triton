#!/usr/bin/env python3
"""Forward-mode autodiff runtime benchmark for layer normalization."""

import torch
import torch.nn.functional as F

import triton
import triton.experimental as texp
import triton.language as tl

from _common import assert_close, check_ir, run


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _layer_norm_fwd_fused(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N

    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        tl.store(Y + cols, y, mask=mask)


def benchmark():
    torch.manual_seed(104)
    M, N, eps = 4, 64, 1e-5
    x = torch.randn((M, N), device=DEVICE, dtype=torch.float32)
    w = torch.randn(N, device=DEVICE, dtype=torch.float32)
    b = torch.randn(N, device=DEVICE, dtype=torch.float32)
    dx = torch.randn_like(x) * 0.1
    dw = torch.randn_like(w) * 0.1
    db = torch.randn_like(b) * 0.1
    y, dy = torch.empty_like(x), torch.zeros_like(x)
    mean, dmean = torch.empty(M, device=DEVICE), torch.zeros(M, device=DEVICE)
    rstd, drstd = torch.empty(M, device=DEVICE), torch.zeros(M, device=DEVICE)
    compiled = texp.fwddiff(_layer_norm_fwd_fused)[(M,)](
        texp.Duplicated(x, dx),
        texp.Duplicated(y, dy),
        texp.Duplicated(w, dw),
        texp.Duplicated(b, db),
        texp.Duplicated(mean, dmean),
        texp.Duplicated(rstd, drstd),
        texp.Const(x.stride(0)),
        texp.Const(N),
        texp.Const(eps),
        BLOCK_SIZE=N,
        num_warps=4,
        num_ctas=1,
    )
    torch.cuda.synchronize()

    def reference(x_arg, w_arg, b_arg):
        return F.layer_norm(x_arg, (N,), w_arg, b_arg, eps)

    ref_y, ref_dy = torch.func.jvp(reference, (x, w, b), (dx, dw, db))
    ref_mean = x.mean(dim=1)
    ref_dmean = dx.mean(dim=1)
    centered = x - ref_mean[:, None]
    dcentered = dx - ref_dmean[:, None]
    ref_var = (centered * centered).mean(dim=1)
    ref_dvar = (2 * centered * dcentered).mean(dim=1)
    ref_rstd = torch.rsqrt(ref_var + eps)
    ref_drstd = -0.5 * ref_dvar * (ref_var + eps).pow(-1.5)
    check_ir(compiled, "_layer_norm_fwd_fused", ("scf.for", "tt.reduce"))
    assert_close("primal", y, ref_y, atol=2e-5, rtol=1e-5)
    assert_close("tangent", dy, ref_dy, atol=5e-5, rtol=1e-5)
    assert_close("mean primal", mean, ref_mean, atol=2e-6, rtol=1e-5)
    assert_close("mean tangent", dmean, ref_dmean, atol=2e-6, rtol=1e-5)
    assert_close("rstd primal", rstd, ref_rstd, atol=2e-6, rtol=1e-5)
    assert_close("rstd tangent", drstd, ref_drstd, atol=5e-6, rtol=1e-5)


if __name__ == "__main__":
    run("layer_norm", benchmark)
