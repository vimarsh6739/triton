#!/usr/bin/env python3
"""Forward-mode autodiff runtime benchmark for fused attention."""

import torch

import triton
import triton.experimental as texp
import triton.language as tl

from _common import assert_close, check_ir, run


DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    desc_k,
    desc_v,
    offset_y,
    dtype: tl.constexpr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    warp_specialize: tl.constexpr,
    IS_HOPPER: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo

    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.jit
def _attn_fwd(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    IS_HOPPER: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(
        desc_q,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(
            desc_v,
            shape=[HEAD_DIM, y_dim],
            strides=[N_CTX, 1],
            block_shape=[HEAD_DIM, BLOCK_N],
        )
    else:
        desc_v = _maybe_make_tensor_desc(
            desc_v,
            shape=[y_dim, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_N, HEAD_DIM],
        )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )
    desc_o = _maybe_make_tensor_desc(
        desc_o,
        shape=[y_dim, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_M, HEAD_DIM],
    )

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    q = desc_q.load([qo_offset_y, 0])
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            offset_y,
            dtype,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
            warp_specialize,
            IS_HOPPER,
        )
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            offset_y,
            dtype,
            start_m,
            qk_scale,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            N_CTX,
            warp_specialize,
            IS_HOPPER,
        )
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


def benchmark():
    torch.manual_seed(105)
    Z, H = 4, 2
    N_CTX, HEAD_DIM = 128, 64
    sm_scale = 0.5
    shape = (Z, H, N_CTX, HEAD_DIM)
    q = (torch.randn(shape, device=DEVICE, dtype=torch.float16) * 0.2).contiguous()
    k = (torch.randn(shape, device=DEVICE, dtype=torch.float16) * 0.2).contiguous()
    v = (torch.randn(shape, device=DEVICE, dtype=torch.float16) * 0.2).contiguous()
    dq = (torch.randn_like(q) * 0.1).contiguous()
    dk = (torch.randn_like(k) * 0.1).contiguous()
    dv = (torch.randn_like(v) * 0.1).contiguous()
    o, do = torch.empty_like(q), torch.zeros_like(q)
    softmax_lse = torch.empty((Z, H, N_CTX), device=DEVICE, dtype=torch.float32)
    dsoftmax_lse = torch.zeros_like(softmax_lse)

    def alloc_fn(size, align, stream):
        del align, stream
        return torch.empty(size, dtype=torch.int8, device=DEVICE)

    triton.set_allocator(alloc_fn)
    grid = (triton.cdiv(N_CTX, 64), Z * H, 1)
    compiled = texp.fwddiff(_attn_fwd, keep_temps=True)[grid](
        texp.Const(sm_scale),
        texp.Duplicated(softmax_lse, dsoftmax_lse),
        texp.Const(Z),
        texp.Const(H),
        texp.Duplicated(q, dq),
        texp.Duplicated(k, dk),
        texp.Duplicated(v, dv),
        texp.Duplicated(o, do),
        N_CTX=N_CTX,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=64,
        BLOCK_N=32,
        FP8_OUTPUT=False,
        STAGE=1,
        warp_specialize=False,
        IS_HOPPER=False,
        num_warps=4,
        num_stages=2,
    )
    torch.cuda.synchronize()

    # write the derivative of attention to a file
    with open("attn_fwd.mlir", "w") as f:
        f.write(compiled.asm["ttir"])

    def reference(q_arg, k_arg, v_arg):
        scores = torch.matmul(q_arg, k_arg.transpose(2, 3)) * sm_scale
        probs = torch.softmax(scores.float(), dim=-1).to(q_arg.dtype)
        return torch.matmul(probs, v_arg).half()

    ref_o, ref_do = torch.func.jvp(reference, (q, k, v), (dq, dk, dv))
    check_ir(compiled, "_attn_fwd", ("scf.for", "tt.dot"))
    assert_close("primal", o, ref_o, atol=1e-2)
    assert_close("tangent", do, ref_do, atol=3e-2, rtol=1e-2)


if __name__ == "__main__":
    run("attention", benchmark)
