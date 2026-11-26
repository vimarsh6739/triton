"""A dense FP32 AdamW step with separate output buffers, at optimizer step 32."""

import triton
import triton.language as tl


B1 = tl.constexpr(0.9)
B2 = tl.constexpr(0.999)
BC1 = tl.constexpr(1 - 0.9**32)
BC2 = tl.constexpr(1 - 0.999**32)
LR = tl.constexpr(3e-3)
WD = tl.constexpr(1e-2)
EPS = tl.constexpr(1e-8)


@triton.jit
def store_three(P, M, V, i, mask, p, m, v):
    tl.store(P + i, p, mask)
    tl.store(M + i, m, mask)
    tl.store(V + i, v, mask)


@triton.jit
def update_tile(P, G, M, V, P1, M1, V1, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    p, g = tl.load(P + i, mask, 0), tl.load(G + i, mask, 0)
    m, v = tl.load(M + i, mask, 0), tl.load(V + i, mask, 1)
    m1 = B1 * m + (1 - B1) * g
    v1 = B2 * v + (1 - B2) * g * g
    root = tl.sqrt(v1 / BC2)
    denom = root + EPS
    p1 = (1 - LR * WD) * p - (LR / BC1) * m1 / denom
    store_three(P1, M1, V1, i, mask, p1, m1, v1)
    return i, mask, g, m1, root, denom


@triton.jit
def step(P, G, M, V, P1, M1, V1, n, BLOCK: tl.constexpr):
    update_tile(P, G, M, V, P1, M1, V1, n, BLOCK)


def launch(kernel, args, n, block, warps):
    return kernel[(triton.cdiv(n, block),)](*args, BLOCK=block, num_warps=warps)


def primal(inputs, outputs):
    n = inputs[0].numel()
    return launch(step, (*inputs, *outputs, n), n, 128, 8)
