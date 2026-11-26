"""One out-of-place AdamW update and its hand-differentiated counterparts."""

import triton
import triton.language as tl


# Bias corrections correspond to the 32nd update of AdamW.
B1 = tl.constexpr(0.9)
B2 = tl.constexpr(0.999)
LR = tl.constexpr(3e-3)
WD = tl.constexpr(1e-2)
EPS = tl.constexpr(1e-8)
BC1 = tl.constexpr(1 - 0.9**32)
BC2 = tl.constexpr(1 - 0.999**32)


@triton.jit
def _update(p, g, m, v):
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g * g
    root = tl.sqrt(v / BC2)
    denom = root + EPS
    p = (1 - LR * WD) * p - (LR / BC1) * m / denom
    return p, m, v, root, denom


@triton.jit
def step(P, G, M, V, P1, M1, V1, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    p, g = tl.load(P + i, mask, 0), tl.load(G + i, mask, 0)
    m, v = tl.load(M + i, mask, 0), tl.load(V + i, mask, 1)
    p1, m1, v1, _, _ = _update(p, g, m, v)
    tl.store(P1 + i, p1, mask)
    tl.store(M1 + i, m1, mask)
    tl.store(V1 + i, v1, mask)


@triton.jit
def step_jvp(P, DP, G, DG, M, DM, V, DV, P1, DP1, M1, DM1, V1, DV1, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    p, g = tl.load(P + i, mask, 0), tl.load(G + i, mask, 0)
    m, v = tl.load(M + i, mask, 0), tl.load(V + i, mask, 1)
    dp, dg = tl.load(DP + i, mask, 0), tl.load(DG + i, mask, 0)
    dm, dv = tl.load(DM + i, mask, 0), tl.load(DV + i, mask, 0)
    p1, m1, v1, root, denom = _update(p, g, m, v)
    dm1 = B1 * dm + (1 - B1) * dg
    dv1 = B2 * dv + 2 * (1 - B2) * g * dg
    droot = dv1 / (2 * BC2 * root)
    dp1 = (1 - LR * WD) * dp - (LR / BC1) * (dm1 - (m1 / denom) * droot) / denom
    tl.store(P1 + i, p1, mask)
    tl.store(M1 + i, m1, mask)
    tl.store(V1 + i, v1, mask)
    tl.store(DP1 + i, dp1, mask)
    tl.store(DM1 + i, dm1, mask)
    tl.store(DV1 + i, dv1, mask)


@triton.jit
def _accumulate(ptr, value, mask, ATOMIC: tl.constexpr):
    if ATOMIC:
        tl.atomic_add(ptr, value, mask, sem="relaxed")
    else:
        tl.store(ptr, tl.load(ptr, mask, 0) + value, mask)


@triton.jit
def step_vjp(P, DP, G, DG, M, DM, V, DV, P1, DP1, M1, DM1, V1, DV1, N, BLOCK: tl.constexpr, ATOMIC: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    p, g = tl.load(P + i, mask, 0), tl.load(G + i, mask, 0)
    m, v = tl.load(M + i, mask, 0), tl.load(V + i, mask, 1)
    p1, m1, v1, root, denom = _update(p, g, m, v)
    bp = tl.load(DP1 + i, mask, 0)
    bm = tl.load(DM1 + i, mask, 0) - (LR / BC1) * bp / denom
    bv = tl.load(DV1 + i, mask, 0) + (LR / BC1) * bp * (m1 / denom) / (denom * 2 * BC2 * root)
    # This launch assigns a different contiguous segment to each CTA. Its seven
    # primal and seven shadow arguments have disjoint storage.
    _accumulate(DP + i, (1 - LR * WD) * bp, mask, ATOMIC)
    _accumulate(DG + i, (1 - B1) * bm + 2 * (1 - B2) * g * bv, mask, ATOMIC)
    _accumulate(DM + i, B1 * bm, mask, ATOMIC)
    _accumulate(DV + i, B2 * bv, mask, ATOMIC)
    tl.store(P1 + i, p1, mask)
    tl.store(M1 + i, m1, mask)
    tl.store(V1 + i, v1, mask)
    tl.store(DP1 + i, 0, mask)
    tl.store(DM1 + i, 0, mask)
    tl.store(DV1 + i, 0, mask)
