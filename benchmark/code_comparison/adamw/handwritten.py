import triton
import triton.language as tl

from .shared import B1, B2, BC1, BC2, LR, WD, launch, primal, store_three, update_tile


@triton.jit
def forward(P, G, M, V, P1, M1, V1, DP, DG, DM, DV, DP1, DM1, DV1, n, BLOCK: tl.constexpr):
    i, mask, g, m1, root, denom = update_tile(P, G, M, V, P1, M1, V1, n, BLOCK)
    dp, dg = tl.load(DP + i, mask, 0), tl.load(DG + i, mask, 0)
    dm, dv = tl.load(DM + i, mask, 0), tl.load(DV + i, mask, 0)
    dm1 = B1 * dm + (1 - B1) * dg
    dv1 = B2 * dv + 2 * (1 - B2) * g * dg
    droot = dv1 / (2 * BC2 * root)
    dp1 = (1 - LR * WD) * dp - (LR / BC1) * (dm1 - (m1 / denom) * droot) / denom
    store_three(DP1, DM1, DV1, i, mask, dp1, dm1, dv1)


@triton.jit
def accumulate(ptr, value, mask):
    tl.store(ptr, tl.load(ptr, mask, 0) + value, mask)


@triton.jit
def reverse(P, G, M, V, P1, M1, V1, DP, DG, DM, DV, DP1, DM1, DV1, n, BLOCK: tl.constexpr):
    i, mask, g, m1, root, denom = update_tile(P, G, M, V, P1, M1, V1, n, BLOCK)
    bp = tl.load(DP1 + i, mask, 0)
    bm = tl.load(DM1 + i, mask, 0) - (LR / BC1) * bp / denom
    bv = tl.load(DV1 + i, mask, 0) + (LR / BC1) * bp * (m1 / denom) / (denom * 2 * BC2 * root)
    accumulate(DP + i, (1 - LR * WD) * bp, mask)
    accumulate(DG + i, (1 - B1) * bm + 2 * (1 - B2) * g * bv, mask)
    accumulate(DM + i, B1 * bm, mask)
    accumulate(DV + i, B2 * bv, mask)
    store_three(DP1, DM1, DV1, i, mask, 0, 0, 0)


def apply(kernel, inputs, outputs, input_shadows, output_shadows, block, warps):
    n = inputs[0].numel()
    args = (*inputs, *outputs, *input_shadows, *output_shadows, n)
    return launch(kernel, args, n, block, warps)


def jvp(inputs, outputs, input_shadows, output_shadows):
    return apply(forward, inputs, outputs, input_shadows, output_shadows, 128, 8)


def vjp(inputs, outputs, input_shadows, output_shadows):
    return apply(reverse, inputs, outputs, input_shadows, output_shadows, 128, 4)
