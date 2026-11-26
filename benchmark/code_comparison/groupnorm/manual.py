import torch
import triton
import triton.language as tl

from common import GroupNorm, group_indices


@triton.jit
def backward_kernel(X, W, Stats, DY, DX, DW, DB, GROUPS: tl.constexpr, K: tl.constexpr,
                    H: tl.constexpr, CK: tl.constexpr, SH: tl.constexpr):
    offset, valid, ci, active = group_indices(GROUPS, K, H, CK, SH)
    x = tl.load(X + offset, valid, 0).to(tl.float32)
    dy = tl.load(DY + offset, valid, 0).to(tl.float32)
    w = tl.load(W + ci, active, 0).to(tl.float32)
    group = tl.program_id(0)
    mean = tl.load(Stats + 2 * group)
    inv = tl.load(Stats + 2 * group + 1)
    normalized = (x - mean) * inv
    dw = tl.sum(dy * normalized, 1)
    db = tl.sum(dy, 1)
    c1 = tl.sum(dw * w, 0) / (K * H)
    c2 = tl.sum(db * w, 0) / (K * H)
    dx = (dy * w[:, None] - normalized * c1 - c2) * inv
    tl.store(DX + offset, dx, valid)
    tl.atomic_add(DW + ci, dw, active, sem="relaxed")
    tl.atomic_add(DB + ci, db, active, sem="relaxed")


def pullback(ctx, x, w, b, stats, dy, dw, db):
    dx = torch.empty_like(x)
    backward_kernel[ctx.grid](x, w, stats, dy, dx, dw, db, **ctx.opts)
    return dx


def groupnorm(x, w, b, groups, eps):
    return GroupNorm.apply(x, w, b, groups, eps, pullback, True)
