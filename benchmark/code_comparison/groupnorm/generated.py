import torch
from triton.experimental import Const, Duplicated, autodiff

from common import GroupNorm, forward_kernel


def pullback(ctx, x, w, b, stats, dy, dw, db):
    dx = torch.zeros_like(x)
    y = torch.empty_like(x)
    autodiff(forward_kernel)[ctx.grid](
        Duplicated(x, dx), Duplicated(w, dw), Duplicated(b, db),
        Duplicated(y, dy.clone()), Const(x), EPS=ctx.eps, KEEP_STATS=False, **ctx.opts,
    )
    return dx


def groupnorm(x, w, b, groups, eps):
    return GroupNorm.apply(x, w, b, groups, eps, pullback, False)
