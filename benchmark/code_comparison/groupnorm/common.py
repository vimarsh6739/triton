import torch
import triton
import triton.language as tl


@triton.jit
def group_indices(GROUPS: tl.constexpr, K: tl.constexpr, H: tl.constexpr,
                  CK: tl.constexpr, SH: tl.constexpr):
    group = tl.program_id(0)
    channel = tl.arange(0, CK)
    space = tl.arange(0, SH)
    offset = group * K * H + channel[:, None] * H + space[None, :]
    valid = (channel[:, None] < K) & (space[None, :] < H)
    return offset, valid, (group % GROUPS) * K + channel, channel < K


@triton.jit
def forward_kernel(X, W, B, Y, Stats, GROUPS: tl.constexpr, K: tl.constexpr,
                   H: tl.constexpr, CK: tl.constexpr, SH: tl.constexpr,
                   EPS: tl.constexpr, KEEP_STATS: tl.constexpr):
    offset, valid, ci, active = group_indices(GROUPS, K, H, CK, SH)
    x = tl.load(X + offset, valid, 0).to(tl.float32)
    w = tl.load(W + ci, active, 0).to(tl.float32)
    b = tl.load(B + ci, active, 0).to(tl.float32)
    mean = tl.sum(tl.sum(x, 1), 0) / (K * H)
    var = tl.sum(tl.sum(x * x, 1), 0) / (K * H) - mean * mean
    inv = tl.rsqrt(var + EPS)
    tl.store(Y + offset, (x - mean) * inv * w[:, None] + b[:, None], valid)
    if KEEP_STATS:
        group = tl.program_id(0)
        tl.store(Stats + 2 * group, mean)
        tl.store(Stats + 2 * group + 1, inv)


class GroupNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, groups, eps, pullback, keep_stats):
        batch, channels, h = x.shape
        if groups <= 0 or channels % groups or min(batch, channels, h) <= 0:
            raise ValueError("expected nonempty inputs and equal-sized groups")
        if any(t.device != x.device or t.dtype != x.dtype or not t.is_contiguous() for t in (x, w, b)):
            raise ValueError("expected contiguous inputs on one device with one dtype")
        if not x.is_cuda or x.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("expected CUDA FP32 or BF16 inputs")
        if w.shape != (channels,) or b.shape != (channels,):
            raise ValueError("expected one scale and bias per channel")
        k = channels // groups
        if triton.next_power_of_2(k) * triton.next_power_of_2(h) > 65536:
            raise ValueError("the padded group must fit in one tile of at most 65536 elements")
        ctx.grid = (batch * groups,)
        ctx.opts = dict(GROUPS=groups, K=k, H=h, CK=triton.next_power_of_2(k),
                        SH=triton.next_power_of_2(h))
        ctx.eps, ctx.pullback = eps, pullback
        y = torch.empty_like(x)
        stats = x.new_empty((2 * batch * groups,), dtype=torch.float32) if keep_stats else None
        forward_kernel[ctx.grid](x, w, b, y, stats, EPS=eps, KEEP_STATS=keep_stats, **ctx.opts)
        ctx.save_for_backward(x, w, b, stats)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, stats = ctx.saved_tensors
        dw, db = torch.zeros_like(w), torch.zeros_like(b)
        dx = ctx.pullback(ctx, x, w, b, stats, dy.contiguous(), dw, db)
        return dx, dw, db, None, None, None, None
