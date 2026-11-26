import triton
import triton.language as tl


@triton.jit
def factor_row(table, index, rows, lanes, mask, RANK: tl.constexpr):
    ids = tl.load(index + rows, mask, other=0)
    offsets = ids[:, None] * RANK + lanes[None, :]
    return offsets, tl.load(table + offsets, mask[:, None], other=0)


@triton.jit
def squared_loss(U, V, W, I, J, K, ratings, losses, N: tl.constexpr, RANK: tl.constexpr,
                 BLOCK: tl.constexpr, RETURN_STATE: tl.constexpr = False):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lanes = tl.arange(0, RANK)
    mask = rows < N
    iu, u = factor_row(U, I, rows, lanes, mask, RANK)
    iv, v = factor_row(V, J, rows, lanes, mask, RANK)
    iw, w = factor_row(W, K, rows, lanes, mask, RANK)
    residual = tl.sum(u * v * w, axis=1) - tl.load(ratings + rows, mask, other=0)
    tl.store(losses + rows, 0.5 * residual * residual, mask)
    if RETURN_STATE:
        return rows, mask, iu, iv, iw, u, v, w, residual


def launch(kernel, args, ratings, rank):
    grid = (triton.cdiv(ratings.numel(), 16),)
    return kernel[grid](*args, N=ratings.numel(), RANK=rank, BLOCK=16, num_warps=1)
