import triton
import triton.language as tl

from shared import launch, squared_loss


@triton.jit
def pullback(U, V, W, I, J, K, ratings, losses, dU, dV, dW, seed,
             N: tl.constexpr, RANK: tl.constexpr, BLOCK: tl.constexpr):
    rows, mask, iu, iv, iw, u, v, w, residual = squared_loss(
        U, V, W, I, J, K, ratings, losses, N, RANK, BLOCK, RETURN_STATE=True)
    weight = (residual * tl.load(seed + rows, mask, other=0))[:, None]
    tl.atomic_add(dU + iu, weight * v * w, mask[:, None], sem="relaxed")
    tl.atomic_add(dV + iv, weight * u * w, mask[:, None], sem="relaxed")
    tl.atomic_add(dW + iw, weight * u * v, mask[:, None], sem="relaxed")
    tl.store(seed + rows, 0, mask)


def differentiate(factors, indices, ratings, losses, gradients, seed):
    args = (*factors, *indices, ratings, losses, *gradients, seed)
    return launch(pullback, args, ratings, factors[0].shape[1])
