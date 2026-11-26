"""Give the existing descriptor tutorial compile-time batch, head and sequence dimensions."""

import importlib

import triton
import triton.language as tl

_tutorial = importlib.import_module("06-fused-attention")._attn_fwd


@triton.jit
def attention(SCALE, LSE, Q, K, V, O, B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
              STAGE: tl.constexpr, QUERY_BLOCK: tl.constexpr = None):
    _tutorial(SCALE, LSE, B, H, Q, K, V, O, N, D, 64, 32, False, STAGE, False, False, QUERY_BLOCK=QUERY_BLOCK)
