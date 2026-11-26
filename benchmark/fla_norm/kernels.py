"""Paired Q/K launches and a saved-state TTIR pullback for FLA normalization."""
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import triton
import triton.language as tl
from fla.modules.norm.l2norm import l2norm_bwd_kernel, l2norm_fwd_kernel
from triton._C.libtriton import ir
from triton.compiler import ASTSource
from triton.compiler.compiler import make_backend
from triton.experimental.autodiff import _find_tt_func

from fold import fold_zero_sums

# Use the exact upstream block kernels. T is rows flattened across tokens and heads.
fla_forward = l2norm_fwd_kernel.fn
fla_backward = l2norm_bwd_kernel.fn


@triton.jit
def paired_forward(q, k, yq, yk, rq, rk, eps, T, D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr):
    query = tl.program_id(1) == 0
    fla_forward(tl.where(query, q, k), tl.where(query, yq, yk), tl.where(query, rq, rk), eps, T, D, BD, BT)


@triton.jit
def paired_backward(yq, yk, rq, rk, dq, dk, gq, gk, eps, T, D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr):
    query = tl.program_id(1) == 0
    fla_backward(tl.where(query, yq, yk), tl.where(query, rq, rk), tl.where(query, dq, dk),
                 tl.where(query, gq, gk), eps, T, D, BD, BT)


@triton.jit
def normalized_perturbation(y, delta):
    # The normalized output for a perturbation in normalized coordinates.
    # At delta=0 the multiplier is exactly one, including eps regularization.
    correction = tl.sum(delta * (2 * y + delta), axis=1)
    return (y + delta) * tl.rsqrt(1.0 + correction)[:, None]


@triton.jit
def saved_kernel(yq, yk, rq, rk, dq, dk, gq, gk, T: tl.constexpr, D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr):
    query = tl.program_id(1) == 0
    rows = tl.program_id(0).to(tl.int64) * BT + tl.arange(0, BT)
    cols = tl.arange(0, BD)
    offsets = rows[:, None] * D + cols[None, :]
    mask = (rows[:, None] < T) & (cols[None, :] < D)
    y = tl.load(tl.where(query, yq, yk) + offsets, mask, other=0.).to(tl.float32)
    r = tl.load(tl.where(query, rq, rk) + rows, rows < T, other=0.)
    seed = tl.load(tl.where(query, dq, dk) + offsets, mask, other=0.).to(tl.float32)
    # The builder replaces this private function body with its VJP at delta=0.
    g = normalized_perturbation(y, seed) * r[:, None]
    tl.store(tl.where(query, gq, gk) + offsets, g, mask)


@lru_cache(None)
def saved_pullback(rows, width, dtype, bt, warps, opt, finite, fold=False):
    """Generate and inline the tile VJP, leaving address selection in Triton."""
    bd = triton.next_power_of_2(width)
    target = triton.runtime.driver.active.get_current_target()
    backend = make_backend(target)
    options = backend.parse_options({'num_warps': warps})
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    signature = {arg: '*' + ('fp32' if arg in ('rq', 'rk') else dtype)
                 for arg in saved_kernel.arg_names[:8]}
    constants = dict(T=rows, D=width, BD=bd, BT=bt)
    signature.update({arg: 'constexpr' for arg in constants})
    source = ASTSource(saved_kernel, signature, constexprs=constants)
    primal = source.make_ir(target, options, backend.get_codegen_implementation(options),
                            backend.get_module_map(), ctx).str_nodebug()
    name, = re.findall(r'tt.func private @([^ (]*normalized_perturbation[^ (]*)\(', primal)

    def run(module, *passes):
        return subprocess.check_output([opt, *passes], input=module, text=True)

    public = primal.replace('tt.func private @' + name, 'tt.func public @' + name)
    local = _find_tt_func(run(public, '--inline', '--canonicalize', '--cse'), name)
    if finite:
        schedule = Path(__file__).with_name('finite.mlir').read_text()
        scheduled = 'module attributes {transform.with_named_sequence} {\n' + local + schedule + '\n}'
        local = _find_tt_func(run(scheduled, '--transform-interpreter'), name)
    wrap = f'infn={name} outfn=pullback mode=ReverseModeCombined argTys=enzyme_const,enzyme_active retTys=enzyme_active'
    derivative = _find_tt_func(run(local, '--enzyme-wrap=' + wrap, '--canonicalize',
                                   '--remove-unnecessary-enzyme-ops', '--cse'), 'pullback')
    tile = f'tensor<{bt}x{bd}xf32>'
    body = f"""tt.func private @{name}(%y: {tile}, %seed: {tile}) -> {tile} {{
      %zero = arith.constant dense<0.0> : {tile}
      %g = tt.call @pullback(%y, %zero, %seed) : ({tile}, {tile}, {tile}) -> {tile}
      tt.return %g : {tile}
    }}"""
    main = _find_tt_func(primal, 'saved_kernel')
    linked = run('module {\n' + '\n'.join([main, body, derivative]) + '\n}', '--inline', '--canonicalize', '--cse',
                 '--symbol-dce')
    if fold:
        rewritten = fold_zero_sums(run(linked, '--mlir-print-op-generic'))
        linked = run(rewritten, '--canonicalize', '--cse')
    with TemporaryDirectory() as scratch:
        path = Path(scratch) / 'saved_kernel.ttir'
        path.write_text(linked)
        return triton.compile(str(path), target=target, options={'num_warps': warps})
