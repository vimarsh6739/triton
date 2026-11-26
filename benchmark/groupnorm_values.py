"""Link an Enzyme-generated value VJP to the GroupNorm tile's memory operations."""

import functools
from pathlib import Path
import re
import tempfile

import torch
import triton
import triton.language as tl
from enzyme_ad.jax import enzyme_call
from triton._C.libtriton import ir
from triton.compiler import ASTSource
from triton.compiler.compiler import make_backend
from triton.experimental.autodiff import _find_tt_func, _make_public
from triton.runtime.driver import driver


@triton.jit
def vjp_signature(x, w, b, seed):
    # Only the signature is used. The linker installs Enzyme's generated body.
    return x, w, b


@triton.jit
def value_entry(X, W, B, Seed, DX, DW, DB, GROUPS: tl.constexpr, K: tl.constexpr, H: tl.constexpr,
                CK: tl.constexpr, SH: tl.constexpr):
    g = tl.program_id(0)
    c = tl.arange(0, CK)
    s = tl.arange(0, SH)
    mask = (c[:, None] < K) & (s[None, :] < H)
    i = g * K * H + c[:, None] * H + s[None, :]
    ci = (g % GROUPS) * K + c
    x = tl.load(X + i, mask, 0).to(tl.float32)
    w = tl.load(W + ci, c < K, 0).to(tl.float32)
    b = tl.load(B + ci, c < K, 0).to(tl.float32)
    seed = tl.load(Seed + i, mask, 0).to(tl.float32)
    dx, dw, db = vjp_signature(x, w, b, seed)
    # Every input element has one owner. Parameter gradients are shared across batches.
    tl.store(DX + i, dx, mask)
    tl.atomic_add(DW + ci, dw, c < K, sem="relaxed")
    tl.atomic_add(DB + ci, db, c < K, sem="relaxed")


def lower(fn, pointer_type, constants, target):
    backend = make_backend(target)
    opts = backend.parse_options({})
    ctx = ir.context()
    ir.load_dialects(ctx)
    backend.load_dialects(ctx)
    signature = {key: pointer_type for key in fn.arg_names if key not in constants}
    source = ASTSource(fn, signature=signature, constexprs=constants)
    return source.make_ir(target, opts, backend.get_codegen_implementation(opts),
                          backend.get_module_map(), ctx).str_nodebug()


def name_callee(module, helper, name):
    symbols = re.findall(r'tt\.call (@(?:"[^"]+"|[\w.$-]+))\(', module)
    matches = {sym for sym in symbols if sym.strip('@"').split('.')[-1].startswith(helper + "__")}
    if len(matches) != 1:
        raise ValueError(f"Expected one {helper} callee, found {matches}")
    return module.replace(matches.pop() + "(", "@" + name + "(")


@functools.lru_cache(None)
def value_kernel(dtype, groups, k, h, eps, device):
    from groupnorm_pullback import group_primal

    target = driver.active.get_current_target()
    ptr = {torch.float32: "*fp32", torch.bfloat16: "*bf16"}[dtype]
    shape = dict(GROUPS=groups, K=k, H=h, CK=triton.next_power_of_2(k), SH=triton.next_power_of_2(h))
    forward = lower(group_primal, ptr, dict(**shape, EPS=eps), target)
    wrapper = lower(value_entry, ptr, shape, target)
    core_name, adj_name = "group_math", "group_vjp"
    forward = name_callee(forward, "normalized_affine", core_name)
    wrapper = name_callee(wrapper, "vjp_signature", adj_name)
    core = _find_tt_func(forward, core_name)
    # Keep the value function available while inlining tl.sum and tl.rsqrt.
    forward = forward.replace(core, _make_public(core), 1)
    active = "enzyme_active,enzyme_active,enzyme_active"
    wrap = f"enzyme-wrap{{infn={core_name} outfn={adj_name} argTys={active} retTys=enzyme_active mode=ReverseModeCombined}}"
    _, derivative = enzyme_call.run_pass_pipeline([], forward, "inline,canonicalize," + wrap +
                                                  ",canonicalize,remove-unnecessary-enzyme-ops,canonicalize,cse,strip-debuginfo")
    entry = _find_tt_func(wrapper, value_entry.__name__)
    adjoint = _find_tt_func(derivative, adj_name)
    locations = "\n".join(line for line in derivative.splitlines() if line.startswith("#loc"))
    module = locations + "\nmodule {\n" + entry + "\n" + adjoint + "\n}"
    _, linked = enzyme_call.run_pass_pipeline([], module, "inline,canonicalize,cse,symbol-dce")
    if "tt.call" in linked:
        raise ValueError("Expected a single inlined entry")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "groupnorm_vjp.ttir"
        path.write_text(linked)
        return triton.compile(str(path), target=target, options={"num_warps": 4})
