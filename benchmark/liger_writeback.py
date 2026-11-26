"""Specialize the memory ABI of the two pinned, elementwise Liger SwiGLU kernels."""

import functools
import os
import subprocess
import tempfile
from pathlib import Path

import torch
import triton
from jaxlib.mlir import ir
from jaxlib.triton import dialect as tt
from liger_kernel.ops import swiglu
from triton.experimental import Const, Duplicated, autodiff


def address(pointer, arguments):
    """Return a base argument and the SSA offsets/mapping on the path to it."""
    path = []
    while pointer not in arguments:
        op = pointer.owner
        if op.name not in ("tt.addptr", "tt.splat"):
            raise ValueError(f"Unexpected address producer {op.name}")
        path.append((op.name, tuple(op.operands[1:])))
        pointer = op.operands[0]
    return arguments.index(pointer), path


def specialize(module, mode):
    fn, = module.body.operations
    block, = fn.regions[0].blocks
    args = list(block.arguments)
    accesses = []
    ops = list(block.operations)
    barrier, = [op for op in ops if op.name == "gpu.barrier"]
    for op in ops:
        if isinstance(op, (tt.LoadOp, tt.StoreOp, tt.AtomicRMWOp)):
            base, offset = address(op.ptr, args)
            accesses.append((base, op, offset))
        elif op.name != "gpu.barrier" and not (
            op.name.startswith(("arith.", "math.")) or
            op.name in {"tt.addptr", "tt.splat", "tt.make_range", "tt.get_program_id", "tt.return"}
        ):
            raise ValueError(f"Unexpected operation {op.name}")

    # The launch uses Liger's own contiguous indexing, one program per row
    # (or row/tile pair). Each valid element has one writer. Check all six
    # pointers use identical address expressions and masks. A caller outside
    # these two specific launch patterns cannot infer uniqueness from this test.
    expected = {0: ["tt.load"], 1: ["tt.atomic_rmw"], 2: ["tt.load"],
                3: ["tt.atomic_rmw"], 4: ["tt.store"], 5: ["tt.load", "tt.store"]}
    if len(args) not in (8, 9) or any([op.name for idx, op, _ in accesses if idx == base] != names
                             for base, names in expected.items()):
        raise ValueError("Unexpected Liger reverse memory ABI")
    mask, path = accesses[0][1].mask, accesses[0][2]
    if any(offset != path or op.mask != mask for _, op, offset in accesses):
        raise ValueError("Each address must traverse the same tile under the same mask")

    access = {(idx, op.name): op for idx, op, _ in accesses}
    seed_load = access[5, "tt.load"]
    seed_clear = access[5, "tt.store"]
    if not ops.index(seed_clear) == ops.index(barrier) + 1 == ops.index(seed_load) + 2:
        raise ValueError("Expected a synchronized seed read followed by its clear")
    atomics = [access[b, "tt.atomic_rmw"] for b in (1, 3)]
    for op in atomics:
        if (int(op.atomic_rmw_op) != tt.RMWOp.FADD or int(op.sem) != tt.MemSemantic.RELAXED or
                list(op.result.uses) or ops.index(op) <= ops.index(seed_clear)):
            raise ValueError("Expected two final relaxed additions with unused results")

    if mode != "discard":
        seed_clear.erase()
        if mode != "donate":
            barrier.erase()  # All written buffers are disjoint from the reads.
    if mode in ("overwrite", "donate"):
        for op in atomics:
            tt.StoreOp(op.ptr, op.val, mask=op.mask, loc=op.location, ip=ir.InsertionPoint(op))
            op.erase()


@functools.lru_cache()
def compiled_backward(dtype, width, scale, device, mode):
    kernel = swiglu._swiglu_forward_kernel
    block, warps = swiglu.calculate_settings(width)
    if swiglu._should_tile(width):
        kernel = swiglu._swiglu_forward_kernel_tiled
        block, warps = swiglu._swiglu_tile_settings(width)

    # The CUDA allocations we accept have the same specialization as these
    # aligned dummy pointers. Only the specialization, not their data, is used.
    dummy = torch.empty(16, dtype=dtype, device=device)
    dup = Duplicated(dummy, dummy)
    vjp = autodiff(kernel).warmup(dup, dup, dup, Const(width), Const(scale), width,
                                  BLOCK_SIZE=block, num_warps=warps, grid=(1,))
    opt = os.environ["ENZYME_OPT"]
    def run_opt(source, *passes):
        return subprocess.check_output([opt, *passes], input=source, text=True)
    generic = run_opt(vjp.asm["ttir"], "--strip-debuginfo", "--mlir-print-op-generic")
    with ir.Context() as ctx:
        ctx.allow_unregistered_dialects = True
        tt.register_dialect(ctx)
        mod = ir.Module.parse(generic)
        specialize(mod, mode)
        discard = f"--discard-triton-output=function=revdiffe{kernel.__name__} argument=4"
        canonical = run_opt(mod.operation.get_asm(print_generic_op_form=True), discard, "--canonicalize", "--cse")
    if canonical.count("tt.store ") != {"discard": 1, "read_seed": 0, "overwrite": 2, "donate": 2}[mode]:
        raise ValueError("Unexpected memory writes after specialization")
    if (mode == "donate") != ("gpu.barrier" in canonical) and mode != "discard":
        raise ValueError("Input donation requires all CTA readers to finish before overwrite")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "liger_backward.ttir"
        path.write_text(canonical)
        return triton.compile(str(path), options={"num_warps": warps}), block


def backward(a, b, seed, scale, mode):
    """Run on dense storage. 'donate' consumes both input buffers, once."""
    rows, width = a.shape
    seed = seed.contiguous().view_as(a)
    if a.device != b.device or a.device != seed.device or a.dtype != b.dtype or a.dtype != seed.dtype:
        raise ValueError("Expected matching device and dtype")
    if not a.is_contiguous() or not b.is_contiguous() or b.shape != a.shape:
        raise ValueError("Expected contiguous, equal-size Liger inputs")
    if any(t.data_ptr() % 16 for t in (a, b, seed)):
        raise ValueError("Expected 16-byte aligned inputs and cotangent")
    if mode == "donate":
        buffers = {t.untyped_storage().data_ptr() for t in (a, b, seed)}
        if len(buffers) != 3:
            raise ValueError("Input donation requires distinct storage for a, b and seed")
        da, db = a, b
    elif mode == "overwrite":
        da, db = torch.empty_like(a), torch.empty_like(b)
    elif mode in ("discard", "read_seed"):
        da, db = torch.zeros_like(a), torch.zeros_like(b)
    else:
        raise ValueError(f"Unknown pullback mode {mode}")

    executable, block = compiled_backward(a.dtype, width, scale, a.device, mode)
    grid = (rows, triton.cdiv(width, block), 1) if swiglu._should_tile(width) else (rows, 1, 1)
    cotangent = seed.clone() if mode == "discard" else seed
    # Liger specializes n_cols only in the one-program-per-row implementation.
    runtime_width = (width,) if swiglu._should_tile(width) else ()
    executable[grid](a, da, b, db, seed, cotangent, width, scale, *runtime_width)
    if mode == "donate":
        torch.autograd.graph.increment_version((a, b))
    return da, db


class SpecializedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, scale, mode):
        ctx.shape, ctx.mode, ctx.scale = a.shape, mode, scale
        a, b, out = swiglu.swiglu_forward(a, b, scale)
        ctx.save_for_backward(a, b)
        return out

    @staticmethod
    def backward(ctx, seed):
        a, b = ctx.saved_tensors
        da, db = backward(a, b, seed, ctx.scale, ctx.mode)
        return da.view(ctx.shape), db.view(ctx.shape), None, None
