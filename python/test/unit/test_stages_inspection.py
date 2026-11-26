import triton
import triton.language as tl

import os
import pathlib
import hashlib
import importlib
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from triton._internal_testing import is_cuda
from triton._instrumentation import instrument, is_enabled, register_instrumentation, unregister_instrumentation
from triton.experimental.autodiff import (
    Const,
    Duplicated,
    TTIRTransform,
    _AutodiffKernel,
    _DiffSpec,
    _autodiff_pipeline,
    _build_enzyme_module,
    _enzyme_version,
    _load_enzyme,
    _pipeline_cache_key,
    autodiff,
    fwddiff,
)


def test_instrumentation_mode():
    options = {"instrumentation_mode": "gsan,consan"}

    assert is_enabled(options, "consan")
    assert not is_enabled(options, "proton")


def test_dynamic_instrumentation_and_dialect_loading():
    calls = []

    assert not instrument("context", point="load-dialects", backend="test")
    register_instrumentation(point="load-dialects", backend="test", callback=lambda context: calls.append(
        ("dialects", context)))
    register_instrumentation(point="ttgpuir-to-llvmir", backend="test", callback=lambda _pm: calls.append("passes"))
    try:
        assert instrument("context", point="load-dialects", backend="test")
        assert instrument("pm", point="ttgpuir-to-llvmir", backend="test", context="module-context")
        assert calls == [("dialects", "context"), ("dialects", "module-context"), "passes"]
    finally:
        unregister_instrumentation(point="load-dialects", backend="test")
        unregister_instrumentation(point="ttgpuir-to-llvmir", backend="test")

    assert not instrument("context", point="load-dialects", backend="test")


def _has_enzyme():
    try:
        _load_enzyme()
    except RuntimeError:
        return False
    return True


def test_fwddiff_is_experimental_api():
    assert not hasattr(triton, "fwddiff")
    assert not hasattr(triton, "revdiff")
    assert not hasattr(triton, "autodiff")
    assert not hasattr(triton.experimental, "revdiff")
    assert triton.experimental.autodiff is autodiff
    assert triton.experimental.fwddiff is fwddiff


@pytest.mark.parametrize("decorate, expected_mode, tensor_shape", [
    (autodiff, "ReverseModeCombined", None),
    (autodiff(), "ReverseModeCombined", None),
    (fwddiff, "ForwardMode", None),
    (fwddiff(tensor_shape=128), "ForwardMode", (128, )),
])
def test_autodiff_decorators(decorate, expected_mode, tensor_shape):

    @decorate
    @triton.jit
    def kernel(x):
        pass

    assert kernel._mode == expected_mode
    assert kernel._tensor_shape == tensor_shape
    assert isinstance(kernel.primal, triton.JITFunction)


@pytest.mark.parametrize("mode", ["forward", "fwd", "reverse", "rev", "ReverseMode", "invalid"])
def test_autodiff_rejects_mode_aliases(mode):

    @triton.jit
    def kernel(x):
        pass

    with pytest.raises(ValueError, match="Use 'ForwardMode' or 'ReverseModeCombined'"):
        autodiff(kernel, mode=mode)


@pytest.mark.parametrize("mode", ["ForwardMode", "ReverseModeCombined"])
def test_autodiff_mode_and_arguments(mode):

    @triton.jit
    def kernel(x, scale, output, BLOCK: tl.constexpr):
        pass

    x, dx, output, seed = (object() for _ in range(4))
    diff = autodiff(kernel, mode=mode)
    assert diff.primal is kernel
    assert diff._mode == mode
    args, kwargs, bound, flat, names, activities = diff._flatten_args(
        (Duplicated(x, dx), ), {"scale": Const(2.0), "output": Duplicated(output, seed), "BLOCK": 128, "num_warps": 4})
    assert args == (x, 2.0, output, 128)
    assert kwargs == {"num_warps": 4}
    assert bound == {"x": x, "scale": 2.0, "output": output, "BLOCK": 128}
    assert flat == {"x": (x, dx), "scale": (2.0, ), "output": (output, seed), "BLOCK": (128, )}
    assert names == ("x", "scale", "output", "BLOCK")
    assert activities == ("enzyme_dup", "enzyme_const", "enzyme_dup", "enzyme_const")


def test_reverse_autodiff_module_and_launcher_signature():
    spec = _DiffSpec("ReverseModeCombined", ("x", "scale", "out", "BLOCK"),
                     ("enzyme_dup", "enzyme_const", "enzyme_dup", "enzyme_const"), ("x", "scale", "out"),
                     "test-version", None, False)
    ttir = "module { tt.func public @kernel(%x: !tt.ptr<f32>, %scale: f32, %out: !tt.ptr<f32>) { tt.return } }"
    module, options = _build_enzyme_module(ttir, "kernel", ("*fp32", "fp32", "*fp32"), spec, spec.arg_activities[:3])
    assert module == ttir
    assert options == ("infn=kernel outfn=revdiffekernel argTys=enzyme_dup,enzyme_const,enzyme_dup "
                       "mode=ReverseModeCombined")
    assert _pipeline_cache_key(spec) != _pipeline_cache_key(replace(spec, mode="ForwardMode"))
    keys = {_pipeline_cache_key(s)[1] for s in (
        spec,
        replace(spec, pre_ad="cse"),
        replace(spec, post_ad="cse"),
        replace(spec, pre_ad=TTIRTransform("first")),
        replace(spec, pre_ad=TTIRTransform("second")),
    )}
    assert len(keys) == 5
    compiled = SimpleNamespace(src=SimpleNamespace(
        signature={"x": "*fp32", "scale": "fp32", "out": "*fp32", "BLOCK": "constexpr"}))
    for _ in range(2):
        names = _AutodiffKernel._patch_launcher_signature(compiled, dict(zip(spec.arg_names, spec.arg_activities)),
                                                          spec.runtime_arg_names)
        assert names == spec.runtime_arg_names
        assert compiled.src.signature == {0: "*fp32", 1: "*fp32", 2: "fp32", 3: "*fp32", 4: "*fp32"}


def test_reverse_autodiff_rejects_active_scalar():
    spec = _DiffSpec("ReverseModeCombined", ("x", ), ("enzyme_dup", ), ("x", ), "test-version", None, False)
    with pytest.raises(TypeError, match="Use Const for scalar arguments"):
        _build_enzyme_module("", "kernel", ("fp32", ), spec, spec.arg_activities)
    with pytest.raises(RuntimeError, match="activity/signature mismatch"):
        _build_enzyme_module("", "kernel", ("*fp32", ), spec, ())


def test_reverse_autodiff_launch_arguments(monkeypatch, fresh_knobs):
    ad = importlib.import_module("triton.experimental.autodiff")
    from triton.runtime.driver import driver

    @triton.jit
    def kernel(x, scale, output, stride, BLOCK: tl.constexpr):
        pass

    x, dx, output, seed = (object() for _ in range(4))
    launches = []
    compiled = SimpleNamespace(
        src=SimpleNamespace(
            signature={"x": "*fp32", "scale": "fp32", "output": "*fp32", "stride": "constexpr", "BLOCK": "constexpr"}),
        function="function", packed_metadata="metadata", launch_metadata=lambda *args: None,
        run=lambda *args: launches.append(args))
    binder = lambda *args, **kwargs: (None, [("*fp32", None), ("fp32", None), ("*fp32", None), ("constexpr", 1),
                                             ("constexpr", 128)], None)
    monkeypatch.setattr(kernel, "device_caches", {0: (binder, )})
    monkeypatch.setattr(driver, "_active",
                        SimpleNamespace(get_current_device=lambda: 0, get_current_stream=lambda device: 42))
    monkeypatch.setattr(ad, "_load_enzyme", lambda: None)
    monkeypatch.setattr(ad, "_enzyme_version", lambda: "test-version")
    original_hook = fresh_knobs.runtime.add_stages_inspection_hook

    def compile_kernel(*args, **kwargs):
        assert args == (x, 2.0, output, 1, 128)
        assert kwargs["warmup"]
        assert fresh_knobs.runtime.add_stages_inspection_hook is not original_hook
        return compiled

    monkeypatch.setattr(kernel, "run", compile_kernel)
    diff = autodiff(kernel)
    grid = lambda meta: (meta["BLOCK"] // 128, )
    args = (Duplicated(x, dx), Const(2.0), Duplicated(output, seed), Const(1))
    assert diff.warmup(*args, grid=grid, BLOCK=128) is compiled
    assert not launches
    for _ in range(2):
        diff[grid](*args, BLOCK=128)
        assert launches[-1][:6] == (1, 1, 1, 42, "function", "metadata")
        assert launches[-1][9:] == (x, dx, 2.0, output, seed)
        assert fresh_knobs.runtime.add_stages_inspection_hook is original_hook
    with pytest.raises(TypeError, match="cannot differentiate constexpr argument 'stride'"):
        diff[grid](*args[:3], Duplicated(1, 1), BLOCK=128)


@pytest.mark.skipif(not _has_enzyme(), reason="Enzyme-JAX is required for Triton autodiff")
@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("mode", ["ForwardMode", "ReverseModeCombined"])
def test_autodiff_compile(mode, scheduled, fresh_triton_cache):
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    @triton.jit
    def scaled_square(x, scale, output, n_elements, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        value = tl.load(x + offsets, mask, other=0)
        tl.store(output + offsets, scale * value * value, mask)

    def annotate(op, attribute):
        return TTIRTransform(f"""module attributes {{transform.with_named_sequence}} {{
          transform.named_sequence @__transform_main(%root: !transform.any_op {{transform.readonly}}) {{
            %ops = transform.structured.match ops{{["{op}"]}} in %root : (!transform.any_op) -> !transform.any_op
            transform.annotate %ops "{attribute}" : !transform.any_op
            transform.yield
          }}
        }}""")

    pre = annotate("tt.load", "test.pre") if scheduled else None
    post = annotate("tt.store", "test.post") if scheduled else None
    diff = autodiff(scaled_square, mode=mode, pre_ad=pre, post_ad=post)
    spec = _DiffSpec(diff._mode, tuple(scaled_square.arg_names),
                     ("enzyme_dup", "enzyme_const", "enzyme_dup", "enzyme_const", "enzyme_const"),
                     ("x", "scale", "output", "n_elements"), _enzyme_version(), None, False, diff._pre_ad,
                     diff._post_ad)
    source = ASTSource(scaled_square, signature={"x": "*fp32", "scale": "fp32", "output": "*fp32", "n_elements": "i32"},
                       constexprs={"BLOCK": 128})
    with _autodiff_pipeline(spec):
        compiled = triton.compile(source, target=GPUTarget("cuda", 80, 32))
    ttir = compiled.asm["ttir"]
    if mode == "ReverseModeCombined":
        assert "tt.func public @revdiffescaled_square" in ttir
        assert "tt.atomic_rmw fadd" in ttir
        assert "atom.global" in compiled.asm["ptx"] or "red.global" in compiled.asm["ptx"]
    else:
        assert "tt.func public @fwddiffescaled_square" in ttir
    assert ("test.pre" in ttir) == scheduled
    assert ("test.post" in ttir) == scheduled
    assert ttir.count("tt.store") == 2
    assert "enzyme." not in ttir


@pytest.mark.skipif(not is_cuda(), reason="CUDA is required")
@pytest.mark.skipif(not _has_enzyme(), reason="Enzyme-JAX is required for Triton autodiff")
@pytest.mark.parametrize("constant_y", [False, True])
def test_reverse_autodiff_product(constant_y, fresh_triton_cache):
    device = triton.runtime.driver.active.get_active_torch_device()

    @triton.jit
    def product(x, y, output, stride, n_elements, BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        a = tl.load(x + offsets * stride, mask, other=0)
        b = tl.load(y + offsets * stride, mask, other=0)
        tl.store(output + offsets, a * b, mask)

    n_elements = 259
    x = torch.rand(384, device=device)
    y = torch.rand_like(x)
    dx = torch.full_like(x, 2.0)
    dy = torch.full_like(y, 3.0)
    output = torch.full_like(x, -1.0)
    seed = torch.rand_like(x)
    doutput = seed.clone()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK"]), )
    diff = autodiff(product)
    args = (Duplicated(x, dx), Const(y) if constant_y else Duplicated(y, dy), Duplicated(output, doutput), Const(1))
    compiled = diff.warmup(*args, grid=grid, n_elements=n_elements, BLOCK=128)
    assert "revdiffeproduct" in compiled.asm["ttir"]
    for launch in range(1, 3):
        doutput.copy_(seed)
        diff[grid](*args, n_elements=n_elements, BLOCK=128)
        torch.testing.assert_close(output[:n_elements], (x * y)[:n_elements])
        torch.testing.assert_close(dx[:n_elements], (2.0 + launch * y * seed)[:n_elements])
        if constant_y:
            torch.testing.assert_close(dy, torch.full_like(dy, 3.0))
        else:
            torch.testing.assert_close(dy[:n_elements], (3.0 + launch * x * seed)[:n_elements])
        torch.testing.assert_close(doutput[:n_elements], torch.zeros_like(doutput[:n_elements]))
        torch.testing.assert_close(doutput[n_elements:], seed[n_elements:])
        torch.testing.assert_close(dx[n_elements:], torch.full_like(dx[n_elements:], 2.0))
        torch.testing.assert_close(dy[n_elements:], torch.full_like(dy[n_elements:], 3.0))
        torch.testing.assert_close(output[n_elements:], torch.full_like(output[n_elements:], -1.0))


@pytest.mark.skipif(not is_cuda(), reason="only currently tested on CUDA")
def test_inspection(monkeypatch, fresh_knobs, tmp_path: pathlib.Path):
    stage_name = 'make_ttgir'
    curr_repro_path = tmp_path / ("repro_prefix." + stage_name + ".repro.mlir")
    repro_path = tmp_path / "repro_prefix"

    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_REPRODUCER_PATH", str(repro_path))

    inspect_stages_hook_called = False
    make_ttgir_wrapper_called = False

    def get_key():
        return pathlib.Path(__file__).read_text()

    def get_hash():
        return hashlib.sha256(get_key().encode('utf-8')).hexdigest()

    def inspect_stages_hook(self=None, stages=None, options=None, language=None, capability=None):
        if all(arg is None for arg in (stages, options, language, capability)):
            return get_key(), get_hash()
        nonlocal inspect_stages_hook_called
        inspect_stages_hook_called = True

        def make_ttgir_wrapper(src, metadata, options, capability):
            nonlocal make_ttgir_wrapper_called
            make_ttgir_wrapper_called = True
            return self.make_ttgir(src, metadata, options, capability)

        stages["ttgir"] = lambda src, metadata: make_ttgir_wrapper(src, metadata, options, capability)

    @triton.jit
    def k1():
        return

    @triton.jit
    def k2():
        return

    # Run once to get the clean/golden repro dump
    k1[(1, )]()
    assert not inspect_stages_hook_called and not make_ttgir_wrapper_called
    assert os.path.exists(curr_repro_path)
    golden_repro = curr_repro_path.read_text()
    curr_repro_path.unlink()

    # Setup hook and call again, check if hooks got called
    fresh_knobs.runtime.add_stages_inspection_hook = inspect_stages_hook
    k2[(1, )]()
    assert inspect_stages_hook_called and make_ttgir_wrapper_called
    assert os.path.exists(curr_repro_path)
    hook_repro = curr_repro_path.read_text()

    # Check that repros match
    assert golden_repro.replace('k1', 'dummy') == hook_repro.replace('k2', 'dummy')


@pytest.mark.skipif(not is_cuda(), reason="only currently tested on CUDA")
@pytest.mark.skipif(not _has_enzyme(), reason="Enzyme-JAX is required for Triton fwddiff")
def test_fwddiff_vector_add(fresh_triton_cache):
    device = triton.runtime.driver.active.get_active_torch_device()

    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, stride, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        strided_offsets = offsets * stride
        x = tl.load(x_ptr + strided_offsets, mask=mask)
        y = tl.load(y_ptr + strided_offsets, mask=mask)
        tl.store(output_ptr + strided_offsets, x + y, mask=mask)

    n_elements = 1024
    x = torch.rand(n_elements, device=device)
    y = torch.rand(n_elements, device=device)
    dx = torch.full_like(x, 2.0)
    dy = torch.full_like(y, 3.0)
    output = torch.empty_like(x)
    doutput = torch.empty_like(output)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    compiled = fwddiff(add_kernel).warmup(
        Duplicated(x, dx),
        Duplicated(y, dy),
        Duplicated(output, doutput),
        Const(1),
        n_elements=n_elements,
        BLOCK_SIZE=1024,
        grid=grid,
    )
    assert "fwddiffeadd_kernel" in compiled.asm["ttir"]
    assert compiled.asm["ttir"].count("tt.store") == 2

    fwddiff(add_kernel)[grid](
        Duplicated(x, dx),
        Duplicated(y, dy),
        Duplicated(output, doutput),
        Const(1),
        n_elements=n_elements,
        BLOCK_SIZE=1024,
    )

    torch.testing.assert_close(output, x + y)
    torch.testing.assert_close(doutput, dx + dy)


@pytest.mark.skipif(not is_cuda(), reason="CUDA is required")
@pytest.mark.skipif(not _has_enzyme(), reason="Enzyme-JAX is required for Triton autodiff")
def test_reverse_autodiff_projection_hoisting(fresh_triton_cache):
    @triton.jit
    def cell(w, x, out, decay, steps, BLOCK: tl.constexpr):
        idx = tl.arange(0, BLOCK)
        weight = tl.load(w + idx)
        feature = tl.load(x + idx)
        state = 0.0
        for _ in range(steps):
            state = decay * state + tl.sum(weight * feature, axis=0)
        tl.store(out, state)

    device = triton.runtime.driver.active.get_active_torch_device()
    w = torch.linspace(0.25, 1, 128, device=device)
    x = torch.linspace(-0.5, 1, 128, device=device)
    dw = torch.empty_like(w)
    out = torch.empty(1, device=device)
    seed = torch.empty_like(out)
    decay, steps = 0.875, 13
    factor = (1 - decay**steps) / (1 - decay)
    outputs = []
    for before in [None, "loop-invariant-code-motion", None]:
        dw.zero_()
        seed.fill_(-0.5)
        compiled = autodiff(cell, pre_ad=before, post_ad="cse")[(1,)](
            Duplicated(w, dw), Const(x), Duplicated(out, seed), Const(decay), Const(steps), BLOCK=128)
        torch.testing.assert_close(dw, -0.5 * factor * x)
        torch.testing.assert_close(out, (torch.dot(w, x) * factor).reshape(1))
        torch.testing.assert_close(seed, torch.zeros_like(seed))
        outputs.append(compiled)
    assert outputs[0] is outputs[2]
    assert outputs[0].hash != outputs[1].hash
    loops = [[line for line in kernel.asm["ttir"].splitlines() if "scf.for " in line] for kernel in outputs]
    assert "tensor<128xf32>" in loops[0][-1]
    assert "tensor<128xf32>" not in loops[1][-1]
