import triton
import triton.language as tl

import os
import pathlib
import hashlib
import pytest
import torch
from triton._internal_testing import is_cuda
from triton._instrumentation import instrument, is_enabled, register_instrumentation, unregister_instrumentation
from triton.experimental.autodiff import Const, Duplicated, _load_enzyme, fwddiff


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
    assert not hasattr(triton, "autodiff")


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


def test_rolling_update_algebraic_correction():
    import math
    pytest.importorskip("sympy")
    from triton.tools.rolling_update import derive_correction

    def symbol(name):
        return {"op": "symbol", "name": name}

    def op(name, *args):
        return {"op": name, "args": list(args)}

    def evaluate(expr, values):
        if expr["op"] == "symbol":
            return values[expr["name"]]
        if expr["op"] == "constant":
            return expr["value"]
        arguments = [evaluate(arg, values) for arg in expr["args"]]
        functions = {
            "add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y, "div": lambda x, y: x / y,
            "pow": lambda x, y: x**y, "exp": math.exp, "exp2": lambda x: 2.**x, "log": math.log
        }
        return functions[expr["op"]](*arguments)

    x, y, m, z = (symbol(name) for name in ("x", "y", "m", "z"))
    attention_term = op("mul", op("div", op("exp", op("sub", x, m)), z), y)
    repair = derive_correction(attention_term, ["m", "z"])
    assert repair is not None
    # Two independent terms must be repaired by the same accumulated-state map.
    old = {"m": 2., "z": 3.}
    new = {"m": 4., "z": 5.}
    payloads = [{"x": 1., "y": 7.}, {"x": 3., "y": -2.}]
    accumulator = sum(evaluate(attention_term, old | payload) for payload in payloads)
    repaired = evaluate(repair, {"acc": accumulator, "old_m": 2., "new_m": 4., "old_z": 3., "new_z": 5.})
    expected = sum(evaluate(attention_term, new | payload) for payload in payloads)
    assert repaired == pytest.approx(expected)

    square = op("mul", op("div", x, m), op("div", x, m))
    repair = derive_correction(square, ["m"])
    assert repair is not None
    assert evaluate(repair, {"acc": 3., "old_m": 2., "new_m": 4.}) == pytest.approx(0.75)
    # Raising old/new scales separately would overflow or underflow in FP32.
    for scale in (1.e-30, 1.e30):
        actual = evaluate(repair,
                          {"acc": torch.tensor(3.), "old_m": torch.tensor(scale), "new_m": torch.tensor(2. * scale)})
        torch.testing.assert_close(actual, torch.tensor(0.75))
    assert derive_correction(op("add", x, m), ["m"]) is None
    assert derive_correction(op("tanh", attention_term), ["m", "z"]) is None
    assert derive_correction(attention_term, ["m", "z"], reducer="max") is None


@pytest.mark.parametrize("supported", [True, False])
def test_rolling_update_proof_protocol(supported):
    import json
    import subprocess
    import sys
    pytest.importorskip("sympy")

    expression = {"op": "div", "args": [{"op": "symbol", "name": "payload"}, {"op": "symbol", "name": "denominator"}]}
    if not supported:
        expression = {"op": "tanh", "args": [expression]}
    request = {"expression": expression, "parameters": ["denominator"], "reducer": "add"}
    process = subprocess.run([sys.executable, "-m", "triton.tools.rolling_update"], input=json.dumps(request),
                             text=True, capture_output=True, check=True)
    response = json.loads(process.stdout)
    assert response["ok"] is supported
    if supported:
        assert isinstance(response["correction"], dict)
        assert "op" in response["correction"]
    else:
        assert response["reason"]
        assert "correction" not in response


def _install_rolling_update_hook(fresh_knobs, tile_size):
    import sys
    from triton._C.libtriton import ir, passes

    pytest.importorskip("sympy")
    key = f"rolling-update:{tile_size}:" + pathlib.Path(__file__).read_text()
    digest = hashlib.sha256(key.encode()).hexdigest()

    def inspect_stages(self=None, stages=None, options=None, language=None, capability=None):
        if stages is None:
            return key, digest
        original_ttir = stages["ttir"]

        def rolling_ttir(src, metadata):
            module = original_ttir(src, metadata)
            pm = ir.pass_manager(module.context)
            passes.ttir.add_rolling_update(pm, tile_size, True, False, sys.executable)
            pm.run(module, "rolling_update")
            return module

        stages["ttir"] = rolling_ttir

    fresh_knobs.runtime.add_stages_inspection_hook = inspect_stages


@pytest.mark.skipif(not is_cuda(), reason="CUDA rolling-update correctness")
@pytest.mark.parametrize("mode", ["plain", "causal", "softcap"])
@pytest.mark.parametrize("tile_size,n,d", [(32, 113, 32), (64, 241, 64)])
def test_rolling_update_attention(fresh_knobs, fresh_triton_cache, mode, tile_size, n, d):
    _install_rolling_update_hook(fresh_knobs, tile_size)

    @triton.jit
    def attention(Q, K, V, O, M: tl.constexpr, N: tl.constexpr, D: tl.constexpr, FULL_N: tl.constexpr,
                  MODE: tl.constexpr, BM: tl.constexpr):
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        keys = tl.arange(0, FULL_N)
        dims = tl.arange(0, D)
        q = tl.load(Q + rows[:, None] * D + dims[None, :], rows[:, None] < M, 0)
        k = tl.load(K + keys[None, :] * D + dims[:, None], keys[None, :] < N, 0)
        scores = tl.dot(q, k, input_precision="ieee") * (D**-0.5)
        if MODE == "softcap":
            scores = 4. * tl.extra.libdevice.tanh(scores / 4.)
        valid = keys[None, :] < N
        if MODE == "causal":
            valid = valid & (rows[:, None] >= keys[None, :])
        scores = tl.where(valid, scores, -1.e6)
        weights = tl.exp(scores - tl.max(scores, 1)[:, None])
        weights = weights / tl.sum(weights, 1)[:, None]
        v = tl.load(V + keys[:, None] * D + dims[None, :], keys[:, None] < N, 0)
        result = tl.dot(weights, v, input_precision="ieee")
        tl.store(O + rows[:, None] * D + dims[None, :], result, rows[:, None] < M)

    torch.manual_seed(42)
    m, bm = 37, 16
    q = torch.randn(m, d, device="cuda", dtype=torch.float32)
    k = torch.randn(n, d, device="cuda", dtype=torch.float32)
    v = torch.randn_like(k)
    output = torch.empty_like(q)
    compiled = attention[(triton.cdiv(m, bm), )](q, k, v, output, m, n, d, triton.next_power_of_2(n), mode, bm,
                                                 num_warps=4)
    scores = (q @ k.T) * (d**-0.5)
    if mode == "softcap":
        scores = 4. * torch.tanh(scores / 4.)
    if mode == "causal":
        scores = scores.masked_fill(
            torch.arange(m, device="cuda")[:, None] < torch.arange(n, device="cuda")[None, :], -1.e6)
    expected = scores.softmax(-1) @ v
    torch.testing.assert_close(output, expected, atol=1.e-4, rtol=1.e-4)

    ttir = compiled.asm["ttir"]
    assert "scf.for" in ttir
    dots = [line for line in ttir.splitlines() if " = tt.dot " in line]
    assert len(dots) == 2
    assert f"tensor<{bm}x{d}xf32> * tensor<{d}x{tile_size}xf32> -> tensor<{bm}x{tile_size}xf32>" in dots[0]
    assert f"tensor<{bm}x{tile_size}xf32> * tensor<{tile_size}x{d}xf32> -> tensor<{bm}x{d}xf32>" in dots[1]


@pytest.mark.skipif(not is_cuda(), reason="CUDA rolling-update correctness")
@pytest.mark.parametrize("n,tile_size,zero_prefix", [(127, 32, False), (257, 64, False), (127, 32, True)])
def test_rolling_update_stable_l2(fresh_knobs, fresh_triton_cache, n, tile_size, zero_prefix):
    _install_rolling_update_hook(fresh_knobs, tile_size)

    @triton.jit
    def stable_l2(X, O, N: tl.constexpr, FULL_N: tl.constexpr):
        row = tl.program_id(0)
        columns = tl.arange(0, FULL_N)
        x = tl.load(X + row * N + columns, columns < N, 0)
        maximum = tl.max(tl.abs(x), 0)
        normalized = x / maximum
        norm = tl.sqrt(tl.sum(normalized * normalized, 0)) * maximum
        tl.store(O + row, norm)

    torch.manual_seed(43)
    x = torch.randn(7, n, device="cuda") * torch.logspace(-30, 30, 7, device="cuda")[:, None]
    if zero_prefix:
        x[:, :tile_size] = 0.
    output = torch.empty(7, device="cuda")
    compiled = stable_l2[(7, )](x, output, n, triton.next_power_of_2(n), num_warps=4)
    expected = x.double().square().sum(-1).sqrt().float()
    torch.testing.assert_close(output, expected, atol=0., rtol=2.e-5)
    ttir = compiled.asm["ttir"]
    assert "scf.for" in ttir
    assert ttir.count('"tt.reduce"') == 2
    assert f"(tensor<{tile_size}xf32>) -> f32" in ttir
    assert f"(tensor<{triton.next_power_of_2(n)}xf32>) -> f32" not in ttir


@pytest.mark.skipif(not is_cuda(), reason="CUDA rolling-update correctness")
def test_rolling_update_signed_normalization_zero_prefix(fresh_knobs, fresh_triton_cache):
    _install_rolling_update_hook(fresh_knobs, 32)

    @triton.jit
    def normalize(X, Y, O, N: tl.constexpr):
        columns = tl.arange(0, N)
        x = tl.load(X + columns)
        y = tl.load(Y + columns)
        denominator = tl.sum(x, 0)
        result = tl.sum((x / denominator) * y, 0)
        tl.store(O, result)

    x = torch.ones(128, device="cuda")
    x[1:64:2] = -1.
    y = torch.arange(1, 129, device="cuda", dtype=torch.float32)
    output = torch.empty(1, device="cuda")
    # The first two tiles have zero denominators but nonzero weighted sums.
    assert x[:64].sum().item() == 0.
    compiled = normalize[(1, )](x, y, output, 128, num_warps=4)
    expected = ((x / x.sum()) * y).sum().reshape(1)
    torch.testing.assert_close(output, expected, atol=1.e-4, rtol=1.e-4)
    assert "scf.for" in compiled.asm["ttir"]


@pytest.mark.skipif(not is_cuda(), reason="CUDA rolling-update correctness")
@pytest.mark.parametrize("all_masked", [False, True])
def test_rolling_update_masked_normalization(fresh_knobs, fresh_triton_cache, all_masked):
    _install_rolling_update_hook(fresh_knobs, 32)

    @triton.jit
    def normalize(S, V, O, N: tl.constexpr, FULL_N: tl.constexpr):
        columns = tl.arange(0, FULL_N)
        scores = tl.load(S + columns, columns < N, -float("inf"))
        values = tl.load(V + columns, columns < N, 0.)
        weight = tl.exp(scores - tl.max(scores, 0))
        probability = weight / tl.sum(weight, 0)
        tl.store(O, tl.sum(probability * values, 0))

    n = 113
    scores = torch.linspace(-1002., -998., n, device="cuda")
    scores[:32] = -torch.inf
    if all_masked:
        scores.fill_(-torch.inf)
    values = torch.linspace(-1., 2., n, device="cuda")
    output = torch.empty(1, device="cuda")
    compiled = normalize[(1, )](scores, values, output, n, 128, num_warps=4)
    # An entirely masked original softmax is NaN; the pass must retain that.
    expected = (scores.softmax(0) * values).sum().reshape(1)
    torch.testing.assert_close(output, expected, atol=1.e-4, rtol=1.e-4, equal_nan=True)
    assert "scf.for" in compiled.asm["ttir"]
