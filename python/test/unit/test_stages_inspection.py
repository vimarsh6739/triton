import triton
import triton.language as tl

import os
import pathlib
import hashlib
import pytest
import torch
from triton._internal_testing import is_cuda
from triton.autodiff import _find_enzyme_opt


def _has_enzyme_opt():
    try:
        _find_enzyme_opt()
    except RuntimeError:
        return False
    return True


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
@pytest.mark.skipif(not _has_enzyme_opt(), reason="enzymexlamlir-opt is required for Triton fwddiff")
def test_fwddiff_vector_add(fresh_triton_cache):
    device = triton.runtime.driver.active.get_active_torch_device()

    @triton.fwddiff
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x + y, mask=mask)

    n_elements = 1024
    x = torch.rand(n_elements, device=device)
    y = torch.rand(n_elements, device=device)
    dx = torch.full_like(x, 2.0)
    dy = torch.full_like(y, 3.0)
    output = torch.empty_like(x)
    doutput = torch.empty_like(output)
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]), )

    compiled = add_kernel.warmup(
        triton.Duplicated(x, dx),
        triton.Duplicated(y, dy),
        triton.Duplicated(output, doutput),
        triton.Const(n_elements),
        BLOCK_SIZE=1024,
        grid=grid,
    )
    assert "fwddiffeadd_kernel" in compiled.asm["ttir"]
    assert compiled.asm["ttir"].count("tt.store") == 2

    add_kernel[grid](
        triton.Duplicated(x, dx),
        triton.Duplicated(y, dy),
        triton.Duplicated(output, doutput),
        triton.Const(n_elements),
        BLOCK_SIZE=1024,
    )

    torch.testing.assert_close(output, x + y)
    torch.testing.assert_close(doutput, dx + dy)
