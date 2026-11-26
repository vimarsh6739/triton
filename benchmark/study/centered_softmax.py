"""Build the softmax derivative on a tensor, then pass its centered seed."""

import subprocess
from pathlib import Path
from string import Template

from triton.experimental.autodiff import _find_tt_func

TEMPLATES = Path(__file__).with_name("centered")


def compile_tensor_derivative(opt, cols, mode, centered, directory):
    """Emit and inline a typed call to Enzyme's generated function."""
    tile = 1 << (cols - 1).bit_length()
    t = f"tensor<{tile}xf32>"
    indices = f"tensor<{tile}xi32>"
    ptrs = f"tensor<{tile}x!tt.ptr<f32>>"
    forward = mode == "ForwardMode"
    activity = "enzyme_dup" if forward else "enzyme_active"
    primal = Template((TEMPLATES / "row.mlir.in").read_text()).substitute(tile=tile)
    wrap = f"infn=row outfn=derivative argTys={activity} retTys={activity} mode={mode}"
    generated = subprocess.check_output(
        [opt, "--detach-triton-softmax-shift", f"--enzyme-wrap={wrap}", "--canonicalize",
         "--remove-unnecessary-enzyme-ops", "--canonicalize", "--cse"], input=primal, text=True)
    source = "dx" if forward else "dy"
    pointers = []
    for arg in ("x", "dx", "y", "dy"):
        pointers.append(f"  %{arg}_base = tt.splat %{arg} : !tt.ptr<f32> -> {ptrs}")
        pointers.append(f"  %{arg}_ptr = tt.addptr %{arg}_base, %offsets : {ptrs}, {indices}")

    seed = "%raw_seed"
    center = ""
    if centered:
        center = f"""  %first = arith.constant dense<0> : tensor<1xi32>
  %lane_zero = tt.gather %raw_seed[%first] {{axis = 0 : i32}} : ({t}, tensor<1xi32>) -> tensor<1xf32>
  %anchor = tt.unsplat %lane_zero : tensor<1xf32>
  %anchors = tt.splat %anchor : f32 -> {t}
  %delta = arith.subf %raw_seed, %anchors : {t}
  %direction = arith.select %valid, %delta, %zero : tensor<{tile}xi1>, {t}"""
        seed = "%direction"

    if forward:
        differentiate = f"  %probability, %grad = tt.call @derivative(%logits, {seed}) : ({t}, {t}) -> ({t}, {t})"
        write_gradient = f"  tt.store %dy_ptr, %grad, %valid : {ptrs}"
    else:
        differentiate = (f"  %probability = tt.call @row(%logits) : ({t}) -> {t}\n"
                         f"  %grad = tt.call @derivative(%logits, {seed}) : ({t}, {t}) -> {t}")
        write_gradient = f"""  %initial = tt.load %dx_ptr, %valid, %zero : {ptrs}
  %accumulated = arith.addf %initial, %grad : {t}
  tt.store %dx_ptr, %accumulated, %valid : {ptrs}
  // A layout may replicate a seed load across warps before this clear.
  gpu.barrier
  tt.store %dy_ptr, %zero, %valid : {ptrs}"""
    kernel = Template((TEMPLATES / "entry.mlir.in").read_text()).substitute(
        cols=cols, tile=tile, indices=indices, vector=t, ptrs=ptrs,
        pointers="\n".join(pointers), source=source, center=center,
        differentiate=differentiate, write_gradient=write_gradient)
    # Copy whole function definitions, then let MLIR check and inline the calls.
    functions = [_find_tt_func(generated, name) for name in ("row", "derivative")]
    linked = "module {\n" + kernel + "\n" + "\n".join(functions) + "\n}\n"
    optimized = subprocess.check_output(
        [opt, "--inline", "--canonicalize", "--cse", "--symbol-dce"], input=linked, text=True)
    if "tt.call" in optimized or optimized.count("tt.func ") != 1:
        raise ValueError("Expected one fully inlined softmax kernel")
    name = "tensor-centered" if centered else "tensor-control"
    path = directory / f"{mode}-{name}.ttir"
    path.write_text(optimized)
    return path, optimized.count('"tt.reduce"')
