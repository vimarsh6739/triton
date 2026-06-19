"""
Vector Addition
===============

In this tutorial, you will write a simple vector addition using Triton.

In doing so, you will learn about:

* The basic programming model of Triton.

* The `triton.jit` decorator, which is used to define Triton kernels.

* The best practices for validating and benchmarking your custom ops against native reference implementations.

"""

# %%
# Compute Kernel
# --------------

import os

import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
SKIP_BENCHMARK_ENV = "TRITON_VECTOR_ADD_SKIP_BENCHMARK"


@triton.fwddiff
@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM. @fwddiff asks Enzyme to generate the tangent store.
    tl.store(output_ptr + offsets, output, mask=mask)


# %%
# Let's also declare a helper function to (1) allocate the `z` tensor
# and (2) enqueue the above kernel with appropriate grid/block sizes:


def _prepare_add_launch(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    return output, n_elements, grid


def compile_add_kernel(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024):
    output, n_elements, grid = _prepare_add_launch(x, y)
    dx = torch.empty_like(x)
    dy = torch.empty_like(y)
    doutput = torch.empty_like(output)
    return add_kernel.warmup(
        triton.Duplicated(x, dx),
        triton.Duplicated(y, dy),
        triton.Duplicated(output, doutput),
        triton.Const(n_elements),
        BLOCK_SIZE=block_size,
        grid=grid,
    )


def print_add_kernel_ir(compiled_kernel, stage: str):
    stage = stage.lower()
    if stage not in compiled_kernel.asm:
        available = ", ".join(sorted(compiled_kernel.asm))
        raise ValueError(f"Unknown stage {stage!r}. Available stages: {available}")
    print(f"\n=== add_kernel {stage} ===")
    print(compiled_kernel.asm[stage])


def assert_fwddiff_ir(compiled_kernel):
    ttir = compiled_kernel.asm["ttir"]
    if "fwddiffeadd_kernel" not in ttir or "arith.addf" not in ttir or ttir.count("tt.store") < 2:
        raise AssertionError("@fwddiff did not produce the expected differentiated TTIR")


def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output, n_elements, grid = _prepare_add_launch(x, y)
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel.primal[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


def add_forward_diff(x: torch.Tensor, dx: torch.Tensor, y: torch.Tensor, dy: torch.Tensor):
    output, n_elements, grid = _prepare_add_launch(x, y)
    doutput = torch.empty_like(output)
    assert dx.device == DEVICE and dy.device == DEVICE and doutput.device == DEVICE
    add_kernel[grid](
        triton.Duplicated(x, dx),
        triton.Duplicated(y, dy),
        triton.Duplicated(output, doutput),
        triton.Const(n_elements),
        BLOCK_SIZE=1024,
    )
    return output, doutput


# %%
# We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:


def run_demo():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    dx = torch.full_like(x, 2.0)
    dy = torch.full_like(y, 3.0)

    compiled = compile_add_kernel(x, y)
    assert_fwddiff_ir(compiled)
    print_add_kernel_ir(compiled, "ttir")

    output_torch = x + y
    doutput_torch = dx + dy
    output_triton = add(x, y)
    output_diff_triton, doutput_triton = add_forward_diff(x, dx, y, dy)
    torch.testing.assert_close(output_triton, output_torch)
    torch.testing.assert_close(output_diff_triton, output_torch)
    torch.testing.assert_close(doutput_triton, doutput_torch)
    print(output_torch)
    print(output_diff_triton)
    print(doutput_triton)
    print(f'The maximum primal difference between torch and triton is '
          f'{torch.max(torch.abs(output_torch - output_diff_triton))}')
    print(f'The maximum tangent difference between torch and triton is '
          f'{torch.max(torch.abs(doutput_torch - doutput_triton))}')


# %%
# Seems like we're good to go!

# %%
# Benchmark
# ---------
#
# We can now benchmark our custom op on vectors of increasing sizes to get a sense of how it does relative to PyTorch.
# To make things easier, Triton has a set of built-in utilities that allow us to concisely plot the performance of our custom ops.
# for different problem sizes.


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:


def main():
    run_demo()
    if os.environ.get(SKIP_BENCHMARK_ENV) == "1":
        return
    benchmark.run(print_data=True, show_plots=True)


if __name__ == "__main__":
    main()
