# Forward-mode autodiff runtime benchmarks

This directory contains standalone Triton runtime benchmarks based on the
workloads in tutorials 01 through 06. They own their kernel definitions and do
not import, parse, or modify anything under `python/tutorials/`.

The benchmark validates both the primal result and its forward derivative
against PyTorch references. Run one case from the Triton repository root with:

```bash
mamba activate triton-dev
export PYTHONPATH="$PWD/python"
export TRITON_ENZYME_OPT="$PWD/../Enzyme-JAX/bazel-bin/enzymexlamlir-opt"
export TRITON_ALWAYS_COMPILE=1

python benchmark/01-vector-add.py
```

The other workloads are directly runnable in the same way:

```bash
python benchmark/02-fused-softmax.py
python benchmark/03-matrix-multiplication.py
python benchmark/04-low-memory-dropout.py
python benchmark/05-layer-norm.py
python benchmark/06-fused-attention.py
```

Each script differentiates its kernel with `triton.experimental.fwddiff` and
validates both primal and tangent outputs against a PyTorch reference. Matmul
and attention use fixed launch configurations so fwddiff receives a
`JITFunction` directly rather than an autotuner wrapper.
