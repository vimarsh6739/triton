# Coverage on RTX 5090

The six existing Python examples ran from frontend `46f4934ff8` with fresh compilation. Their generated forward kernels checked both primal and tangent outputs. We also ran every checked-in `RUN` pipeline from the six Enzyme-JAX `diffrules/triton/benchmarks` files at `a56c4856e`. Both the driver and FileCheck exited successfully on all 14 pipelines.

| Example | Runtime input | Maximum primal error | Maximum JVP error | Forward/reverse IR checks |
| --- | --- | ---: | ---: | --- |
| Vector add | 98,432 elements | 0 | 0 | 1 / 1 passed |
| Softmax | 13 × 781 | 3.73e-9 | 5.59e-9 | 1 / 1 passed |
| Matmul | 32 × 64 times 64 × 32 | 9.54e-7 | 4.77e-7 | 1 / 1 passed |
| Dropout | 777 elements, explicit mask and seeded RNG | 0 | 0 | 2 / 2 passed |
| Layer norm | 4 × 64 | 4.77e-7 | 5.96e-8 | 1 / 1 passed |
| Attention | Batch 4, 2 heads, length 128, head dim 64 | 3.05e-5 | 3.05e-5 | 1 / 1 passed |

The error columns report each script's reference comparison. Layer norm additionally checks mean and reciprocal-standard-deviation outputs. Attention checks the returned attention result and tangent. Dropout issued a deprecation warning about its integer mask. The six original GPU entry points exercise forward mode. Their reverse snapshot tests inspect compiled TTIR structure, so a paper claim about reverse execution needs separate numerical data.

Run the original scripts with `TRITON_ALWAYS_COMPILE=1` and a Triton-enabled Enzyme-JAX wheel. Use a scratch working directory because the attention example writes its TTIR to the current directory. These tests used ptxas 12.9.86 and the `enzymexlamlir-opt` binary with SHA-256 `eef40f3a80642f7538b9d79245ed0b6030180252aaabe73ba23deb514336e44d`. [Runtime logs](runtime.txt)

## Direct comparison with prior TTIR autodiff

Iaroslav Elistratov's implementation already differentiates TTIR. Its wrapper reads the compiled primal TTIR, runs a dedicated reverse transformation, compiles the resulting TTIR and exposes separate forward/backward launches to PyTorch autograd. We replayed published generated modules from commit `d8b850a082cc2b2da28364ad66e4ba81aa66b9fa` on the same local RTX 5090 backend. We also asked Enzyme to generate combined reverse TTIR directly from each identical archived primal. [Prior implementation](https://github.com/IaroslavElistratov/triton-autodiff/blob/d8b850a082cc2b2da28364ad66e4ba81aa66b9fa/third_party/autodiff/python/api.py)

| Published primal | Archived VJP | Enzyme VJP on identical primal |
| --- | --- | --- |
| Four-element add of a constant | Exact input gradients vs FP64 PyTorch | Exact input gradients |
| 16 × 16 dot, exactly representable dyadic inputs | Exact gradients for both matrices | Exact gradients |
| FP16 layer norm, four rows of 8,192, seed 17 | Maximum input-gradient error 0.10304 | Maximum input-gradient error 0.00571 |

Layer norm used the precision encoded in the archived module. We removed a diagnostic preamble before compiling its backward module. At both one and four warps, 7,588 of 32,768 archived input gradients exceeded `atol=.015, rtol=.002`; Enzyme passed for all three differentiable inputs. Weight and bias gradients passed for both implementations. This is one FP16 artifact and initialization. The prior repository's Python layer-norm example uses FP32. The comparison establishes a reproducible artifact-level difference; rerunning its full compiler on the same source and shapes is needed to compare current tool behavior. [Archived layer-norm module and example](https://github.com/IaroslavElistratov/triton-autodiff/tree/d8b850a082cc2b2da28364ad66e4ba81aa66b9fa/third_party/autodiff/test/layernorm)

```sh
git clone https://github.com/IaroslavElistratov/triton-autodiff.git /tmp/prior-triton-ad
git -C /tmp/prior-triton-ad checkout d8b850a082cc2b2da28364ad66e4ba81aa66b9fa
python replay_prior.py /tmp/prior-triton-ad /path/to/enzymexlamlir-opt add 2d_dot
python replay_prior.py /tmp/prior-triton-ad /path/to/enzymexlamlir-opt layernorm --warps 4
```

The last command exits with an assertion for the archived input-gradient discrepancy on this backend. Direct timing would need to include the prior method's two launches and Enzyme's combined launch, with matching auxiliary outputs, seed restoration, and gradient initialization.
