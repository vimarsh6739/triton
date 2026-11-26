# Differentiating an AdamW optimizer step

The same three calls are available from [generated.py](generated.py) and [handwritten.py](handwritten.py). `primal(inputs, outputs)` writes the updated parameter and two moments. `jvp(inputs, outputs, input_shadows, output_shadows)` also propagates directions through all four inputs. `vjp` takes the same arguments, accumulates the four input adjoints into their supplied buffers and consumes the three output seeds. Inputs are the parameter, minibatch gradient, first moment and second moment. All buffers are contiguous, disjoint FP32 arrays supplied by the caller. The update uses bias corrections for optimizer step 32.

Both implementations import [shared.py](shared.py). It contains the entire optimizer update, masks, output stores, launch helper and primal entry point. Both handwritten derivative kernels call the same update helper and reuse its intermediates. Their reverse accumulation uses ordinary loads and stores because each input element has a single writer. Enzyme receives the entire primal kernel with its buffers marked active. It generates both derivative kernels without an AdamW-specific pass. The count includes the public launch functions, imports, decorators and constants. Runtime allocation of the input, output and shadow buffers occurs in the measurement harness.

Run from the parent directory with reverse-enabled Triton, its matching Enzyme-JAX extension and CUDA PyTorch on PYTHONPATH.

```sh
python -m adamw.bench --output /tmp/adamw-code-comparison
```

The harness uses 4,194,307 FP32 elements, two random seeds, positive second moments and nonuniform, nonzero initial adjoints. It checks all outputs and output seed clearing against independent FP64 PyTorch derivatives before timing and after replay. It shuffles the six method/mode combinations for each round, warms ten rounds and records 41 GPU event samples per combination and seed. Each measurement covers one fused state-and-derivative kernel for JVP or VJP, or one state-only kernel for primal. Reverse buffers are restored inside the CUDA graph before the starting event. Python dispatch, buffer allocation, restoration and compilation are outside the reported times.

The launch configurations come from the [earlier eight-configuration sweep](https://github.com/vimarsh6739/triton/blob/9a4425e503/benchmark/results/adamw-sm120/README.md). Both JVPs use 128 elements and eight warps. The generated VJP uses 1,024 and eight, and the handwritten VJP uses 128 and four. These settings are fixed before measuring this refactored code.

The complete generated program uses 48 physical code lines (35 shared and 13 additional). The handwritten program uses 70 (35 shared and 35 additional). With AST statements as a second convention the totals are 45 and 64 respectively. These counts price all three calls together. The two derivative timing columns each describe an independent call.

| Fused GPU call | Generated, seed 17 / 67 (µs) | Handwritten, seed 17 / 67 (µs) |
| --- | ---: | ---: |
| Update + JVP | 142.18 / 143.36 | 144.48 / 146.43 |
| Update + VJP | 256.00 / 253.95 | 225.28 / 219.14 |

All 152 output checks passed, including all four accumulated input adjoints and all three consumed seeds. The largest absolute discrepancy from FP64 rounded to FP32 was 1.15e-5. The generated reverse kernel still accumulates gradients with atomics. These measurements use synthetic optimizer states with positive second moments on an RTX 5090. The source includes the same primal for both methods. [492 individual timing samples](timings.csv) · [Accuracy checks](checks.csv) · [Toolchain and device](environment.json)
