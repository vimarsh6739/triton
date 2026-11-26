# GroupNorm from one tiled forward

Both implementations normalize the same contiguous `(batch, channels, spatial)` tensor. A Triton program owns all channels and spatial positions of one group. The hand-written backward saves the mean and reciprocal standard deviation in FP32 during forward. It evaluates the gradient from those two values, stores each input gradient, and atomically adds the shared parameter gradients. The generated backward calls the public `autodiff` launcher on the same forward. It recomputes the statistics, allocates zeroed destinations, and clones the incoming cotangent because the reverse store rule consumes it.

The source to count for either method is [common.py](common.py) plus [manual.py](manual.py) or [generated.py](generated.py). Those files include the complete PyTorch autograd entry, imports, shape and grid setup, mask and offset calculation, allocations, launches, and differentiation request. Both versions reuse the index calculation. There is no external per-operator schedule. [benchmark.py](benchmark.py) contains the independent FP64 oracle, published Liger control, and timing harness.

The handwritten operator has 87 nonblank, noncomment source lines. The generated operator has 71. Each includes the same 58 lines of forward and autograd code. Their backward-specific files contribute 29 and 13 lines respectively. Counting Python AST statements gives 77 and 60. The two seeds gave the following median full-call times in microseconds.

| BF16 batch | Handwritten tile | Generated from tile | Upstream Liger |
| --- | ---: | ---: | ---: |
| 2 | 16.032–16.096 | 26.368–26.432 | 28.256–28.320 |
| 8 | 91.904–92.416 | 235.936–236.544 | 99.840–99.936 |

The direct TTIR pullback uses fewer lines and the matched handwritten tile is faster at both sizes. At batch two the direct pullback also beats upstream Liger. Across the 192 output and gradient comparisons, relative L2 error stayed below 0.00746 in BF16 and 1.12e−7 in FP32.

One timing replays forward plus `torch.autograd.grad` with a nonuniform output cotangent in a CUDA graph. It includes all captured allocations' GPU operations, initialization and seed copying. Python dispatch, compilation and graph setup occur before timing. For each seed we shuffle methods in each round and record 51 samples after five warmup rounds. The output and all three gradients are compared to PyTorch FP64 GroupNorm before capture and again after replay. [Timing samples](timings.csv) · [Accuracy checks](accuracy.csv)

The batch-two and batch-eight BF16 inputs have 4,096 channels, 1,024 groups and 512 spatial positions. We also test imperfect tiles with three channels and 47 positions in FP32 and BF16. The one-tile example covers contiguous three-dimensional training inputs with equal-sized groups. Upstream Liger also handles groups spanning multiple Triton tiles. Its original implementation runs as a separate performance reference. [Liger source](https://github.com/linkedin/Liger-Kernel/blob/cc29eda82744aece335c1ca4bd3ace5c9d768048/src/liger_kernel/ops/group_norm.py) · [Upstream benchmark](https://github.com/linkedin/Liger-Kernel/blob/cc29eda82744aece335c1ca4bd3ace5c9d768048/benchmark/scripts/benchmark_group_norm.py)

## Reproduce

Use reverse-enabled Triton, its compatible Enzyme-JAX extension with the seed synchronization change, CUDA PyTorch, and the pinned Liger source on `PYTHONPATH`. The recorded machine has an RTX 5090, driver 575.57.08 and PyTorch 2.12.0a0+git0d62256 with CUDA 12.9. The Triton frontend was 03bead828c7d. The loaded Enzyme-JAX extension had SHA-256 72451477afb2a5f4cda29409d3e17fa9eb667d62c401fb640f23e9d3cbc2e852. Run from this directory.

```sh
python benchmark.py --output /tmp/groupnorm-comparison
```
