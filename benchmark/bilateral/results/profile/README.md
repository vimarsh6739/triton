# Which kernels run on the 4 MP image?

The PyTorch compiler fused much of the math around interpolation. Eager dispatched 36 forward and 35 backward kernels. Compilation reduced those counts to four and nine. Both versions still dispatched the same native 3-D grid-sampling backward. In the three compiled traces that single device launch accounted for 92.1–93.7% of summed backward GPU activity.

| 2,048 × 2,048 on RTX 5090 | Forward launches | Backward launches | Dominant backward kernel | Time for that launch across three traces |
| --- | ---: | ---: | --- | ---: |
| Eager PyTorch | 36 kernels | 35 kernels + 3 copies | `grid_sampler_3d_backward_kernel` | 12.98–16.34 ms |
| Compiled PyTorch | 4 kernels | 9 kernels | `grid_sampler_3d_backward_kernel` | 12.86–16.05 ms |
| TTIR AD | 1 kernel | 4 kernels + 1 copy | `revdiffe_slice` | 3.65–3.83 ms |

The PyTorch program samples all 12 affine coefficients at every pixel, then applies them to RGB. The sampling result alone has 50,331,648 FP32 values (192 MiB). Its native CUDA backward computes gradients in all three sampling coordinates. The Triton primal computes the final RGB in the same tile, with only the depth coordinate depending on the learned guide. Its generated pullback performs interpolation and affine derivatives in one launch, alongside three PyTorch zero-fill kernels and one device-to-device seed copy. These traces localize the difference to the interpolation and derivative kernels. Separating the cost of memory traffic, coordinate calculations and atomic contention would require additional experiments. [PyTorch implementation](../../torch_impl.py) · [ATen 3-D backward](https://github.com/pytorch/pytorch/blob/0d62256a2b23365f8e1604297eb23a6545102aa8/aten/src/ATen/native/cuda/GridSampler.cu#L632-L720) · [Triton implementation](../../triton_ad.py)

The profiler uses the 4 MP pretrained Local Laplacian guide and grid and seed 17. Each method first runs three complete warm-up calls. We then capture three forward and backward pairs, tracing each phase separately using CUPTI. Here the host calls each PyTorch function directly. The primary runtime figure uses complete CUDA-graph replays timed without the profiler. The activity table contains device timestamps ordered within each phase and one identifier per demangled symbol. Its durations are instrumented measurements. [Device activities](activities.csv) · [Kernel names](symbols.csv) · [Environment](environment.json)

Run from the bilateral benchmark directory with its CUDA environment active, CUPTI available and the learned tensors already extracted.

```sh
TORCHINDUCTOR_COMPILE_THREADS=1 python profile_ops.py /tmp/hdrnet-input.png /tmp/hdrnet-trained/2048.npz /tmp/hdrnet-device-profile
```
