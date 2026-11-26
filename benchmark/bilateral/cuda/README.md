# HDRNet CUDA baseline

The four device kernels and their four depth-weight helpers come from the initial public HDRNet release, dated August 5, 2017. They implement the smoothed trilinear bilateral slice, affine RGB transform, and handwritten gradients for the grid, guide and input image. The device code in [hdrnet_kernels.cuh](hdrnet_kernels.cuh) is the corresponding source excerpt with trailing whitespace removed. The original license and copyright notice accompany it. [Pinned Google source](https://github.com/google/hdrnet/blob/e2f1833e0afe9781eaed942d4c9bb6db496f4bc0/hdrnet/ops/bilateral_slice.cu.cc#L224-L521)

The PyTorch entry is `cuda.slice_apply(grid, guide, image)`. It accepts contiguous CUDA FP32 tensors with shapes `(N, GH, GW, D, 3, 4)`, `(N, H, W)`, and `(N, H, W, 3)`. The C++ adapter launches the four original kernels on the current PyTorch stream, implements TensorFlow's grid-stride loop and returns freshly allocated tensors. It selects grid sizes from the element counts. A 768 × 1,024 held-out image with a 16 × 16 × 8 grid selected 64 threads per block. Shuffling five candidate sizes over eleven rounds gave median forward-plus-backward times of 5.57, 5.72, 5.67, 8.07 and 15.97 ms at 64, 128, 256, 512 and 1,024 threads. The 1 MP and 4 MP comparisons use the 64-thread setting. [Calibration samples](launch-sweep.csv) · [PyTorch entry](__init__.py) · [Native adapter](launch.cu)

The counting convention includes 258 nonblank, noncomment lines in the original device excerpt, 72 lines in the C++ launcher and 20 lines in the Python autograd binding. The calibration and build controls are separate. The CUDA release handles variable channel counts and optional offsets internally. This benchmark exercises its RGB-to-RGB affine path.

At a clamped depth edge, the historical grid gradient assigns weight one to the first or last depth bin. The forward's two smoothed interpolation weights can sum to slightly less than one. The one-pixel check puts a guide just below the first depth-bin center and measures this difference against the analytical FP64 result. CUDA returned 1.000000000 and FP64 gave 0.999900203 (absolute difference 9.98e−5). [Edge probe](check_edge.py)

From the parent bilateral benchmark directory, run with CUDA PyTorch, a matching NVCC and a writable PyTorch extension cache. With CUDA 12.9 on RTX 5090:

```sh
export CUDA_HOME=/path/to/cuda-12.9
export TORCH_CUDA_ARCH_LIST=12.0
export TORCH_EXTENSIONS_DIR=/tmp/hdrnet-cuda-cache
python -m cuda.check_edge
python -m cuda.calibrate --out /tmp/hdrnet-launch.csv
```

The optional compiler control fixes the original device functions' RGB channel counts and offset flag at compile time. It keeps their indexing, interpolation and grid-gradient gather and launches one forward and three backward kernels through the same adapter.

```sh
python cuda/specialize_rgb.py /tmp/hdrnet-variants/cuda_rgb
PYTHONPATH=/tmp/hdrnet-variants python -c 'import cuda_rgb; print(cuda_rgb.kernels())'
```

On the same second RTX 5090, specializing RGB reduced the 1 MP and 4 MP full-call times by about 1%. Both versions used the selected 64-thread launch. The benchmark checked RGB output and all three gradients twice on each shape and seed. Maximum relative L2 across 96 comparisons to FP64 was 4.02e−6. Each entry spans two separately measured seed medians in milliseconds. [Interleaved raw samples](rgb-timings.csv) · [FP64 checks](rgb-accuracy.csv)

| Image size | Original runtime channel counts | Compile-time RGB |
| --- | ---: | ---: |
| 768 × 1024 | 5.609–5.619 | 5.536–5.553 |
| 1024 × 1024 | 7.476–7.481 | 7.391–7.395 |
| 2048 × 2048 | 32.263–32.278 | 31.901–31.971 |

```sh
PYTHONPATH=/tmp/hdrnet-variants python bench.py /path/to/input.png /tmp/rgb-comparison \
    --methods cuda cuda_rgb --sizes 768x1024 1024x1024 2048x2048 --rounds 11
```

## A guide gradient at the depth-bin centers

The original guide derivative first rounds `sqrt(distance² + 1e−8)` to FP32 and tests whether it exceeds one. At distance exactly one this rounds to one and admits a nonzero derivative for a neighbor whose real-valued weight is zero. The RGB-specialized control retains this behavior. The optional CUDA correction changes only the guide derivative's weight helper. It keeps the caller's double-precision distance for the support test `abs(distance) <= sqrt(1 − 1e−8)` and uses the original FP32 slope within the support. The forward, coefficient-grid adjoint and image adjoint use the original code.

Changing the old comparison from `>` to `>=` alone would lose an adjacent valid derivative. At depth 8, the FP32 guide immediately below 0.0625 has distance −0.9999999701976776 to the left neighbor. Its exact smoothed weight is positive, yet its FP32 smoothed absolute value also rounds to one. The exact center and its float neighbors are exercised by [the guide probe](../check_knots.py).

```sh
python cuda/prepare_strict.py /tmp/hdrnet-variants/cuda_strict
PYTHONPATH=/tmp/hdrnet-variants python -c 'import cuda_strict; print(cuda_strict.kernels())'
```

On a 28-point probe containing every depth center and its adjacent FP32 values, the maximum guide-gradient difference from FP64 fell from 64 to 3.15e−6. The 2,048 × 2,048 trained guide contained three pixels at 0.3125. The original CUDA derivative ranged from −8.57 to 19.15 at those pixels across two output cotangents. The corrected CUDA and the independent FP64 implementation gave zero for all six. Across the entire trained 4 MP images the corrected guide-gradient relative L2 error was 1.14e−7 on both seeds. The separate one-pixel grid-adjoint approximation above is present in both CUDA versions.
