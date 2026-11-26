#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <algorithm>
#include <climits>
#include <cuda_runtime.h>

using int64 = int64_t;
#define CUDA_1D_KERNEL_LOOP(idx, count) \
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count; idx += blockDim.x * gridDim.x)
#include "hdrnet_kernels.cuh"
#undef CUDA_1D_KERNEL_LOOP

using at::Tensor;

static void check(const Tensor &grid, const Tensor &guide, const Tensor &image, int64_t threads) {
  TORCH_CHECK(threads > 0 && threads <= 1024, "threads must be in [1, 1024]");
  TORCH_CHECK(grid.dim() == 6 && guide.dim() == 3 && image.dim() == 4, "expected batched grid, guide and image");
  TORCH_CHECK(grid.size(0) == image.size(0) && guide.sizes() == image.sizes().slice(0, 3),
              "mismatching batch or spatial shape");
  TORCH_CHECK(image.size(-1) == 3 && grid.size(-2) == 3 && grid.size(-1) == 4, "expected RGB affine grid");
  for (const Tensor &t : {grid, guide, image}) {
    TORCH_CHECK(t.device() == grid.device() && t.is_cuda() && t.scalar_type() == at::kFloat && t.is_contiguous(),
                "expected contiguous CUDA FP32 tensors on one device");
    TORCH_CHECK(t.numel() > 0 && t.numel() < INT_MAX / 2, "unsupported shape");
  }
}

static int blocks(int64_t count, int64_t threads) {
  return std::min<int64_t>(65535, (count + threads - 1) / threads);
}

static Tensor forward(const Tensor &grid, const Tensor &guide, const Tensor &image, int64_t threads) {
  check(grid, guide, image, threads);
  c10::cuda::CUDAGuard device(grid.device());
  auto out = at::empty_like(image);
  BilateralSliceApplyKernel<<<blocks(out.numel(), threads), threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      out.numel(), grid.data_ptr<float>(), guide.data_ptr<float>(), image.data_ptr<float>(),
      image.size(0), image.size(1), image.size(2), grid.size(1), grid.size(2), grid.size(3), 3, 3, true,
      out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

static std::tuple<Tensor, Tensor, Tensor> backward(const Tensor &grid, const Tensor &guide, const Tensor &image,
                                                    const Tensor &grad, int64_t threads) {
  check(grid, guide, image, threads);
  TORCH_CHECK(grad.sizes() == image.sizes() && grad.options().type_equal(image.options()) &&
              grad.device() == image.device(), "invalid output cotangent");
  c10::cuda::CUDAGuard device(grid.device());
  const Tensor seed = grad.contiguous();
  auto dg = at::empty_like(grid), ds = at::empty_like(guide), di = at::empty_like(image);
  const auto stream = c10::cuda::getCurrentCUDAStream();
  const int b = image.size(0), h = image.size(1), w = image.size(2);
  const int gh = grid.size(1), gw = grid.size(2), gd = grid.size(3);
  const float *g = grid.data_ptr<float>(), *s = guide.data_ptr<float>(), *im = image.data_ptr<float>();
  const float *dy = seed.data_ptr<float>();
  BilateralSliceApplyGridGradKernel<<<blocks(dg.numel(), threads), threads, 0, stream>>>(
      dg.numel(), g, s, im, dy, b, h, w, gh, gw, gd, 3, 3, true, dg.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  BilateralSliceApplyGuideGradKernel<<<blocks(ds.numel(), threads), threads, 0, stream>>>(
      ds.numel(), g, s, im, dy, b, h, w, gh, gw, gd, 3, 3, true, ds.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  BilateralSliceApplyInputGradKernel<<<blocks(di.numel(), threads), threads, 0, stream>>>(
      di.numel(), g, s, im, dy, b, h, w, gh, gw, gd, 3, 3, true, di.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dg, ds, di};
}

TORCH_LIBRARY(hdrnet_bench, m) {
  m.def("forward(Tensor grid, Tensor guide, Tensor image, int threads) -> Tensor");
  m.def("backward(Tensor grid, Tensor guide, Tensor image, Tensor grad, int threads) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(hdrnet_bench, CUDA, m) {
  m.impl("forward", &forward);
  m.impl("backward", &backward);
}
