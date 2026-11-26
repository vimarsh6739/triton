#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <cuda_runtime.h>

using at::Tensor;

// One thread owns a pixel and its input/guide gradients. Grid entries are shared.
template <bool backward>
__global__ void slice_kernel(const float* grid, const float* guide, const float* image,
                             const float* seed, float* output, float* dgrid, float* dguide,
                             float* dimage, int pixels, int h, int w, int gh, int gw, int depth) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pixels) return;
  const float gx = ((p % w) + 0.5f) * (float(gw) / w);
  const float gy = (((p / w) % h) + 0.5f) * (float(gh) / h);
  const float gz = guide[p] * depth;
  const int x0 = __float2int_rd(gx - 0.5f);
  const int y0 = __float2int_rd(gy - 0.5f);
  const int z0 = __float2int_rd(gz - 0.5f);
  const float rgba[] = {image[p * 3], image[p * 3 + 1], image[p * 3 + 2], 1.0f};
  float rgb[3] = {}, input_grad[3] = {}, guide_grad = 0;

  #pragma unroll
  for (int corner = 0; corner < 8; ++corner) {
    const int x = x0 + ((corner >> 1) & 1);
    const int y = y0 + (corner >> 2);
    const int z = z0 + (corner & 1);
    const float xy = (1 - fabsf(x + 0.5f - gx)) * (1 - fabsf(y + 0.5f - gy));
    const float center = z + 0.5f, delta = center - gz;
    const float smooth = sqrtf(delta * delta + 1.0e-8f);
    const bool inside = (gz - (center - 1)) * ((center + 1) - gz) > 1e-8f;
    const float weight = inside ? xy * (1 - smooth) : 0;
    const int row = (p / (w * h)) * gh + min(max(y, 0), gh - 1);
    const int cell = (row * gw + min(max(x, 0), gw - 1)) * depth + min(max(z, 0), depth - 1);
    float local_grad = 0;
    #pragma unroll
    for (int out = 0; out < 3; ++out) {
      float affine = 0;
      #pragma unroll
      for (int in = 0; in < 4; ++in) {
        const int k = cell * 12 + out * 4 + in;
        const float coeff = grid[k];
        affine += coeff * rgba[in];
        if constexpr (backward) {
          const float d = seed[p * 3 + out] * weight;
          atomicAdd(dgrid + k, d * rgba[in]);
          if (in < 3) input_grad[in] += d * coeff;
        }
      }
      if constexpr (backward) local_grad += seed[p * 3 + out] * affine;
      else rgb[out] += weight * affine;
    }
    if constexpr (backward) {
      if (inside) guide_grad += local_grad * xy * (delta / smooth) * depth;
    }
  }
  #pragma unroll
  for (int c = 0; c < 3; ++c) {
    if constexpr (backward) dimage[p * 3 + c] = input_grad[c];
    else output[p * 3 + c] = rgb[c];
  }
  if constexpr (backward) dguide[p] = guide_grad;
}

template <bool backward>
static void launch(const Tensor& grid, const Tensor& guide, const Tensor& image,
                   const float* seed, float* out, float* dg, float* ds, float* di) {
  c10::cuda::CUDAGuard guard(grid.device());
  const int threads = 128;
  const int pixels = guide.numel();
  slice_kernel<backward><<<(pixels + threads - 1) / threads, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      grid.data_ptr<float>(), guide.data_ptr<float>(), image.data_ptr<float>(), seed, out, dg, ds, di,
      pixels, guide.size(1), guide.size(2), grid.size(1), grid.size(2), grid.size(3));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

static Tensor forward(const Tensor& grid, const Tensor& guide, const Tensor& image) {
  auto out = at::empty_like(image);
  launch<false>(grid, guide, image, nullptr, out.data_ptr<float>(), nullptr, nullptr, nullptr);
  return out;
}

static std::tuple<Tensor,Tensor,Tensor> backward(const Tensor& grid, const Tensor& guide,
                                                  const Tensor& image, const Tensor& seed) {
  auto dg = at::zeros_like(grid), ds = at::empty_like(guide), di = at::empty_like(image);
  auto dy = seed.contiguous();
  launch<true>(grid, guide, image, dy.data_ptr<float>(), nullptr,
               dg.data_ptr<float>(), ds.data_ptr<float>(), di.data_ptr<float>());
  return {dg, ds, di};
}

TORCH_LIBRARY(bilateral_fused, m) {
  m.def("forward(Tensor grid, Tensor guide, Tensor image) -> Tensor");
  m.def("backward(Tensor grid, Tensor guide, Tensor image, Tensor seed) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(bilateral_fused, CUDA, m) {
  m.impl("forward", &forward);
  m.impl("backward", &backward);
}
