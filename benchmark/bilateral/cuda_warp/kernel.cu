#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <cuda_runtime.h>

using at::Tensor;

// Consecutive groups of four lanes cover a contiguous row of the 3x4 affine map.
template <int lanes, bool reverse>
__global__ void slice_kernel(const float* grid, const float* guide, const float* image,
                             const float* seed, float* output, float* dgrid, float* dguide,
                             float* dimage, int pixels, int h, int w, int gh, int gw, int depth) {
  const int thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int p = thread / lanes;
  if (p >= pixels) return;
  const unsigned mask = __activemask();
  const int lane = thread % lanes, in = lane % 4;
  constexpr int rows = lanes / 4, trips = (3 + rows - 1) / rows;
  const float gx = ((p % w) + 0.5f) * (float(gw) / w);
  const float gy = (((p / w) % h) + 0.5f) * (float(gh) / h);
  const float gz = guide[p] * depth;
  const int x0 = __float2int_rd(gx - 0.5f), y0 = __float2int_rd(gy - 0.5f);
  const int z0 = __float2int_rd(gz - 0.5f);
  const float color = in == 3 ? 1.0f : image[p * 3 + in];
  float rgb[trips] = {}, guide_grad = 0, input_grad = 0;
  #pragma unroll
  for (int corner = 0; corner < 8; ++corner) {
    const int x = x0 + ((corner >> 1) & 1), y = y0 + (corner >> 2), z = z0 + (corner & 1);
    const float xy = (1 - fabsf(x + 0.5f - gx)) * (1 - fabsf(y + 0.5f - gy));
    const float center = z + 0.5f, delta = center - gz;
    const float smooth = sqrtf(delta * delta + 1e-8f);
    const bool inside = (gz - (center - 1)) * ((center + 1) - gz) > 1e-8f;
    const float weight = inside ? xy * (1 - smooth) : 0;
    const int row = (p / (w * h)) * gh + min(max(y, 0), gh - 1);
    const int cell = (row * gw + min(max(x, 0), gw - 1)) * depth + min(max(z, 0), depth - 1);
    float corner_grad = 0;
    #pragma unroll
    for (int trip = 0; trip < trips; ++trip) {
      const int out = trip * rows + lane / 4;
      if (out < 3) {
        const int k = cell * 12 + out * 4 + in;
        const float coeff = grid[k];
        if constexpr (reverse) {
          const float dy = seed[3 * p + out];
          atomicAdd(dgrid + k, weight * dy * color);
          input_grad += weight * dy * coeff;
          corner_grad += dy * coeff * color;
        } else {
          rgb[trip] += weight * coeff * color;
        }
      }
    }
    if constexpr (reverse) {
      if (inside) guide_grad += xy * (delta / smooth) * depth * corner_grad;
    }
  }
  if constexpr (reverse) {
    #pragma unroll
    for (int span = 4; span < lanes; span *= 2)
      input_grad += __shfl_xor_sync(mask, input_grad, span, lanes);
    if (lane < 3) dimage[3 * p + in] = input_grad;
    #pragma unroll
    for (int span = 1; span < lanes; span *= 2)
      guide_grad += __shfl_xor_sync(mask, guide_grad, span, lanes);
    if (lane == 0) dguide[p] = guide_grad;
  } else {
    #pragma unroll
    for (int trip = 0; trip < trips; ++trip) {
      float value = rgb[trip];
      value += __shfl_xor_sync(mask, value, 1, 4);
      value += __shfl_xor_sync(mask, value, 2, 4);
      const int out = trip * rows + lane / 4;
      if (in == 0 && out < 3) output[3 * p + out] = value;
    }
  }
}

template <int lanes, bool reverse>
static void launch(const Tensor& grid, const Tensor& guide, const Tensor& image, int threads,
                   const float* seed, float* out, float* dg, float* ds, float* di) {
  c10::cuda::CUDAGuard guard(grid.device());
  const int pixels = guide.numel(), tile = threads / lanes;
  slice_kernel<lanes, reverse><<<(pixels + tile - 1) / tile, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
      grid.data_ptr<float>(), guide.data_ptr<float>(), image.data_ptr<float>(), seed, out, dg, ds, di,
      pixels, guide.size(1), guide.size(2), grid.size(1), grid.size(2), grid.size(3));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int lanes>
static Tensor forward(const Tensor& grid, const Tensor& guide, const Tensor& image, int64_t threads) {
  auto out = at::empty_like(image);
  launch<lanes, false>(grid, guide, image, threads, nullptr, out.data_ptr<float>(), nullptr, nullptr, nullptr);
  return out;
}

template <int lanes>
static std::tuple<Tensor,Tensor,Tensor> backward(const Tensor& grid, const Tensor& guide,
                                                  const Tensor& image, const Tensor& seed, int64_t threads) {
  auto dg = at::zeros_like(grid), ds = at::empty_like(guide), di = at::empty_like(image);
  auto dy = seed.contiguous();
  launch<lanes, true>(grid, guide, image, threads, dy.data_ptr<float>(), nullptr,
                      dg.data_ptr<float>(), ds.data_ptr<float>(), di.data_ptr<float>());
  return {dg, ds, di};
}

TORCH_LIBRARY(bilateral_warp, m) {
  m.def("forward4(Tensor grid, Tensor guide, Tensor image, int threads) -> Tensor");
  m.def("forward16(Tensor grid, Tensor guide, Tensor image, int threads) -> Tensor");
  m.def("backward4(Tensor grid, Tensor guide, Tensor image, Tensor seed, int threads) -> (Tensor, Tensor, Tensor)");
  m.def("backward16(Tensor grid, Tensor guide, Tensor image, Tensor seed, int threads) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(bilateral_warp, CUDA, m) {
  m.impl("forward4", &forward<4>);
  m.impl("forward16", &forward<16>);
  m.impl("backward4", &backward<4>);
  m.impl("backward16", &backward<16>);
}
