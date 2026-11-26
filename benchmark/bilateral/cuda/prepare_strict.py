"""Make an HDRNet CUDA copy with a full-precision activity test for the smoothed depth weight."""

import argparse
from pathlib import Path
from shutil import copyfile

from specialize_rgb import replace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New Python module directory named cuda_strict")
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    src = Path(__file__).resolve().parent

    kernels = (src / "hdrnet_kernels.cuh").read_text()
    old = """__device__ float d_weight_z(float x) {
  float abx = diff_abs(x);
  if(abx > 1.0f) {"""
    new = """__device__ float d_weight_z(double x) {
  if (fabs(x) > sqrt(1.0 - 1e-8)) {"""
    (out / "hdrnet_kernels.cuh").write_text(replace(kernels, old, new, 1))

    launch = (src / "launch.cu").read_text()
    (out / "launch.cu").write_text(replace(launch, "hdrnet_bench,", "hdrnet_bench_strict,", 2))
    wrapper = (src / "__init__.py").read_text()
    wrapper = replace(wrapper, 'name="hdrnet_bilateral_bench"', 'name="hdrnet_bilateral_bench_strict"', 1)
    (out / "__init__.py").write_text(replace(wrapper, "torch.ops.hdrnet_bench", "torch.ops.hdrnet_bench_strict", 1))
    copyfile(src / "LICENSE", out / "LICENSE")


if __name__ == "__main__":
    main()
