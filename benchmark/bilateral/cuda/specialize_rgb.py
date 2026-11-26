"""Build a separate copy of the historical kernels with constexpr RGB and affine-offset parameters."""

import argparse
from pathlib import Path
from shutil import copyfile


def replace(source, old, new, count):
    if source.count(old) != count:
        raise ValueError(f"Expected {count} occurrences of {old!r} in the pinned file")
    return source.replace(old, new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New module directory named cuda_rgb")
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent

    # NVCC now sees the RGB dimensions in each of the four kernel bodies.
    kernels = (source / "hdrnet_kernels.cuh").read_text()
    kernels = replace(kernels, "    const int input_chans, const int output_chans, const bool has_offset,\n", "", 4)
    kernels = replace(kernels, "    float* out)\n{", "    float* out)\n{\n"
                      "  constexpr int input_chans = 3, output_chans = 3;\n"
                      "  constexpr bool has_offset = true;", 4)
    (out / "hdrnet_kernels.cuh").write_text(kernels)

    launch = (source / "launch.cu").read_text()
    launch = replace(launch, ", 3, 3, true,", ",", 4)
    (out / "launch.cu").write_text(replace(launch, "hdrnet_bench,", "hdrnet_bench_rgb,", 2))
    wrapper = (source / "__init__.py").read_text()
    wrapper = replace(wrapper, 'name="hdrnet_bilateral_bench"', 'name="hdrnet_bilateral_bench_rgb"', 1)
    (out / "__init__.py").write_text(replace(wrapper, "torch.ops.hdrnet_bench", "torch.ops.hdrnet_bench_rgb", 1))
    copyfile(source / "LICENSE", out / "LICENSE")


if __name__ == "__main__":
    main()
