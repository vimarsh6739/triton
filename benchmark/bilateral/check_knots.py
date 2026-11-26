"""Compare guide derivatives at depth-bin centers, adjacent FP32 inputs and HDRNet image guides."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from bench import get_methods
from inputs import make_inputs, oracle


def probes(depth):
    rows = []
    for center in (torch.arange(depth, dtype=torch.float32) + 0.5) / depth:
        rows.extend((("below", torch.nextafter(center, center - 1)), ("center", center),
                     ("above", torch.nextafter(center, center + 1))))
    rows.extend((("edge", torch.tensor(0.)), ("edge", torch.tensor(1.)),
                 ("outside", torch.tensor(-0.1)), ("outside", torch.tensor(1.1))))
    return rows


def bias_reference(g, depth):
    """Evaluate the scalar bias b[z]=z+1 and its guide derivative in the input dtype."""
    z = g * depth
    lower = (z - 0.5).floor()
    value, slope = torch.zeros_like(z), torch.zeros_like(z)
    for index in (lower, lower + 1):
        delta = index + 0.5 - z
        smooth = torch.sqrt(delta * delta + 1.0e-8)
        coeff = index.clamp(0, depth - 1) + 1
        value += coeff * (1 - smooth).clamp_min(0)
        slope += torch.where(smooth <= 1, coeff * depth * delta / smooth, 0)
    return value, slope


def guide_probe(models, depth, writer):
    rows = probes(depth)
    guide = torch.stack([v for _, v in rows]).reshape(1, 1, -1).cuda().requires_grad_()
    image = torch.zeros((*guide.shape, 3), device="cuda", requires_grad=True)
    grid = torch.zeros((1, 1, 1, depth, 3, 4), device="cuda")
    bias = torch.arange(1, 1 + depth, device="cuda")
    grid[0, 0, 0, :, 0, 3] = bias
    grid[0, 0, 0, :, 0, :3] = bias[:, None] * torch.tensor([0.125, -0.25, 0.5], device="cuda")
    grid.requires_grad_()
    exact_y, exact_d = bias_reference(guide.detach().double(), depth)
    fp_y, fp_d = bias_reference(guide.detach(), depth)
    gd, ud, rgbd = (t.detach().double().requires_grad_() for t in (grid, guide, image))
    exact_grid, _, exact_image = torch.autograd.grad(oracle(gd, ud, rgbd)[..., 0].sum(), (gd, ud, rgbd))
    for name, fn in models.items():
        y = fn(grid, guide, image)[..., 0]
        dg, d, di = torch.autograd.grad(y.sum(), (grid, guide, image))
        grid_error = (dg.double() - exact_grid).abs().max().item()
        image_error = (di.double() - exact_image).abs().max().item()
        torch.testing.assert_close(di.double(), exact_image, rtol=2e-5, atol=2e-5)
        if name in ("ttir", "cuda_strict", "fused_cuda", "warp_cuda", "torch", "compiled"):
            torch.testing.assert_close(dg.double(), exact_grid, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(y, fp_y, rtol=2e-6, atol=2e-6)
        if name in ("cuda", "cuda_rgb"):
            torch.testing.assert_close(d, fp_d, rtol=3e-6, atol=3e-5)
        if name in ("ttir", "cuda_strict", "fused_cuda", "warp_cuda"):
            torch.testing.assert_close(d.double(), exact_d, rtol=3e-6, atol=3e-5)
        if name in ("torch", "compiled"):
            centers = torch.tensor([which == "center" for which, _ in rows], device="cuda")
            torch.testing.assert_close(d.flatten()[centers].double(), exact_d.flatten()[centers], rtol=3e-6, atol=3e-5)
        for i, (which, g) in enumerate(rows):
            vals = [float(t.detach().flatten()[i]) for t in (exact_y, fp_y, y, exact_d, fp_d, d)]
            writer.writerow((name, which, float(g), *vals, grid_error, image_error))


def image_counts(source, guide, depth):
    _, height, width = guide.shape
    g = guide.detach().flatten()
    center = ((g * depth - 0.5).round() + 0.5) / depth
    exact = g == center
    adjacent = (g == torch.nextafter(center, center - 1)) | (g == torch.nextafter(center, center + 1))

    def active(values):
        z = values * depth
        lower = (z - 0.5).floor()
        return [((index + 0.5 - z).square() + 1e-8).sqrt() <= 1 for index in (lower, lower + 1)]

    a, b = active(g), active(g.double())
    changed = (a[0] != b[0]) | (a[1] != b[1])
    return (source, height, width, g.numel(), int(exact.sum()), int(adjacent.sum()), int(changed.sum()))


def trained_pixels(models, image_path, trained_dir, writer, outliers):
    size = 2048
    with np.load(trained_dir / f"{size}.npz") as data:
        grid = torch.from_numpy(data["grid"]).cuda().requires_grad_()
        guide = torch.from_numpy(data["guide"]).cuda().requires_grad_()
    depth = grid.shape[3]
    center = ((guide.detach() * depth - 0.5).round() + 0.5) / depth
    at_center = guide.detach() == center
    positions = torch.nonzero(at_center).tolist()
    for seed_id in (17, 43):
        _, _, rgb, seed = make_inputs(image_path, size, size, seed_id)
        expected = torch.empty_like(guide, dtype=torch.float64)
        for y in range(0, size, 32):
            rows = slice(y, y + 32)
            g = guide[:, rows].detach().double().requires_grad_()
            color = oracle(grid.detach(), g, rgb[:, rows].detach(), row_offset=y, full_height=size)
            expected[:, rows] = torch.autograd.grad(color, g, seed[:, rows].double())[0]
        peak = expected.abs().max()
        for name, fn in models.items():
            result = fn(grid, guide, rgb)
            dg = torch.autograd.grad(result, guide, seed)[0]
            error = (dg.detach().double() - expected).abs() / peak
            outliers.writerow((name, seed_id, float(peak), int((error[at_center] > 1e-2).sum()),
                               int((error[~at_center] > 1e-2).sum()), float(error[~at_center].max())))
            for n, y, x in positions:
                ref = float(expected[n, y, x])
                actual = float(dg[n, y, x])
                writer.writerow((name, seed_id, y, x, float(guide[n, y, x].detach()), ref, actual))
                if name not in ("cuda", "cuda_rgb") and abs(actual - ref) > 3e-5:
                    raise AssertionError(f"{name} gave {actual} instead of {ref} at {y}, {x}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--trained-dir", type=Path, help="directory with official HDRNet activation NPZs")
    parser.add_argument("--methods", nargs="+", default=["torch", "compiled", "cuda", "fused_cuda", "ttir"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "guide-frequency.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("guide_source", "height", "width", "pixels", "exact_center", "one_ulp_from_center", "fp32_fp64_active_disagreements"))
        for size in (1024, 2048):
            _, guide, _, _ = make_inputs(args.image, size, size, 17, device="cpu")
            writer.writerow(image_counts("luminance", guide, 8))
            if args.trained_dir:
                with np.load(args.trained_dir / f"{size}.npz") as data:
                    writer.writerow(image_counts("trained", torch.from_numpy(data["guide"]), data["grid"].shape[3]))
    models = get_methods(args.methods)
    with (args.output / "guide-probes.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("method", "case", "guide", "y_fp64", "y_fp32_stencil", "y_actual",
                         "dguide_fp64", "dguide_fp32_stencil", "dguide_actual",
                         "max_abs_dgrid_error", "max_abs_dimage_error"))
        guide_probe(models, 8, writer)
    if args.trained_dir:
        with (args.output / "trained-pixels.csv").open("w", newline="") as f, (
                args.output / "trained-outliers.csv").open("w", newline="") as stats:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(("method", "seed", "y", "x", "guide", "dguide_fp64", "dguide_actual"))
            counts = csv.writer(stats, lineterminator="\n")
            counts.writerow(("method", "seed", "peak_abs_dguide_fp64", "above_1pct_peak_at_bin_center",
                             "above_1pct_peak_elsewhere", "max_error_over_peak_elsewhere"))
            trained_pixels(models, args.image, args.trained_dir, writer, counts)


if __name__ == "__main__":
    main()
