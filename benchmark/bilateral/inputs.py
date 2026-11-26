"""HDRNet sample-image inputs and an independent FP64 eight-corner reference."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SAMPLE_URL = ("https://raw.githubusercontent.com/google/hdrnet/"
              "e9a5bf59a232ee7562d02a6cf4dc5cef6191b6c6/sample_data/input.png")
SAMPLE_SHA256 = "7c9159b59423080504fdc7c4a061e20a240abcee0f4645348d11c88a4c678375"


def make_inputs(path, height, width, seed, *, grid_shape=(16, 16, 8), batch=1, device="cuda"):
    """Make contiguous FP32 grid, independent luminance guide, image, and output cotangent."""
    with Path(path).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != SAMPLE_SHA256:
            raise ValueError("Expected the original HDRNet sample_data/input.png")
    with Image.open(path) as frame:
        pixels = np.array(frame.convert("RGB").resize((width, height), Image.Resampling.BOX))
    photo = torch.from_numpy(pixels).float().div_(255)
    image = torch.stack([photo if i % 2 == 0 else photo.flip(1) for i in range(batch)])
    guide = image @ torch.tensor([0.299, 0.587, 0.114])
    rng = torch.Generator().manual_seed(seed)
    identity = torch.eye(3, 4)
    grid = identity + 0.1 * torch.randn(batch, *grid_shape, 3, 4, generator=rng)
    cotangent = torch.randn(image.shape, generator=rng)
    args = tuple(value.to(device).contiguous().requires_grad_() for value in (grid, guide, image))
    return (*args, cotangent.to(device))


def oracle(grid, guide, image, *, row_offset=0, full_height=None):
    """Evaluate the stencil directly in FP64, preserving derivatives of all three inputs."""
    grid, guide, image = grid.double(), guide.double(), image.double()
    batch, height, width = guide.shape
    _, gh, gw, depth, _, _ = grid.shape
    device = guide.device
    by = torch.arange(batch, device=device)[:, None, None]
    x = (torch.arange(width, dtype=torch.float64, device=device) + 0.5) * (gw / width) - 0.5
    full_height = height if full_height is None else full_height
    y = (torch.arange(height, dtype=torch.float64, device=device) + row_offset + 0.5) * (gh / full_height) - 0.5
    z = guide * depth - 0.5
    ix, iy, iz = x.floor().long(), y.floor().long(), z.floor().long()
    result = 0
    for dy in (0, 1):
        wy = 1 - (y - (iy + dy)).abs()
        for dx in (0, 1):
            wx = 1 - (x - (ix + dx)).abs()
            for dz in (0, 1):
                wz = (1 - ((z - (iz + dz)).square() + 1e-8).sqrt()).clamp_min(0)
                coeff = grid[by, (iy + dy).clamp(0, gh - 1)[:, None], (ix + dx).clamp(0, gw - 1),
                             (iz + dz).clamp(0, depth - 1)]
                rgb = (coeff[..., :3] * image[..., None, :]).sum(-1) + coeff[..., 3]
                result = result + rgb * (wz * wy[:, None] * wx)[..., None]
    return result
