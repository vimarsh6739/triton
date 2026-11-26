"""HDRNet's smoothed bilateral slice and affine RGB, using native PyTorch autograd."""

import torch
import torch.nn.functional as F


def slice_apply(grid, guide, image):
    batch, height, width = guide.shape
    depth = grid.shape[3]
    z = guide * depth - 0.5
    base = z.floor()
    frac = z - base
    low = (1 - (frac.square() + 1e-8).sqrt()).relu()
    high = (1 - ((1 - frac).square() + 1e-8).sqrt()).relu()
    mass = low + high
    # A trilinear sampler blends two depth slices with weights summing to one.
    # Warping z and restoring their mass implements HDRNet's two smoothed weights.
    z_norm = (base + high / mass + 0.5) * (2 / depth) - 1
    x = (torch.arange(width, device=guide.device, dtype=guide.dtype) + 0.5) * (2 / width) - 1
    y = (torch.arange(height, device=guide.device, dtype=guide.dtype) + 0.5) * (2 / height) - 1
    coords = torch.stack((x.expand_as(guide), y[:, None].expand_as(guide), z_norm), dim=-1)
    volume = grid.flatten(-2).permute(0, 4, 3, 1, 2)
    coeff = F.grid_sample(volume, coords[:, None], padding_mode="border", align_corners=False)
    affine = coeff.reshape(batch, 3, 4, height, width)
    rgb1 = F.pad(image, (0, 1), value=1).permute(0, 3, 1, 2)
    return ((affine * rgb1[:, None]).sum(2).permute(0, 2, 3, 1) * mass[..., None])
