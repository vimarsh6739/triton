"""Slice a bilateral affine grid and apply its RGB transform to an image."""

import torch
import triton
import triton.language as tl
from triton.experimental import Duplicated, autodiff


@triton.jit
def _slice(Grid, Guide, Image, Output, H: tl.constexpr, W: tl.constexpr,
           GH: tl.constexpr, GW: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(1)
    pixel = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pixel < H * W
    index = batch * H * W + pixel
    gx = (pixel % W + 0.5) * (GW / W)
    gy = (pixel // W + 0.5) * (GH / H)
    gz = tl.load(Guide + index, valid, 0) * D
    x0 = tl.floor(gx - 0.5).to(tl.int32)
    y0 = tl.floor(gy - 0.5).to(tl.int32)
    z0 = tl.floor(gz - 0.5).to(tl.int32)

    coefficient = tl.arange(0, 16)
    affine = tl.full((BLOCK, 16), 0, tl.float32)
    for dy in tl.static_range(2):
        y = tl.maximum(0, tl.minimum(y0 + dy, GH - 1))
        wy = 1 - tl.abs(y0 + dy + 0.5 - gy)
        for dx in tl.static_range(2):
            x = tl.maximum(0, tl.minimum(x0 + dx, GW - 1))
            wx = 1 - tl.abs(x0 + dx + 0.5 - gx)
            for dz in tl.static_range(2):
                z = tl.maximum(0, tl.minimum(z0 + dz, D - 1))
                center = z0 + dz + 0.5
                distance = center - gz
                smooth = tl.sqrt(distance * distance + 1.0e-8)
                inside = (gz - (center - 1)) * ((center + 1) - gz) > 1.0e-8
                wz = tl.where(inside, 1 - smooth, 0.0)
                cell = ((batch * GH + y) * GW + x) * D + z
                offset = cell[:, None] * 12 + coefficient[None, :]
                values = tl.load(Grid + offset, valid[:, None] & (coefficient[None, :] < 12), 0)
                affine += (wx * wy * wz)[:, None] * values

    channel = tl.arange(0, 4)
    rgb = index[:, None] * 3 + channel[None, :]
    active = valid[:, None] & (channel[None, :] < 3)
    color = tl.load(Image + rgb, active, 1.0)
    result = tl.sum(affine.reshape((BLOCK, 4, 4)) * color[:, None, :], axis=2)
    tl.store(Output + rgb, result, active)


class _Bilateral(torch.autograd.Function):
    @staticmethod
    def forward(ctx, grid, guide, image):
        n, gh, gw, depth, cout, cin = grid.shape
        _, h, w = guide.shape
        if (cout, cin) != (3, 4) or guide.shape != (n, h, w) or image.shape != (n, h, w, 3):
            raise ValueError("expected a 3-by-4 affine grid, scalar guide and RGB image")
        if not grid.is_cuda or min(n, gh, gw, depth, h, w) <= 0:
            raise ValueError("expected nonempty CUDA tensors")
        if any(t.dtype != torch.float32 or not t.is_contiguous() or t.device != grid.device
               for t in (grid, guide, image)):
            raise ValueError("expected contiguous FP32 tensors on one GPU")
        if max(grid.numel(), image.numel()) > 2**31 - 1:
            raise ValueError("grid and image must fit signed 32-bit indexing")
        block = 32
        ctx.launch = (triton.cdiv(h * w, block), n)
        ctx.options = dict(H=h, W=w, GH=gh, GW=gw, D=depth, BLOCK=block, num_warps=4)
        output = torch.empty_like(image)
        _slice[ctx.launch](grid, guide, image, output, **ctx.options)
        ctx.save_for_backward(grid, guide, image)
        return output

    @staticmethod
    def backward(ctx, dy):
        grid, guide, image = ctx.saved_tensors
        dg, dh, di = (torch.zeros_like(t) for t in (grid, guide, image))
        scratch = torch.empty_like(image)
        autodiff(_slice)[ctx.launch](Duplicated(grid, dg), Duplicated(guide, dh),
                                    Duplicated(image, di), Duplicated(scratch, dy.contiguous().clone()),
                                    **ctx.options)
        return dg, dh, di


slice_apply = _Bilateral.apply
