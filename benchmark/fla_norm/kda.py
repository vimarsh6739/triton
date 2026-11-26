"""Exercise the generated paired normalization in the original chunk KDA operator."""
import argparse
import csv
from pathlib import Path

import torch
import triton
from fla.ops.kda import chunk_kda

from kernels import paired_forward, saved_pullback


class NormalizeQK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, opt):
        if q.shape != k.shape or q.dtype != torch.bfloat16 or not (q.is_contiguous() and k.is_contiguous()):
            raise ValueError("This fixture expects matching contiguous BF16 Q/K")
        width = q.shape[-1]
        rows = q.numel() // width
        ctx.bt, ctx.warps = 16, 4
        qs, ks = torch.empty_like(q), torch.empty_like(k)
        qr = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
        kr = torch.empty_like(qr)
        grid = (triton.cdiv(rows, ctx.bt), 2)
        paired_forward[grid](q, k, qs, ks, qr, kr, 1e-6, rows, width, triton.next_power_of_2(width), ctx.bt,
                             num_warps=ctx.warps)
        ctx.save_for_backward(qs, ks, qr, kr)
        ctx.opt = opt
        return qs, ks

    @staticmethod
    def backward(ctx, dq, dk):
        q, k, qr, kr = ctx.saved_tensors
        rows, width = q.numel() // q.shape[-1], q.shape[-1]
        gq, gk = torch.empty_like(q), torch.empty_like(k)
        kernel = saved_pullback(rows, width, 'bf16', ctx.bt, ctx.warps, ctx.opt, True, fold=True)
        kernel[(triton.cdiv(rows, ctx.bt), 2, 1)](q, k, qr, kr, dq.contiguous(), dk.contiguous(), gq, gk)
        return gq, gk, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opt', default='enzymexlamlir-opt')
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    results = []
    for seq, heads, dim in [(127, 2, 64), (257, 16, 128)]:
        torch.manual_seed(73)
        shape = (1, seq, heads, dim)
        q, k, v = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
        gate = torch.nn.functional.logsigmoid(torch.randn(shape, device='cuda'))
        beta = torch.rand(shape[:-1], device='cuda', dtype=torch.bfloat16)
        seed = torch.randn_like(v)
        inputs = [x.requires_grad_() for x in (q, k, v, gate, beta)]
        reference = None
        for paired in (False, True):
            qk = NormalizeQK.apply(q, k, args.opt) if paired else (q, k)
            out, _ = chunk_kda(*qk, v, gate, beta, use_qk_l2norm_in_kernel=not paired)
            values = (out, *torch.autograd.grad(out, inputs, seed))
            if reference is None:
                reference = [x.detach() for x in values]
                continue
            for name, value, expected in zip(('output', 'dq', 'dk', 'dv', 'dgate', 'dbeta'), values, reference):
                err = ((value.double() - expected.double()).norm() / expected.double().norm()).item()
                assert err < .01, (seq, name, err)
                results.append((seq, heads, dim, name, err))
            print(seq, heads, dim, results[-6:], flush=True)
    with args.output.open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(('tokens', 'heads', 'head_dim', 'output', 'rel_l2_vs_fla'))
        w.writerows(results)


if __name__ == '__main__':
    main()
