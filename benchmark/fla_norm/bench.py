"""Time both normalizations and pullbacks at FLA's Kimi Delta Attention boundary."""
import argparse
import csv
import random
from pathlib import Path
from statistics import median

import torch
import triton
from fla.modules.norm.l2norm import l2norm_bwd, l2norm_fwd, l2norm_fwd_kernel_row
from triton.experimental import autodiff, Const, Duplicated

from kernels import fla_forward, paired_forward, paired_backward, saved_pullback

EPS = 1e-6
DTYPE = {torch.bfloat16: 'bf16', torch.float16: 'fp16', torch.float32: 'fp32'}


def run(method, xs, seeds, bt, warps, opt):
    q, k = xs
    rows, width = q.numel() // q.shape[-1], q.shape[-1]
    bd = triton.next_power_of_2(width)
    if method in ('fla', 'combined'):
        saved = [l2norm_fwd(x) for x in xs]
        ys, rs = zip(*saved)
        if method == 'fla':
            grads = [l2norm_bwd(y, r, d) for y, r, d in zip(ys, rs, seeds)]
        else:
            grads = []
            for x, seed in zip(xs, seeds):
                dx = torch.zeros_like(x)
                y = torch.empty_like(x)
                r = torch.empty(rows, device=q.device, dtype=torch.float32)
                args = (Duplicated(x, dx), Duplicated(y, seed.clone()), Const(r), Const(EPS))
                if width <= 512:
                    autodiff(fla_forward)[(triton.cdiv(rows, bt),)](
                        *args, Const(rows), D=width, BD=bd, BT=bt, num_warps=warps)
                else:
                    autodiff(l2norm_fwd_kernel_row.fn)[(rows,)](*args, Const(width), BD=bd, num_warps=warps)
                grads.append(dx)
    else:
        ys = [torch.empty_like(x) for x in xs]
        rs = [torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32) for x in xs]
        grads = [torch.empty_like(x) for x in xs]
        grid = (triton.cdiv(rows, bt), 2)
        paired_forward[grid](*xs, *ys, *rs, EPS, rows, width, bd, bt, num_warps=warps)
        if method == 'paired-fla':
            paired_backward[grid](*ys, *rs, *seeds, *grads, EPS, rows, width, bd, bt, num_warps=warps)
        elif method in ('paired-ad', 'paired-ad-finite', 'paired-ad-fold'):
            kernel = saved_pullback(rows, width, DTYPE[q.dtype], bt, warps, opt, method != 'paired-ad', method == 'paired-ad-fold')
            kernel[(grid[0], 2, 1)](*ys, *rs, *seeds, *grads)
        else:
            raise ValueError(method)
    return *ys, *rs, *grads


def reference(xs, seeds):
    out, rs, grads = [], [], []
    for x, seed in zip(xs, seeds):
        z = x.double().requires_grad_()
        r = torch.rsqrt(z.square().sum(-1) + EPS)
        y = z * r[..., None]
        g, = torch.autograd.grad(y, z, seed.double())
        out.append(y.detach())
        rs.append(r.detach())
        grads.append(g)
    return *out, *rs, *grads


def make_inputs(shape, dtype, seed):
    torch.manual_seed(seed)
    xs = [torch.randn(shape, device='cuda', dtype=dtype) for _ in range(2)]
    for x in xs:
        rows = x.view(-1, shape[-1])
        rows[0].zero_()
        rows[1].mul_(1e-5)
    seeds = [torch.randn_like(x) for x in xs]
    return xs, seeds


def check(values, truth, dtype):
    # Normalize each row separately, so near-zero rows with gradients ~1000
    # don't hide errors elsewhere. Also check absolute error for exact-zero rows.
    errors = []
    for value, expected in zip(values, truth):
        diff = (value.double() - expected).reshape(-1, value.shape[-1])
        denom = expected.reshape_as(diff).norm(dim=1).clamp_min(1e-8)
        worst = (diff.norm(dim=1) / denom).amax().item()
        assert worst <= {torch.float32: 2e-5, torch.float16: .002, torch.bfloat16: .015}[dtype], worst
        errors.append(worst)
    return errors


def time_graphs(fns, repeats, launches, seed):
    graphs, outputs = {}, {}
    for name, fn in fns.items():
        fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(launches):
                outputs[name] = fn()
        graphs[name] = graph
        graph.replay()
    torch.cuda.synchronize()
    samples = {name: [] for name in fns}
    rng = random.Random(seed)
    for i in range(5 + repeats):
        order = list(graphs)
        rng.shuffle(order)
        for name in order:
            a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            graphs[name].replay()
            b.record()
            b.synchronize()
            if i >= 5:
                samples[name].append(a.elapsed_time(b) * 1000 / launches)
    return samples, outputs


SHAPES = {
    'tail60': ((1, 63, 1, 60), torch.float32),
    'tail512': ((1, 65, 2, 512), torch.float16),
    'tail513': ((1, 65, 2, 513), torch.bfloat16),
    'kda-short': ((1, 128, 16, 128), torch.bfloat16),
    'kda-long': ((1, 2048, 16, 128), torch.bfloat16),
    'wide': ((1, 1024, 4, 1024), torch.float16),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--opt', default='enzymexlamlir-opt')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', nargs='+', default=list(SHAPES))
    p.add_argument('--methods', nargs='+', default=['fla', 'combined', 'paired-fla', 'paired-ad', 'paired-ad-finite', 'paired-ad-fold'])
    p.add_argument('--bt', type=int, default=16)
    p.add_argument('--warps', type=int, default=4)
    p.add_argument('--repeats', type=int, default=31)
    p.add_argument('--launches', type=int, default=4)
    p.add_argument('--seed', type=int, default=71)
    p.add_argument('--ir-dir', type=Path, help='Also save each generated saved-state pullback and its PTX')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'timings.csv').open('w', newline='') as timing, (args.output / 'summary.csv').open('w', newline='') as summary:
        raw, rec = csv.writer(timing, lineterminator='\n'), csv.writer(summary, lineterminator='\n')
        raw.writerow(['case', 'method', 'sample', 'us'])
        rec.writerow(['case', 'shape', 'dtype', 'method', 'median_us', 'q_err', 'k_err', 'qr_err', 'kr_err', 'dq_err', 'dk_err'])
        for label in args.cases:
            shape, dtype = SHAPES[label]
            xs, seeds = make_inputs(shape, dtype, args.seed)
            original = [s.clone() for s in seeds]
            truth = reference(xs, seeds)
            fns = {m: lambda m=m: run(m, xs, seeds, args.bt, args.warps, args.opt) for m in args.methods}
            for name, fn in fns.items():
                check(fn(), truth, dtype)
            samples, outputs = time_graphs(fns, args.repeats, args.launches, args.seed)
            for name, values in outputs.items():
                errors = check(values, truth, dtype)
                times = samples[name]
                raw.writerows((label, name, i, t) for i, t in enumerate(times))
                rec.writerow((label, 'x'.join(map(str, shape)), DTYPE[dtype], name, median(times), *errors))
                print(label, name, median(times), max(errors), flush=True)
                if args.ir_dir is not None and name.startswith('paired-ad'):
                    args.ir_dir.mkdir(parents=True, exist_ok=True)
                    kernel = saved_pullback(xs[0].numel() // shape[-1], shape[-1], DTYPE[dtype],
                                            args.bt, args.warps, args.opt, name != 'paired-ad', name == 'paired-ad-fold')
                    for stage in ('ttgir', 'ptx'):
                        (args.ir_dir / f'{label}-{name}.{stage}').write_text(kernel.asm[stage])
            for seed, initial in zip(seeds, original):
                torch.testing.assert_close(seed, initial, atol=0, rtol=0)
            timing.flush()
            summary.flush()


if __name__ == '__main__':
    main()
