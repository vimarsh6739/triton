"""Differentiate validation loss with respect to a minibatch's example weights."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import triton
from triton.experimental import Const, Duplicated, autodiff

from adamw import step


def validation_hypergradient(x, labels, seed):
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(x), generator=generator)
    train, val = order[256:], order[:256]
    xt, yt = x[train], labels[train]
    xv, yv = x[val], labels[val]
    # A pixel absent throughout training has zero first and second moments.
    # Remove such columns before taking the square-root derivative.
    keep = xt.abs().sum(0) > 0
    xt, xv = xt[:, keep].contiguous(), xv[:, keep].contiguous()
    params = torch.zeros(xt.shape[1], 10, device=x.device, requires_grad=True)
    opt = torch.optim.AdamW([params], lr=3e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    # Start at an actual optimizer checkpoint whose bias corrections match the kernel.
    for _ in range(31):
        opt.zero_grad()
        F.cross_entropy(xt @ params, yt).backward()
        opt.step()
    p = params.detach()
    state = opt.state[params]
    m, v = state["exp_avg"], state["exp_avg_sq"]
    assert (v > 0).all()
    batch = torch.randperm(len(train), generator=generator)[:128]
    xb, yb = xt[batch], yt[batch]
    weights = torch.ones(len(batch), device=x.device)

    def weighted_loss(param, w):
        return (F.cross_entropy(xb @ param, yb, reduction="none") * w).mean()

    train_grad = torch.func.grad(weighted_loss)
    g, pull_grad = torch.func.vjp(lambda w: train_grad(p, w), weights)
    p1, m1, v1 = (torch.empty_like(p) for _ in range(3))
    n = p.numel()
    grid = (triton.cdiv(n, 128),)
    step[grid](p, g.contiguous(), m, v, p1, m1, v1, n, BLOCK=128)
    seed_p = torch.func.grad(lambda w: F.cross_entropy(xv @ w, yv))(p1).contiguous()
    dg = torch.zeros_like(p)
    # Only the training-gradient input and the parameter output are active in this pullback.
    compiled = autodiff(step)[grid](Const(p), Duplicated(g.contiguous(), dg), Const(m), Const(v),
                                    Duplicated(p1, seed_p), Const(m1), Const(v1), Const(n), BLOCK=128)
    assert "@revdiffestep(" in compiled.asm["ttir"]
    assert seed_p.count_nonzero() == 0
    hypergrad, = pull_grad(dg)

    # Independently differentiate the full scalar objective in FP64. Use the
    # same rounded checkpoint as the measured kernel.
    xb64, xv64, p64, m64, v64 = (a.double() for a in (xb, xv, p, m, v))

    def meta_loss(w):
        def train_loss(q):
            losses = F.cross_entropy(xb64 @ q, yb, reduction="none")
            return torch.dot(losses, w) / len(w)

        grad = torch.func.grad(train_loss)(p64)
        mt = torch.lerp(grad, m64, 0.9)
        vt = v64 * 0.999 + grad.square() * 0.001
        direction = mt / (1 - 0.9**32) / ((vt / (1 - 0.999**32)).sqrt() + 1e-8)
        updated = p64 * (1 - 3e-3 * 1e-2) - 3e-3 * direction
        return F.cross_entropy(xv64 @ updated, yv)

    w64 = weights.double()
    reference = torch.func.grad(meta_loss)(w64)
    delta = hypergrad.double() - reference
    torch.testing.assert_close(hypergrad.double(), reference, rtol=1e-4, atol=2e-9)
    # Check one seeded direction per split independently with two objective evaluations.
    probe = torch.randn(len(batch), generator=generator, dtype=torch.float64).to(x.device)
    probe /= probe.norm()
    h = 1e-3
    finite_difference = (meta_loss(w64 + h * probe) - meta_loss(w64 - h * probe)) / (2 * h)
    directional = hypergrad.double().dot(probe)
    torch.testing.assert_close(directional, finite_difference, rtol=2e-4, atol=2e-9)
    return [seed, len(train), len(val), len(batch), n, meta_loss(w64).item(), reference.norm().item(),
            delta.norm().item() / reference.norm().item(), delta.abs().max().item(), directional.item(),
            finite_difference.item()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_csv", type=Path, help="UCI optdigits.tra")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43, 67, 97])
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    digest = hashlib.sha256(args.training_csv.read_bytes()).hexdigest()
    if digest != "e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd":
        parser.error("expected the published UCI optdigits.tra training split")
    data = torch.from_numpy(np.loadtxt(args.training_csv, delimiter=",")).to("cuda")
    x, labels = (data[:, :-1] / 16).float().contiguous(), data[:, -1].long()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "train_size", "validation_size", "batch", "parameters", "validation_loss",
                         "reference_grad_norm", "relative_l2_error", "max_abs_error", "directional_ad", "directional_fd"])
        for seed in args.seeds:
            row = validation_hypergradient(x, labels, seed)
            writer.writerow(row)
            print(f"seed={seed} relative L2 error={row[7]:.3g} directional FD={row[-1]:.6g}", flush=True)


if __name__ == "__main__":
    main()
