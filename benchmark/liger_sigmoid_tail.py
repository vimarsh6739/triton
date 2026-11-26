"""Stress-test the actual FP32 Liger SiLU pullback around exponential overflow."""

import csv
import sys

import torch
from liger_kernel.ops import swiglu
from liger_swiglu import generated_swiglu


def main():
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["mode", "input", "da", "reference_da", "db", "reference_db"])
    values = [-120., -100., -90., -88., -80., -64., -32., -1., -0., 0., 1., 32., 64., 88., 100., 120.]
    x = torch.tensor([values], dtype=torch.float64, device="cuda")
    s = x.sigmoid()
    ref_a = s + x * s * (1 - s)
    ref_b = x * s
    for mode in ("liger", "generated", "overwrite", "stable"):
        fn = swiglu.LigerSiLUMulFunction.apply if mode == "liger" else generated_swiglu(mode)
        a = x.float().requires_grad_()
        b = torch.ones_like(a).requires_grad_()
        c = fn(a, b, 1.0)
        da, db = torch.autograd.grad(c, (a, b), torch.ones_like(c))
        for val, g1, r1, g2, r2 in zip(values, da.flatten().tolist(), ref_a.flatten().tolist(),
                                        db.flatten().tolist(), ref_b.flatten().tolist()):
            writer.writerow([mode, val, g1, r1, g2, r2])
        if mode == "stable":
            torch.testing.assert_close(da.double(), ref_a, atol=1e-43, rtol=2e-5)
            torch.testing.assert_close(db.double(), ref_b, atol=1e-43, rtol=2e-5)


if __name__ == "__main__":
    main()
