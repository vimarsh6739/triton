"""Check the memory contract and measure live allocations during backward."""

import csv
from pathlib import Path
import argparse

import torch
from liger_kernel.ops import swiglu
from liger_swiglu import generated_swiglu
from liger_writeback import backward


def ownership():
    a, b = [torch.randn((8, 128), device="cuda").requires_grad_() for _ in range(2)]
    seed = torch.randn((128, 8), device="cuda").T  # Noncontiguous incoming cotangent.
    originals = [t.clone() for t in (a, b, seed)]
    fresh = generated_swiglu("overwrite")(a, b, 1.0)
    expected = torch.autograd.grad(fresh, (a, b), seed, retain_graph=True)
    repeated = torch.autograd.grad(fresh, (a, b), seed)
    for grad, again in zip(expected, repeated):
        torch.testing.assert_close(grad, again, rtol=0, atol=0)
    for t, old in zip((a, b, seed), originals):
        torch.testing.assert_close(t, old, rtol=0, atol=0)

    donated = generated_swiglu("donate")(a, b, 1.0)
    grads = torch.autograd.grad(donated, (a, b), seed, retain_graph=True)
    for t, grad, ref in zip((a, b), grads, expected):
        assert grad.data_ptr() == t.data_ptr()
        torch.testing.assert_close(grad, ref, rtol=0, atol=0)
    try:
        torch.autograd.grad(donated, (a, b), seed)
    except RuntimeError as error:
        if "inplace" not in str(error):
            raise
    else:
        raise AssertionError("Expected reuse of consumed activations to fail")
    try:
        backward(a, a, seed, 1.0, "donate")
    except ValueError as error:
        if "distinct storage" not in str(error):
            raise
    else:
        raise AssertionError("Expected shared-storage donation to fail")


def memory(directory):
    shape = (3072, 12288)
    with (directory / "backward-memory.csv").open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["mode", "peak_increment_bytes", "da_reuses_a", "db_reuses_b"])
        for mode in ("liger", "generated", "discard", "read_seed", "overwrite", "donate"):
            fn = swiglu.LigerSiLUMulFunction.apply if mode == "liger" else generated_swiglu(mode)
            # Warm compile first. Reset PyTorch allocator counters immediately
            # before the measured backward, with input, output and seed live.
            for _ in range(2):
                a, b = [torch.randn(shape, device="cuda", dtype=torch.bfloat16).requires_grad_() for _ in range(2)]
                seed = torch.randn_like(a)
                y = fn(a, b, 1.0)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                allocated = torch.cuda.memory_allocated()
                da, db = torch.autograd.grad(y, (a, b), seed)
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() - allocated
                reuse = da.data_ptr() == a.data_ptr(), db.data_ptr() == b.data_ptr()
                del a, b, y, da, db, seed
            writer.writerow([mode, peak, *reuse])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    ownership()
    memory(args.directory)
