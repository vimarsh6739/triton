import time

import torch


def assert_close(label, actual, expected, *, atol, rtol=0.0):
    actual_f = actual.detach().float()
    expected_f = expected.detach().float()
    max_abs = (actual_f - expected_f).abs().max().item()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    print(f"  {label}: max_abs={max_abs:.6g}")


def check_ir(compiled, function_name, required=()):
    ttir = compiled.asm["ttir"]
    derivative_name = f"fwddiffe{function_name}"
    if derivative_name not in ttir:
        raise AssertionError(f"missing @{derivative_name} in differentiated TTIR")
    for text in required:
        if text not in ttir:
            raise AssertionError(f"missing {text!r} in differentiated TTIR for {function_name}")
    print(f"  TTIR: @{derivative_name}, {len(ttir)} bytes")


def run(name, benchmark):
    print(f"=== {name} ===")
    start = time.perf_counter()
    benchmark()
    print(f"PASS {name} ({time.perf_counter() - start:.2f}s)")
