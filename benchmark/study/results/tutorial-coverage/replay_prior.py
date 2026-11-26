"""Compare published backward TTIR against Enzyme differentiating the identical published primal."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import torch
import triton
from triton.experimental.autodiff import _find_tt_func, _make_public


def run(name, root, opt):
    files = root / "third_party/autodiff/test" / name / "generated/annotated"
    count = 2 if name == "add" else 3
    arguments = ",".join(["enzyme_dup"] * count)
    wrap = f"--enzyme-wrap=infn=add_kernel outfn=gradient argTys={arguments} mode=ReverseModeCombined"
    generated = subprocess.check_output([opt, files / "inp.ttir", wrap, "--canonicalize",
                                         "--remove-unnecessary-enzyme-ops", "--canonicalize"], text=True)
    expected_shape = (4,) if name == "add" else (16, 16)
    torch.manual_seed(17)
    inputs = [torch.randn(expected_shape, device="cuda") for _ in range(count - 1)]
    upstream = torch.randn_like(inputs[0])
    # Exactly representable small dyadics also distinguish both matrix transposes.
    if name == "2d_dot":
        inputs = [x.mul(8).round().div(8) for x in inputs]
        upstream = upstream.mul(8).round().div(8)
    params = [p.double().requires_grad_() for p in inputs]
    reference = params[0] + 42 if name == "add" else params[0] @ params[1]
    grads = torch.autograd.grad(reference, params, upstream.double())
    grid = (1, 1, 1)
    with tempfile.TemporaryDirectory() as d:
        enzyme_path = Path(d) / "out.ttir"
        enzyme_path.write_text("module {\n" + _make_public(_find_tt_func(generated, "gradient")) + "\n}\n")
        for method in ("published", "enzyme"):
            output = torch.full_like(inputs[0], float("nan"))
            seed = upstream.clone()
            # Evaluate the same initial primal and gradients for each method.
            adjoints = [torch.zeros_like(x) for x in inputs]
            if method == "published":
                triton.compile(str(files / "inp.ttir"))[grid](*inputs, output)
                triton.compile(str(files / "out.ttir"))[grid](*inputs, output, *adjoints, seed)
            else:
                args = [v for pair in zip(inputs, adjoints) for v in pair]
                triton.compile(str(enzyme_path))[grid](*args, output, seed)
            torch.testing.assert_close(output.double(), reference, rtol=1e-6, atol=5e-6)
            for actual, expected in zip(adjoints, grads):
                torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=5e-6)
            print(name, method, "max_grad_abs", max((a.double()-e).abs().max().item() for a,e in zip(adjoints,grads)),
                  "max_primal_abs", (output.double()-reference).abs().max().item(),flush=True)


def run_layernorm(root, opt, warps):
    files = root / "third_party/autodiff/test/layernorm/generated/annotated"
    entry = "_layer_norm_fwd_fused"
    wrap = f"--enzyme-wrap=infn={entry} outfn=gradient argTys=" + ",".join(["enzyme_dup"] * 6) + " mode=ReverseModeCombined"
    generated = subprocess.check_output([opt, files / "inp.ttir", wrap, "--canonicalize",
                                         "--remove-unnecessary-enzyme-ops", "--canonicalize"], text=True)
    torch.manual_seed(17)
    x, w, b = [torch.randn(shape, device="cuda", dtype=torch.float16) for shape in [(4, 8192), (8192,), (8192,)]]
    params = [t.double().requires_grad_() for t in (x, w, b)]
    y_ref = torch.nn.functional.layer_norm(params[0], (8192,), params[1], params[2], eps=1e-5)
    v = torch.randn_like(x)
    grad_ref = torch.autograd.grad(y_ref, params, v.double())
    grid = (4, 1, 1)
    failures = []
    with tempfile.TemporaryDirectory() as d:
        for method in ("published", "enzyme"):
            out = Path(d) / f"{method}.ttir"
            if method == "published":
                # The annotated artifact begins with non-MLIR compiler diagnostics.
                text = (files / "out.ttir").read_text()
                out.write_text(text[text.index("module {"):])
            else:
                out.write_text("module {\n" + _make_public(_find_tt_func(generated, "gradient")) + "\n}\n")
            y = torch.full_like(x, float("nan"))
            mu, rstd = [torch.zeros(4, device="cuda") for _ in range(2)]
            buffers = [x, y, w, b, mu, rstd]
            shadows = [torch.zeros_like(t) for t in buffers]
            shadows[1] = v.clone()
            if method == "published":
                triton.compile(str(files / "inp.ttir"), options={"num_warps": warps})[grid](*buffers)
                triton.compile(str(out), options={"num_warps": warps})[grid](*buffers, *shadows)
            else:
                triton.compile(str(out), options={"num_warps": warps})[grid](*[t for pair in zip(buffers, shadows) for t in pair])
            actual = (shadows[0], shadows[2], shadows[3])
            errors = [(g.double() - e).abs().max().item() for g, e in zip(actual, grad_ref)]
            print("layernorm", method, "warps", warps, "grad_errors_x_w_b", errors,
                  "max_primal_abs", (y.double() - y_ref).abs().max().item(), flush=True)
            torch.testing.assert_close(y.double(), y_ref, atol=.003, rtol=.001)
            for var, g, e in zip(("x", "weight", "bias"), actual, grad_ref):
                mismatches = (~torch.isclose(g.double(), e, atol=.015, rtol=.002)).sum().item()
                if mismatches:
                    failures.append(f"{method}:{var} mismatches={mismatches}/{e.numel()}")
    if failures:
        raise AssertionError(", ".join(failures))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("upstream_repo", type=Path)
    args.add_argument("enzyme_opt", type=Path)
    args.add_argument("--warps", type=int, default=4, help="warp count for the layernorm replay")
    args.add_argument("cases", nargs="+", choices=["add", "2d_dot", "layernorm"])
    opts = args.parse_args()
    for case in opts.cases:
        if case == "layernorm":
            run_layernorm(opts.upstream_repo, opts.enzyme_opt, opts.warps)
        else:
            run(case, opts.upstream_repo, opts.enzyme_opt)
