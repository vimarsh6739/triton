"""Run the native softmax pass through the experimental Triton AD launcher."""

import torch
from triton.experimental import Duplicated, autodiff
from softmax import softmax


def main():
    torch.manual_seed(8)
    x = torch.randn(64, 127, device="cuda")
    u = torch.randn_like(x)
    probabilities = x.double().softmax(-1)
    ref = probabilities * (u.double() - (probabilities * u.double()).sum(-1, keepdim=True))
    for mode in ("ForwardMode", "ReverseModeCombined"):
        y = torch.full_like(x, float("nan"))
        for schedule, expected in [(None, 5), ("detach-triton-softmax-shift", 3)]:
            dx = u.clone() if mode == "ForwardMode" else torch.zeros_like(x)
            dy = torch.empty_like(x) if mode == "ForwardMode" else u.clone()
            diff = autodiff(softmax, mode=mode, pre_ad=schedule, post_ad="remove-unnecessary-enzyme-ops,canonicalize,cse")
            compiled = diff[(64,)](Duplicated(x, dx), Duplicated(y, dy), COLS=127, TILE=128, num_warps=4)
            reductions = compiled.asm["ttir"].count('"tt.reduce"')
            assert reductions == expected, (mode, schedule, reductions)
            torch.testing.assert_close(y, probabilities.float(), atol=2e-6, rtol=1e-5)
            result = dy if mode == "ForwardMode" else dx
            torch.testing.assert_close(result, ref.float(), atol=3e-6, rtol=1e-5)
            print(mode, schedule or "baseline", reductions, (result.double() - ref).abs().max().item())


if __name__ == "__main__":
    main()
