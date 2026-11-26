from triton.experimental import Const, Duplicated, autodiff, fwddiff

from .shared import launch, primal, step


forward = fwddiff(step)
reverse = autodiff(step)


def apply(kernel, inputs, outputs, input_shadows, output_shadows, block, warps):
    pairs = zip((*inputs, *outputs), (*input_shadows, *output_shadows), strict=True)
    args = [Duplicated(value, shadow) for value, shadow in pairs]
    n = inputs[0].numel()
    return launch(kernel, (*args, Const(n)), n, block, warps)


def jvp(inputs, outputs, input_shadows, output_shadows):
    return apply(forward, inputs, outputs, input_shadows, output_shadows, 128, 8)


def vjp(inputs, outputs, input_shadows, output_shadows):
    return apply(reverse, inputs, outputs, input_shadows, output_shadows, 1024, 8)
