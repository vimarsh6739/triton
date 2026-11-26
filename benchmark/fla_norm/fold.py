"""Fold a verified TTIR floating sum of positive zeros before GPU lowering."""
import math

from jaxlib.mlir import ir


def sum_of_zero(op):
    if op.name != 'tt.reduce' or len(op.operands) != 1 or len(op.results) != 1:
        return None
    result = op.results[0].type
    if not isinstance(result, ir.RankedTensorType):
        return None
    source = op.operands[0].owner
    if isinstance(source, ir.Block) or source.operation.name != 'arith.constant':
        return None
    value = source.attributes['value']
    if not isinstance(value, ir.DenseFPElementsAttr) or not value.is_splat:
        return None
    zero = ir.FloatAttr(value.get_splat_value())
    if zero.value != 0. or math.copysign(1., zero.value) < 0:
        return None
    block = op.regions[0].blocks[0]
    ops = list(block.operations)
    if len(ops) != 2:
        return None
    add, ret = (view.operation for view in ops)
    if (add.name != 'arith.addf' or list(add.operands) != list(block.arguments)
            or ret.name != 'tt.reduce.return' or list(ret.operands) != list(add.results)):
        return None
    return ir.DenseElementsAttr.get_splat(result, zero)


def fold_zero_sums(generic_mlir):
    """Rewrite the straight-line entry emitted by saved_pullback."""
    with ir.Context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = ir.Module.parse(generic_mlir)
        (entry,) = module.body.operations
        count = 0
        for view in list(entry.regions[0].blocks[0].operations):
            op = view.operation
            zero = sum_of_zero(op)
            if zero is None:
                continue
            constant = ir.Operation.create('arith.constant', results=[op.result.type],
                                           attributes={'value': zero}, ip=ir.InsertionPoint(op))
            op.result.replace_all_uses_with(constant.result)
            op.erase()
            count += 1
        if count != 1:
            raise ValueError(f'Expected one constant-zero reduction in this saved-state kernel, found {count}')
        return str(module)
