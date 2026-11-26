"""Stabilize the single sigmoid in Liger's SwiGLU primal before differentiating."""
from jax._src.interpreters.mlir import make_ir_context
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import arith


def _defining_op(value, name):
    owner = value.owner
    return owner if isinstance(owner, (ir.Operation, ir.OpView)) and owner.name == name else None


def _match_sigmoid(div):
    one, denominator = div.operands
    const = _defining_op(one, "arith.constant")
    add = _defining_op(denominator, "arith.addf")
    if const is None or add is None or len(list(denominator.uses)) != 1:
        return None
    value = const.attributes["value"]
    if not isinstance(value, ir.DenseFPElementsAttr):
        return None
    dense = ir.DenseFPElementsAttr(value)
    if not dense.is_splat or ir.FloatAttr(dense.get_splat_value()).value != 1.0:
        return None
    a, b = add.operands
    term = a if b == one else b if a == one else None
    if term is None or len(list(term.uses)) != 1:
        return None
    exp = _defining_op(term, "math.exp")
    if exp is None:
        return None
    neg = _defining_op(exp.operands[0], "arith.negf")
    if neg is None:
        return None
    return one, exp, neg.operands[0]


def stabilize_sigmoid(generic_mlir: str) -> str:
    """Use exp(-abs(x)) and a sign-dependent numerator for 1/(1+exp(-x))."""
    with make_ir_context() as ctx, ir.Location.unknown():
        ctx.allow_unregistered_dialects = True
        mod = ir.Module.parse(generic_mlir)
        matches = []

        def inspect(op):
            if op.name == "arith.divf":
                match = _match_sigmoid(op)
                if match is not None:
                    matches.append((op, *match))
            return ir.WalkResult.ADVANCE

        mod.operation.walk(inspect)
        if len(matches) != 1:
            raise ValueError(f"expected one Liger sigmoid, found {len(matches)}")
        div, one, exp, x = matches[0]
        with ir.InsertionPoint(exp), exp.location:
            ty = ir.RankedTensorType(x.type)
            zero = ir.DenseElementsAttr.get_splat(ty, ir.FloatAttr.get(ty.element_type, 0.0))
            nonnegative = arith.CmpFOp(arith.CmpFPredicate.OGE, x, arith.ConstantOp(ty, zero)).result
            exp.operands[0] = arith.SelectOp(nonnegative, exp.operands[0], x).result
        with ir.InsertionPoint(div), div.location:
            div.operands[0] = arith.SelectOp(nonnegative, one, exp.results[0]).result
        mod.operation.verify()
        return mod.operation.get_asm(print_generic_op_form=True)
