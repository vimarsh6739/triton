"""Check the local constant-folding matcher and its narrow preconditions."""
from jaxlib.mlir import ir

from fold import sum_of_zero


def test_positive_zero_sum():
    template = '''
    "tt.func"() ({
      %z = "arith.constant"() <{value = dense<VALUE> : tensor<2x8xf32>}> : () -> tensor<2x8xf32>
      %r = "tt.reduce"(%z) ({
        ^bb0(%a: f32, %b: f32):
          %v = "COMBINE"(%a, %b) : (f32, f32) -> f32
          "tt.reduce.return"(%v) : (f32) -> ()
      }) {axis = 1 : i32} : (tensor<2x8xf32>) -> tensor<2xf32>
      "tt.return"(%r) : (tensor<2xf32>) -> ()
    }) : () -> ()
    '''
    for value, combiner, matched in [('0.0', 'arith.addf', True), ('-0.0', 'arith.addf', False),
                                      ('1.0', 'arith.addf', False), ('0.0', 'arith.divf', False)]:
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            root = ir.Module.parse(template.replace('VALUE', value).replace('COMBINE', combiner))
            op = root.body.operations[0].regions[0].blocks[0].operations[1].operation
            assert (sum_of_zero(op) is not None) == matched
