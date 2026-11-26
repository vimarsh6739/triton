// Apply only inside the normalized-coordinate primal. Finite y allows
// canonicalization of delta * y after specializing primal delta to zero.
transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
  %mul = transform.structured.match ops{["arith.mulf"]} in %root : (!transform.any_op) -> !transform.any_op
  %finite = transform.param.constant #arith.fastmath<nnan,ninf,nsz> -> !transform.any_param
  transform.annotate %mul "fastmath" = %finite : !transform.any_op, !transform.any_param
  transform.yield
}
