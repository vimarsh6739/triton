#ifndef TRITON_PASSES
#define TRITON_PASSES

include "mlir/Pass/PassBase.td"

def TritonCombineOps : Pass</*cli-arg*/"triton-combine", /*Op*/"mlir::ModuleOp"> {
  let summary = "combine ops";
  let description = [{
    This pass aims to optimize the five following patterns:
    - `dot(a, b, 0) + c => dot(a, b, c)`

    - `addptr(addptr(ptr, idx0), idx1) => addptr(ptr, AddI(idx0, idx1))`

    - `select(cond, load(ptrs, broadcast(cond), ???), other) =>
         load(ptrs, broadcast(cond), other)`

    - `broadcast(constant) => reshaped_constant`
    - `torch.sum(x[:,:,None].expand(-1,-1,n) * y[None,:,:].expand(m,-1,-1),1)
       => dot(x,y,splat(0))`
  }];

  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def TritonReorderBroadcast : Pass</*cli-arg*/"triton-reorder-broadcast", /*Op*/"mlir::ModuleOp"> {
  let summary = "Moves broadcast and splat after elementwise operations";
  let description = [{
    The purpose of this pass is to transform:
      - `elementwise(broadcast(a)) => broadcast(elementwise(a))`
      - `elementwise(splat(a), splat(b), ...) => splat(elementwise(a, b, ...))`
    In the event of a match, the broadcast (or splat) operation is delayed
    and performed after the ElementWise operation.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonRewriteTensorDescriptorToPointer : Pass</*cli-arg*/"triton-rewrite-tensor-descriptor-to-pointer", /*Op*/"mlir::ModuleOp"> {
  let summary = "Rewrite load/stores of tensor descriptors into pointer load/stores";
  let description = [{
    This pass rewrites all load/store semantics initiated by a `tt.make_tensor_descriptor` into pointer semantics. After
    this pass, `tt.make_tensor_descriptor`  will disappear, and it generates logics to compute the pointer/mask/other
    for each load/store.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonLoopUnroll : Pass</*cli-arg*/"triton-loop-unroll", /*Op*/"mlir::ModuleOp"> {
  let summary = "Loop unroller";
  let description = [{
    The pass unrolls a scf loop with tt.loop_unroll_factor attribute. The attribute specialises how many iterations
    the loop should be unrolled.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonLoopInvariantCodeMotion : Pass</*cli-arg*/"triton-licm", /*Op*/"mlir::ModuleOp"> {
  let summary = "MLIR's LICM plus hoist load ops out of loops with masks.";
  let description = [{
    This pass uses MLIR's LICM pass as base. Additionally, it hoists load ops
    out of loops that consists of pure/read-only ops. For scf.for loops, it
    generates a trip-count check. For scf.while loops, it clones the condition
    from the before body.
  }];

  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonLoopAwareCSE : Pass<"triton-loop-aware-cse", "mlir::ModuleOp"> {
  let summary = "CSE within loop bodies";

  let description = [{
    The `triton-loop-aware-cse` pass performs recursive common subexpression
    elimination within loop bodies. Unlike regular CSE, which is a single-pass
    greedy algorithm, this pass can recursively eliminate loop iteration
    arguments and subcomputations that always have the same value.
  }];
}

def TritonRollingUpdate : Pass<"triton-rolling-update", "mlir::ModuleOp"> {
  let summary = "Fuse dependent TTIR reductions with algebraically derived repairs";
  let description = [{
    Applies Neptune's rolling-update transformation directly to unencoded TTIR.
    Extracts expressions between reductions, proves their additive repair using
    SymPy, and rematerializes producer tiles inside an scf.for loop. Unsupported
    expressions or unsafe memory ordering leave the candidate unchanged.
    Requires explicit permission for floating-point reassociation. Crossing
    precision-changing casts requires a separate explicit permission.

    The supported algebraic case is g(r,c) = A(r) B(c), with an additive
    consumer and a nonzero parameter factor A. The helper proves both term
    correction and compatibility with addition; it does not prove numerical
    stability for arbitrary input ranges. In particular, changing exponential
    references can introduce overflow or underflow. The proof applies over real
    arithmetic where the source map is defined, not to all IEEE exceptional
    cases. Partial reductions use valid temporary references and a final repair.

    The pass is opt-in and runs before layout assignment. It supports fixed
    reduction extents divisible by tile-size, simple add/max/min reductions,
    and rank-two dots with zero accumulators. It does not perform split-K or
    autotuning. SymPy is needed only by the selected helper interpreter; a
    missing helper or unsupported proof leaves the candidate unchanged.
    See https://arxiv.org/abs/2510.08726 for Neptune's transformation.
  }];
  let options = [
    Option<"tileSize", "tile-size", "int", "32", "Rolling reduction tile size">,
    Option<"allowReassociation", "allow-reassociation", "bool", "false",
           "Permit real-arithmetic reassociation of floating-point reductions">,
    Option<"allowPrecisionChanges", "allow-precision-changes", "bool", "false",
           "Permit changed rounding across floating-point casts">,
    Option<"pythonExecutable", "python-executable", "std::string", "\"python3\"",
           "Python interpreter with Triton and SymPy for correction proofs">,
    Option<"reportFailures", "report-failures", "bool", "false",
           "Emit remarks explaining declined rolling-update candidates">
  ];
  let dependentDialects = ["mlir::triton::TritonDialect", "mlir::arith::ArithDialect",
                          "mlir::math::MathDialect", "mlir::scf::SCFDialect"];
}

#endif
