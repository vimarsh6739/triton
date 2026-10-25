#include "RollingUpdateAnalysis.h"
#include "TileRematerialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONROLLINGUPDATE
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace detail {
static Value floatConstant(OpBuilder &b, Location loc, Type type,
                           double value) {
  auto attr = b.getFloatAttr(getElementTypeOrSelf(type), value);
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return arith::ConstantOp::create(b, loc, tensor,
                                     DenseElementsAttr::get(tensor, attr));
  return arith::ConstantOp::create(b, loc, type, attr);
}

Value RollingPlan::project(OpBuilder &b, Location loc, Value value,
                           ArrayRef<int> axes, Type resultType) {
  if (value.getType() == resultType &&
      llvm::all_of(llvm::enumerate(axes),
                   [](auto item) { return item.index() == item.value(); }))
    return value;
  auto target = dyn_cast<RankedTensorType>(resultType);
  if (!target)
    return value;
  if (!isa<RankedTensorType>(value.getType()))
    return SplatOp::create(b, loc, target, value);
  for (int axis = 0; axis < target.getRank(); ++axis)
    if (!llvm::is_contained(axes, axis))
      value = ExpandDimsOp::create(b, loc, value, axis);
  if (value.getType() != resultType)
    value = BroadcastOp::create(b, loc, resultType, value);
  return value;
}

// A consumer may use a partial reduction before it is in the map's domain.
// Represent its accumulator at a valid reference instead. The same reference
// must be used for both the tile's map and the accumulated-state correction.
static Value referenceValue(OpBuilder &b, Location loc, Value value,
                            ReferenceDomain domain) {
  Type type = value.getType();
  Value zero = floatConstant(b, loc, type, 0);
  Value inf =
      floatConstant(b, loc, type, std::numeric_limits<double>::infinity());
  Value absolute = math::AbsFOp::create(b, loc, value);
  Value valid =
      arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OLT, absolute, inf);
  Value fallback = zero;
  if (domain != ReferenceDomain::Finite) {
    auto predicate = domain == ReferenceDomain::Nonzero
                         ? arith::CmpFPredicate::ONE
                         : arith::CmpFPredicate::OGT;
    Value inDomain = arith::CmpFOp::create(b, loc, predicate, value, zero);
    valid = arith::AndIOp::create(b, loc, valid, inDomain);
    fallback = floatConstant(b, loc, type, 1);
  }
  return arith::SelectOp::create(b, loc, valid, value, fallback);
}

LogicalResult RollingPlan::rewrite(IRRewriter &rewriter,
                                   llvm::SmallPtrSetImpl<Operation *> &erased) {
  rewriter.setInsertionPoint(root);
  auto loc = root->getLoc();
  SmallVector<Value> init;
  for (auto *state : order) {
    double identity = 0;
    if (state->combiner == "arith.maxnumf" ||
        state->combiner == "arith.maximumf")
      identity = -std::numeric_limits<double>::infinity();
    if (state->combiner == "arith.minnumf" ||
        state->combiner == "arith.minimumf")
      identity = std::numeric_limits<double>::infinity();
    init.push_back(
        floatConstant(rewriter, loc, state->result.getType(), identity));
  }
  auto lower = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
  auto upper = arith::ConstantIntOp::create(rewriter, loc, extent, 32);
  auto step = arith::ConstantIntOp::create(rewriter, loc, tileSize, 32);
  auto loop = scf::ForOp::create(rewriter, loc, lower, upper, step, init);
  rewriter.setInsertionPointToStart(loop.getBody());
  DenseMap<Value, Value> old, current;
  for (auto [state, arg] : llvm::zip(order, loop.getRegionIterArgs()))
    old[state->result] = arg;
  SmallVector<Value> next;
  for (auto *state : order) {
    DenseMap<Value, Value> oldReferences, newReferences = current;
    for (auto *dep : state->dependencies) {
      auto domain = state->correction.domains.lookup(dep->name);
      oldReferences[dep->result] =
          referenceValue(rewriter, loc, old.lookup(dep->result), domain);
      newReferences[dep->result] =
          referenceValue(rewriter, loc, current.lookup(dep->result), domain);
    }
    // References are consumer-specific: one map may accept zero while another
    // divides by this same state. CSE shares their parameter-independent tiles.
    TileRematerializer tiler(rewriter, extent, tileSize, loop.getInductionVar(),
                             scope, stateValues, newReferences);
    Value tileResult;
    if (auto reduce = dyn_cast<ReduceOp>(state->op)) {
      auto input = tiler.materialize(reduce.getOperand(0), reduce.getAxis());
      if (failed(input)) {
        rewriter.eraseOp(loop);
        return failure();
      }
      IRMapping mapping;
      mapping.map(reduce.getOperand(0), *input);
      tileResult = rewriter.clone(*reduce, mapping)->getResult(0);
    } else {
      auto dot = cast<DotOp>(state->op);
      auto a = tiler.materialize(dot.getA(), 1);
      auto b = tiler.materialize(dot.getB(), 0);
      if (failed(a) || failed(b)) {
        rewriter.eraseOp(loop);
        return failure();
      }
      auto copy = cast<DotOp>(rewriter.clone(*dot));
      copy.getAMutable().assign(*a);
      copy.getBMutable().assign(*b);
      tileResult = copy.getResult();
    }
    auto repair = emitAlgebraicCorrection(
        rewriter, loc, state->correction.expression, state->result.getType(),
        [&](StringRef name) -> Value {
          if (name == "acc")
            return old.lookup(state->result);
          for (auto *dep : state->dependencies) {
            Value value;
            if (name == "old_" + dep->name)
              value = oldReferences.lookup(dep->result);
            if (name == "new_" + dep->name)
              value = newReferences.lookup(dep->result);
            if (value)
              return project(rewriter, loc, value,
                             state->projections[dep->result],
                             state->result.getType());
          }
          return {};
        });
    if (failed(repair)) {
      rewriter.eraseOp(loop);
      return failure();
    }
    Value previous = *repair;
    if (!state->dependencies.empty()) {
      // Additive homomorphisms preserve zero. Avoid 0*inf when an empty prefix
      // is rescaled across a large (but valid) reference change.
      Value zero = floatConstant(rewriter, loc, state->result.getType(), 0);
      Value empty =
          arith::CmpFOp::create(rewriter, loc, arith::CmpFPredicate::OEQ,
                                old.lookup(state->result), zero);
      previous = arith::SelectOp::create(rewriter, loc, empty, zero, previous);
    }
    OperationState combine(loc, state->combiner);
    combine.addOperands({previous, tileResult});
    combine.addTypes(state->result.getType());
    Value updated = rewriter.create(combine)->getResult(0);
    current[state->result] = updated;
    next.push_back(updated);
  }
  rewriter.setInsertionPointToEnd(loop.getBody());
  if (!loop.getBody()->empty() && isa<scf::YieldOp>(loop.getBody()->back()))
    rewriter.eraseOp(&loop.getBody()->back());
  scf::YieldOp::create(rewriter, loc, next);
  rewriter.setInsertionPointAfter(loop);
  DenseMap<Value, Value> raw, finalized;
  for (auto [state, result] : llvm::zip(order, loop.getResults()))
    raw[state->result] = result;
  // Restore the actual final parameters, including singular ones. Guarding
  // these final references would change the source program's NaN/Inf behavior.
  for (auto *state : order) {
    DenseMap<Value, Value> references;
    for (auto *dep : state->dependencies)
      references[dep->result] =
          referenceValue(rewriter, loc, raw.lookup(dep->result),
                         state->correction.domains.lookup(dep->name));
    auto result = emitAlgebraicCorrection(
        rewriter, loc, state->correction.expression, state->result.getType(),
        [&](StringRef name) -> Value {
          if (name == "acc")
            return raw.lookup(state->result);
          for (auto *dep : state->dependencies) {
            Value value;
            if (name == "old_" + dep->name)
              value = references.lookup(dep->result);
            if (name == "new_" + dep->name)
              value = finalized.lookup(dep->result);
            if (value)
              return project(rewriter, loc, value,
                             state->projections[dep->result],
                             state->result.getType());
          }
          return {};
        });
    if (failed(result))
      return failure();
    Value finalValue = *result;
    Value unchanged;
    for (auto *dep : state->dependencies) {
      Value before =
          project(rewriter, loc, references.lookup(dep->result),
                  state->projections[dep->result], state->result.getType());
      Value after =
          project(rewriter, loc, finalized.lookup(dep->result),
                  state->projections[dep->result], state->result.getType());
      Value equal = arith::CmpFOp::create(
          rewriter, loc, arith::CmpFPredicate::OEQ, before, after);
      unchanged = unchanged
                      ? arith::AndIOp::create(rewriter, loc, unchanged, equal)
                      : equal;
    }
    if (unchanged)
      finalValue = arith::SelectOp::create(
          rewriter, loc, unchanged, raw.lookup(state->result), finalValue);
    finalized[state->result] = finalValue;
    rewriter.replaceUsesWithIf(state->result, finalValue, [&](OpOperand &use) {
      return !cone.contains(use.getOwner());
    });
  }
  // Original producers still reference each other. Erase dead consumers first
  // instead of creating uses of loop results before the loop dominates them.
  SmallVector<Operation *> original;
  for (Operation &op : *scope)
    if (cone.contains(&op))
      original.push_back(&op);
  for (Operation *op : llvm::reverse(original)) {
    if (isOpTriviallyDead(op)) {
      erased.insert(op);
      rewriter.eraseOp(op);
    }
  }
  return success();
}

} // namespace detail

namespace {
class RollingUpdatePass
    : public impl::TritonRollingUpdateBase<RollingUpdatePass> {
public:
  using TritonRollingUpdateBase::TritonRollingUpdateBase;
  void runOnOperation() override {
    if (!allowReassociation)
      return;
    if (tileSize <= 0 || !llvm::isPowerOf2_64(tileSize)) {
      getOperation().emitError(
          "rolling update tile size must be a positive power of two");
      return signalPassFailure();
    }
    SmallVector<Operation *> candidates;
    getOperation().walk([&](Operation *op) {
      if (isa<ReduceOp, DotOp>(op))
        candidates.push_back(op);
    });
    IRRewriter rewriter(&getContext());
    llvm::SmallPtrSet<Operation *, 32> erased;
    for (Operation *op : llvm::reverse(candidates)) {
      if (erased.contains(op) || op->use_empty())
        continue;
      int64_t extent = 0;
      if (auto reduce = dyn_cast<ReduceOp>(op))
        extent = cast<RankedTensorType>(reduce.getOperand(0).getType())
                     .getDimSize(reduce.getAxis());
      else {
        auto type = cast<RankedTensorType>(cast<DotOp>(op).getA().getType());
        if (type.getRank() != 2)
          continue;
        extent = type.getDimSize(1);
      }
      if (extent <= tileSize || extent % tileSize != 0 ||
          extent > std::numeric_limits<int32_t>::max())
        continue;
      detail::RollingPlan plan(op, extent, tileSize, allowPrecisionChanges,
                               pythonExecutable);
      if (failed(plan.analyze())) {
        if (reportFailures)
          op->emitRemark() << "rolling update skipped: " << plan.reason;
        continue;
      }
      if (failed(plan.rewrite(rewriter, erased))) {
        op->emitError("failed to emit a validated rolling-update plan");
        return signalPassFailure();
      }
    }
    DominanceInfo domInfo;
    eliminateCommonSubExpressions(rewriter, domInfo, getOperation());
    if (failed(applyPatternsGreedily(getOperation(),
                                     RewritePatternSet(&getContext()))))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::triton
