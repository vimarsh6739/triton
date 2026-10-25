#include "RollingUpdateAnalysis.h"
#include "TileRematerialization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/StringSwitch.h"

namespace mlir::triton::detail {
namespace json = llvm::json;
json::Value symbol(StringRef name) {
  return json::Object{{"op", "symbol"}, {"name", name.str()}};
}
json::Value expression(StringRef op, json::Array args) {
  return json::Object{{"op", op.str()}, {"args", std::move(args)}};
}

ScalarExpression RollingPlan::opaque() {
  return {symbol("c" + std::to_string(nextOpaque++)), false};
}

bool RollingPlan::containsReduction(Value value) {
  auto *op = value.getDefiningOp();
  if (!op || op->getBlock() != scope)
    return false;
  if (isa<ReduceOp>(op))
    return true;
  return llvm::any_of(op->getOperands(),
                      [&](Value v) { return containsReduction(v); });
}

FailureOr<ScalarExpression> RollingPlan::extract(Value value,
                                                 SmallVector<int> axes,
                                                 ReductionState &consumer) {
  auto *op = value.getDefiningOp();
  if (!op || op->getBlock() != scope)
    return opaque();
  if (auto reduction = dyn_cast<ReduceOp>(op)) {
    // A projected reduction is a parameter; other reductions are payloads.
    if (llvm::is_contained(axes, -1)) {
      if (llvm::any_of(op->getOperands(),
                       [&](Value v) { return containsReduction(v); }))
        return failure();
      return opaque();
    }
    auto dep = addState(op);
    if (failed(dep))
      return failure();
    auto out = dyn_cast<RankedTensorType>(consumer.result.getType());
    auto in = dyn_cast<RankedTensorType>(value.getType());
    if (in) {
      if (!out || axes.size() != in.getRank())
        return failure();
      int previous = -1;
      for (auto [axis, dim] : llvm::zip(axes, in.getShape())) {
        if (axis <= previous || axis >= out.getRank() ||
            (dim != 1 && dim != out.getDimSize(axis)))
          return failure();
        previous = axis;
      }
    }
    if (getElementTypeOrSelf(value.getType()) !=
        getElementTypeOrSelf(consumer.result.getType()))
      return failure();
    auto it = consumer.projections.find(value);
    if (it != consumer.projections.end() && it->second != axes)
      return failure();
    if (it == consumer.projections.end()) {
      consumer.projections[value] = axes;
      consumer.dependencies.push_back(*dep);
    }
    return ScalarExpression{symbol((*dep)->name), true};
  }
  if (auto broadcast = dyn_cast<BroadcastOp>(op))
    return extract(broadcast.getSrc(), axes, consumer);
  if (auto expand = dyn_cast<ExpandDimsOp>(op)) {
    axes.erase(axes.begin() + expand.getAxis());
    return extract(expand.getSrc(), axes, consumer);
  }
  if (auto trans = dyn_cast<TransOp>(op)) {
    SmallVector<int> source(axes.size());
    for (auto [i, axis] : llvm::enumerate(trans.getOrder()))
      source[axis] = axes[i];
    return extract(trans.getSrc(), source, consumer);
  }
  if (auto splat = dyn_cast<SplatOp>(op))
    return extract(splat.getSrc(), {}, consumer);
  Attribute attr;
  if (matchPattern(value, m_Constant(&attr))) {
    if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
      if (!dense.isSplat())
        return opaque();
      attr = dense.getSplatValue<Attribute>();
    }
    if (auto number = dyn_cast<FloatAttr>(attr)) {
      if (number.getValue().isFinite())
        return ScalarExpression{
            json::Object{{"op", "constant"},
                         {"value", number.getValueAsDouble()}},
            false};
    }
    return opaque();
  }
  if (isa<LoadOp, DotOp>(op)) {
    if (llvm::any_of(op->getOperands(),
                     [&](Value v) { return containsReduction(v); }))
      return failure();
    return opaque();
  }
  if (!op->hasTrait<OpTrait::Elementwise>() || !isMemoryEffectFree(op)) {
    if (containsReduction(value))
      return failure();
    return opaque();
  }
  json::Array operands;
  bool dependent = false;
  for (Value operand : op->getOperands()) {
    auto expr = extract(
        operand,
        isa<RankedTensorType>(operand.getType()) ? axes : SmallVector<int>{},
        consumer);
    if (failed(expr))
      return failure();
    dependent |= expr->dependent;
    operands.push_back(std::move(expr->value));
  }
  if (!dependent)
    return opaque();
  if (isa<arith::ExtFOp, arith::TruncFOp>(op)) {
    if (!allowPrecisionChanges)
      return failure();
    return ScalarExpression{std::move(operands.front()), true};
  }
  StringRef name = llvm::StringSwitch<StringRef>(op->getName().getStringRef())
                       .Case("arith.addf", "add")
                       .Case("arith.subf", "sub")
                       .Case("arith.mulf", "mul")
                       .Case("arith.divf", "div")
                       .Case("math.exp", "exp")
                       .Case("math.exp2", "exp2")
                       .Case("math.log", "log")
                       .Case("math.powf", "pow")
                       .Default("");
  if (name.empty())
    return failure();
  return ScalarExpression{expression(name, std::move(operands)), true};
}

FailureOr<ReductionState *> RollingPlan::addState(Operation *op) {
  if (op->getNumResults() != 1 || op->getBlock() != scope)
    return failure();
  Value result = op->getResult(0);
  if (auto it = stateFor.find(result); it != stateFor.end())
    return it->second;
  auto state = std::make_unique<ReductionState>();
  state->op = op;
  state->result = result;
  state->name = "r" + std::to_string(storage.size());
  auto *node = state.get();
  storage.push_back(std::move(state));
  stateFor[result] = node;
  stateValues.insert(result);
  FailureOr<ScalarExpression> term = failure();
  if (auto reduce = dyn_cast<ReduceOp>(op)) {
    auto *combiner = reduce.getSingleCombiner();
    if (!combiner ||
        reduce.getCombineOp().front().getOperations().size() != 2 ||
        !isa<arith::AddFOp, arith::MaximumFOp, arith::MaxNumFOp,
             arith::MinimumFOp, arith::MinNumFOp>(combiner))
      return failure();
    auto inputType = cast<RankedTensorType>(reduce.getOperand(0).getType());
    if (inputType.getEncoding() ||
        inputType.getDimSize(reduce.getAxis()) != extent)
      return failure();
    node->combiner = combiner->getName().getStringRef();
    SmallVector<int> axes;
    int outputAxis = 0;
    for (int i = 0; i < inputType.getRank(); ++i)
      axes.push_back(i == reduce.getAxis() ? -1 : outputAxis++);
    term = extract(reduce.getOperand(0), axes, *node);
  } else if (auto dot = dyn_cast<DotOp>(op)) {
    auto aType = cast<RankedTensorType>(dot.getA().getType());
    if (aType.getRank() != 2 || aType.getEncoding() ||
        aType.getDimSize(1) != extent || tileSize < 16 ||
        !(matchPattern(dot.getC(), m_Zero()) ||
          matchPattern(dot.getC(), m_AnyZeroFloat())))
      return failure();
    node->combiner = "arith.addf";
    auto a = extract(dot.getA(), {0, -1}, *node);
    auto b = extract(dot.getB(), {-1, 1}, *node);
    if (failed(a) || failed(b))
      return failure();
    json::Array product;
    product.push_back(std::move(a->value));
    product.push_back(std::move(b->value));
    term = ScalarExpression{expression("mul", std::move(product)),
                            a->dependent || b->dependent};
  } else {
    return failure();
  }
  if (failed(term))
    return failure();
  if (!node->dependencies.empty()) {
    if (node->combiner != "arith.addf")
      return failure();
    json::Array names;
    for (auto *dep : node->dependencies)
      names.push_back(dep->name);
    json::Object request{{"expression", std::move(term->value)},
                         {"parameters", std::move(names)},
                         {"reducer", "add"}};
    auto correction =
        deriveAlgebraicCorrection(request, pythonExecutable, reason);
    if (failed(correction))
      return failure();
    node->correction = std::move(*correction);
  }
  order.push_back(node);
  return node;
}

void RollingPlan::collectCone(Operation *op) {
  if (!op || op->getBlock() != scope || !cone.insert(op).second)
    return;
  for (Value operand : op->getOperands())
    collectCone(operand.getDefiningOp());
}

LogicalResult RollingPlan::analyze() {
  auto target = addState(root);
  if (failed(target)) {
    if (reason.empty())
      reason = "unsupported reduction or dependent map expression";
    return failure();
  }
  if ((*target)->dependencies.empty()) {
    reason = "no dependent reductions";
    return failure();
  }
  collectCone(root);
  reason = "producer cannot be rematerialized at the requested tile shape";
  for (auto *state : order) {
    if (!isa<FloatType>(getElementTypeOrSelf(state->result.getType())))
      return failure();
    if (auto reduce = dyn_cast<ReduceOp>(state->op)) {
      if (failed(TileRematerializer::validate(reduce.getOperand(0),
                                              reduce.getAxis(), extent, scope,
                                              stateValues)))
        return failure();
    } else {
      auto dot = cast<DotOp>(state->op);
      if (failed(TileRematerializer::validate(dot.getA(), 1, extent, scope,
                                              stateValues)) ||
          failed(TileRematerializer::validate(dot.getB(), 0, extent, scope,
                                              stateValues)))
        return failure();
    }
  }
  // Moving a load across a write/unknown effect can change its value even if
  // the intervening operation does not use this SSA cone.
  bool sawLoad = false;
  reason = "intervening memory effect";
  for (Operation &op : *scope) {
    if (&op == root)
      break;
    sawLoad |= cone.contains(&op) && isa<LoadOp>(op);
    if (sawLoad && !isa<LoadOp>(op) && !isMemoryEffectFree(&op))
      return failure();
  }
  for (Operation *op : cone) {
    for (Value value : op->getResults()) {
      bool state = stateValues.contains(value);
      auto ty = dyn_cast<RankedTensorType>(value.getType());
      bool full = ty && isa<FloatType>(ty.getElementType()) &&
                  llvm::is_contained(ty.getShape(), extent) &&
                  !isa<arith::ConstantOp>(op);
      for (Operation *user : value.getUsers()) {
        if (cone.contains(user))
          continue;
        if (state) {
          reason = "external state use before loop insertion point";
          if (user->getBlock() != scope || !root->isBeforeInBlock(user))
            return failure();
        } else if (full) {
          reason = "externally observed full-width intermediate";
          return failure();
        }
      }
    }
  }
  return success();
}

} // namespace mlir::triton::detail
