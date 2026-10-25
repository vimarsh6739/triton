#include "TileRematerialization.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton {
namespace {

// Propagate a logical axis, rather than matching dimension sizes: two distinct
// dimensions can have the same extent (e.g. sequence length and head
// dimension).
FailureOr<SmallVector<int>> operandAxes(Value value, int axis, int64_t extent,
                                        Block *scope,
                                        const DenseSet<Value> &states) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (axis < -1 || (type && type.getEncoding()) ||
      (axis >= 0 &&
       (!type || axis >= type.getRank() || type.getDimSize(axis) != extent)))
    return failure();
  if (states.contains(value)) {
    if (axis != -1)
      return failure();
    return SmallVector<int>{};
  }
  Operation *op = value.getDefiningOp();
  if (!op || op->getBlock() != scope) {
    if (axis != -1)
      return failure();
    return SmallVector<int>{};
  }
  if (op->getNumResults() != 1)
    return failure();

  SmallVector<int> axes(op->getNumOperands(), -1);
  if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    if (type) {
      auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (!dense || !dense.isSplat())
        return failure();
    }
    return axes;
  }
  if (isa<MakeRangeOp, GetProgramIdOp, GetNumProgramsOp>(op))
    return axes;
  if (isa<SplatOp>(op))
    return axes;
  if (auto broadcast = dyn_cast<BroadcastOp>(op)) {
    if (axis >= 0 && broadcast.getSrc().getType().getDimSize(axis) != 1)
      axes[0] = axis;
    return axes;
  }
  if (auto expand = dyn_cast<ExpandDimsOp>(op)) {
    if (axis == expand.getAxis())
      return failure();
    if (axis >= 0)
      axes[0] = axis - (axis > expand.getAxis());
    return axes;
  }
  if (auto trans = dyn_cast<TransOp>(op)) {
    if (axis >= 0)
      axes[0] = trans.getOrder()[axis];
    return axes;
  }
  if (auto dot = dyn_cast<DotOp>(op)) {
    if (!type || type.getRank() != 2)
      return failure();
    if (axis >= 0)
      axes = {axis == 0 ? 0 : -1, axis == 1 ? 1 : -1, axis};
    return axes;
  }
  if (auto reduce = dyn_cast<ReduceOp>(op)) {
    Operation *combiner = reduce.getSingleCombiner();
    if (!combiner ||
        reduce.getCombineOp().front().getOperations().size() != 2 ||
        combiner->getName().getDialectNamespace() != "arith" ||
        !isMemoryEffectFree(combiner) || combiner->getNumRegions())
      return failure();
    if (axis >= 0)
      axes[0] = axis + (axis >= reduce.getAxis());
    return axes;
  }
  if (op->getNumRegions())
    return failure();
  if (auto load = dyn_cast<LoadOp>(op)) {
    if (load.getIsVolatile())
      return failure();
  } else if (!isa<AddPtrOp>(op)) {
    StringRef dialect = op->getName().getDialectNamespace();
    bool supported =
        dialect == "arith" || dialect == "math" || isa<ExternElementwiseOp>(op);
    if (!supported || !op->hasTrait<OpTrait::Elementwise>() ||
        !isMemoryEffectFree(op))
      return failure();
  }
  if (axis >= 0)
    for (auto [i, operand] : llvm::enumerate(op->getOperands()))
      if (isa<RankedTensorType>(operand.getType()))
        axes[i] = axis;
  return axes;
}

LogicalResult validateImpl(Value value, int axis, int64_t extent, Block *scope,
                           const DenseSet<Value> &states,
                           DenseSet<std::pair<Value, int>> &visited) {
  if (!visited.insert({value, axis}).second)
    return success();
  auto axes = operandAxes(value, axis, extent, scope, states);
  if (failed(axes))
    return failure();
  if (states.contains(value))
    return success();
  Operation *op = value.getDefiningOp();
  if (!op || op->getBlock() != scope)
    return success();
  for (auto [operand, operandAxis] : llvm::zip(op->getOperands(), *axes))
    if (failed(
            validateImpl(operand, operandAxis, extent, scope, states, visited)))
      return failure();
  return success();
}

Type tileType(Type type, int axis, int64_t tileSize) {
  if (axis < 0)
    return type;
  auto tensor = cast<RankedTensorType>(type);
  SmallVector<int64_t> shape(tensor.getShape());
  shape[axis] = tileSize;
  return tensor.clone(shape);
}

} // namespace

TileRematerializer::TileRematerializer(
    OpBuilder &builder, int64_t extent, int64_t tileSize, Value offset,
    Block *scope, const DenseSet<Value> &states,
    const DenseMap<Value, Value> &replacements)
    : builder(builder), extent(extent), tileSize(tileSize), offset(offset),
      scope(scope), states(states), replacements(replacements) {}

LogicalResult TileRematerializer::validate(Value value, int axis,
                                           int64_t extent, Block *scope,
                                           const DenseSet<Value> &states) {
  if (extent <= 0)
    return failure();
  DenseSet<std::pair<Value, int>> visited;
  return validateImpl(value, axis, extent, scope, states, visited);
}

bool TileRematerializer::dependsOnState(Value value) {
  if (states.contains(value))
    return true;
  auto found = stateDependencies.find(value);
  if (found != stateDependencies.end())
    return found->second;
  Operation *op = value.getDefiningOp();
  bool dependent = op && op->getBlock() == scope &&
                   llvm::any_of(op->getOperands(), [&](Value operand) {
                     return dependsOnState(operand);
                   });
  stateDependencies[value] = dependent;
  return dependent;
}

bool TileRematerializer::hasReplacements(Value value) {
  if (states.contains(value)) {
    auto found = replacements.find(value);
    return found != replacements.end() &&
           found->second.getType() == value.getType();
  }
  if (!dependsOnState(value))
    return true;
  return llvm::all_of(value.getDefiningOp()->getOperands(),
                      [&](Value operand) { return hasReplacements(operand); });
}

FailureOr<Value> TileRematerializer::materialize(Value value, int axis) {
  if (!offset.getType().isInteger(32) || tileSize <= 0 || tileSize > extent ||
      !llvm::isPowerOf2_64(tileSize) || extent % tileSize ||
      failed(validate(value, axis, extent, scope, states)) ||
      !hasReplacements(value))
    return failure();
  return materializeImpl(value, axis);
}

FailureOr<Value> TileRematerializer::materializeImpl(Value value, int axis) {
  if (states.contains(value))
    return replacements.lookup(value);
  if (axis < 0 && !dependsOnState(value))
    return value;
  auto key = std::make_pair(value, axis);
  auto found = cache.find(key);
  if (found != cache.end())
    return found->second;

  Operation *op = value.getDefiningOp();
  auto axes = operandAxes(value, axis, extent, scope, states);
  if (failed(axes))
    return failure();
  auto loc = op->getLoc();
  Type resultType = tileType(value.getType(), axis, tileSize);
  Value result;
  if (auto range = dyn_cast<MakeRangeOp>(op)) {
    Value start = offset;
    int64_t originalStart = range.getStartAttr().getInt();
    if (originalStart) {
      Value constant =
          arith::ConstantIntOp::create(builder, loc, originalStart, 32);
      start = arith::AddIOp::create(builder, loc, start, constant);
    }
    Value indices = MakeRangeOp::create(builder, loc, resultType, 0, tileSize);
    Value base = SplatOp::create(builder, loc, resultType, start);
    result = arith::AddIOp::create(builder, loc, indices, base);
  } else if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    auto dense = cast<DenseElementsAttr>(constant.getValue());
    auto attr = SplatElementsAttr::get(cast<RankedTensorType>(resultType),
                                       dense.getSplatValue<Attribute>());
    result = arith::ConstantOp::create(builder, loc, resultType, attr);
  } else {
    SmallVector<Value> operands;
    for (auto [operand, operandAxis] : llvm::zip(op->getOperands(), *axes)) {
      auto tiled = materializeImpl(operand, operandAxis);
      if (failed(tiled))
        return failure();
      operands.push_back(*tiled);
    }
    // The same source value can occur on different logical axes (e.g. both
    // operands of a square dot), so replace operands by position, not Value.
    Operation *clone = builder.clone(*op);
    clone->setOperands(operands);
    clone->getResult(0).setType(resultType);
    // Shape/alignment hints may no longer hold after replacing the range or
    // upstream state. Recompute them in later analyses instead of retaining
    // facts about the full-width producer.
    for (StringRef name : {"tt.divisibility", "tt.contiguity", "tt.constancy"})
      clone->removeAttr(name);
    result = clone->getResult(0);
  }
  cache[key] = result;
  return result;
}

} // namespace mlir::triton
