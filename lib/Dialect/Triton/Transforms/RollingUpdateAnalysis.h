#ifndef TRITON_TRANSFORMS_ROLLING_UPDATE_ANALYSIS_H
#define TRITON_TRANSFORMS_ROLLING_UPDATE_ANALYSIS_H

#include "AlgebraicCorrection.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <string>

namespace mlir::triton::detail {

llvm::json::Value symbol(StringRef name);

struct ReductionState {
  Operation *op;
  Value result;
  std::string name;
  StringRef combiner;
  SmallVector<ReductionState *> dependencies;
  DenseMap<Value, SmallVector<int>> projections;
  AlgebraicCorrection correction{symbol("acc"), {}};
};
struct ScalarExpression {
  llvm::json::Value value;
  bool dependent;
};

// Analyze the entire producer cone before modifying it. All algebraic repairs
// are derived from SSA expressions, independently of producer/kernel names.
class RollingPlan {
public:
  RollingPlan(Operation *root, int64_t extent, int64_t tileSize,
              bool allowPrecisionChanges, StringRef pythonExecutable)
      : root(root), scope(root->getBlock()), extent(extent), tileSize(tileSize),
        allowPrecisionChanges(allowPrecisionChanges),
        pythonExecutable(pythonExecutable) {}
  LogicalResult analyze();
  LogicalResult rewrite(IRRewriter &rewriter,
                        llvm::SmallPtrSetImpl<Operation *> &erased);
  std::string reason;

private:
  FailureOr<ReductionState *> addState(Operation *op);
  FailureOr<ScalarExpression> extract(Value value, SmallVector<int> axes,
                                      ReductionState &consumer);
  ScalarExpression opaque();
  bool containsReduction(Value value);
  void collectCone(Operation *op);
  Value project(OpBuilder &builder, Location loc, Value value,
                ArrayRef<int> axes, Type resultType);

  Operation *root;
  Block *scope;
  int64_t extent, tileSize;
  bool allowPrecisionChanges;
  std::string pythonExecutable;
  unsigned nextOpaque = 0;
  SmallVector<std::unique_ptr<ReductionState>> storage;
  SmallVector<ReductionState *> order;
  DenseMap<Value, ReductionState *> stateFor;
  DenseSet<Value> stateValues;
  llvm::SmallPtrSet<Operation *, 32> cone;
};

} // namespace mlir::triton::detail

#endif
