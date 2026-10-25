#ifndef TRITON_TRANSFORMS_TILE_REMATERIALIZATION_H
#define TRITON_TRANSFORMS_TILE_REMATERIALIZATION_H

#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include <utility>

namespace mlir::triton {

// Rebuild a producer slice with one logical dimension restricted to a tile.
// Axis -1 denotes an invariant value. The caller checks memory ordering and
// ensures reused invariant definitions dominate the builder's insertion point.
class TileRematerializer {
public:
  TileRematerializer(OpBuilder &builder, int64_t extent, int64_t tileSize,
                     Value offset, Block *scope, const DenseSet<Value> &states,
                     const DenseMap<Value, Value> &replacements);

  static LogicalResult validate(Value value, int axis, int64_t extent,
                                Block *scope, const DenseSet<Value> &states);
  FailureOr<Value> materialize(Value value, int axis);

private:
  FailureOr<Value> materializeImpl(Value value, int axis);
  bool dependsOnState(Value value);
  bool hasReplacements(Value value);

  OpBuilder &builder;
  int64_t extent, tileSize;
  Value offset;
  Block *scope;
  const DenseSet<Value> &states;
  // The caller fills this map in dependency order. Once an expression has been
  // materialized, replacements for its dependencies must not change.
  const DenseMap<Value, Value> &replacements;
  DenseMap<std::pair<Value, int>, Value> cache;
  DenseMap<Value, bool> stateDependencies;
};

} // namespace mlir::triton

#endif
