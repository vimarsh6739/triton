#ifndef TRITON_TRANSFORMS_ALGEBRAIC_CORRECTION_H
#define TRITON_TRANSFORMS_ALGEBRAIC_CORRECTION_H

#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include <string>

namespace mlir::triton {

enum class ReferenceDomain { Finite, Nonzero, Positive };

struct AlgebraicCorrection {
  llvm::json::Value expression;
  llvm::StringMap<ReferenceDomain> domains;
};

// Ask the symbolic helper for a proved correction. The returned JSON owns its
// contents. No IR is changed when the helper cannot establish a correction.
// Each parameter has a domain and a verified fallback: zero for Finite, one
// for Nonzero or Positive. Callers must guard reference values accordingly.
FailureOr<AlgebraicCorrection>
deriveAlgebraicCorrection(const llvm::json::Object &request,
                          llvm::StringRef pythonExecutable,
                          std::string &reason);

// Symbol values must already have resultType, including any required broadcast.
// Validate the complete expression before emitting operations.
FailureOr<Value> emitAlgebraicCorrection(
    OpBuilder &builder, Location loc, const llvm::json::Value &expression,
    Type resultType, llvm::function_ref<Value(llvm::StringRef)> lookupSymbol);

} // namespace mlir::triton

#endif
