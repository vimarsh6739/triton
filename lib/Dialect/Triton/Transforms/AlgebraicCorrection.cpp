#include "AlgebraicCorrection.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>

namespace mlir::triton {

static LogicalResult validateCorrectionExpression(
    const llvm::json::Value &expression,
    llvm::function_ref<bool(llvm::StringRef)> validSymbol);

FailureOr<AlgebraicCorrection>
deriveAlgebraicCorrection(const llvm::json::Object &request,
                          llvm::StringRef pythonExecutable,
                          std::string &reason) {
  reason.clear();
  auto program = llvm::sys::findProgramByName(pythonExecutable);
  if (!program) {
    reason = "Cannot find the symbolic helper's Python executable: " +
             program.getError().message();
    return failure();
  }

  llvm::SmallString<256> input;
  llvm::raw_svector_ostream(input)
      << llvm::json::Value(llvm::json::Object(request));
  if (input.size() > 65536) {
    reason = "Symbolic correction request exceeds 64 KiB";
    return failure();
  }
  llvm::SmallString<128> paths[3];
  llvm::FileRemover cleanup[3];
  for (unsigned i = 0; i < 3; ++i) {
    if (auto error = llvm::sys::fs::createTemporaryFile("triton-correction",
                                                        "json", paths[i])) {
      reason = "Cannot create symbolic helper input/output: " + error.message();
      return failure();
    }
    cleanup[i].setFile(paths[i]);
  }
  {
    std::error_code error;
    llvm::raw_fd_ostream stream(paths[0], error);
    if (error) {
      reason = "Cannot open symbolic helper input: " + error.message();
      return failure();
    }
    stream << input;
    stream.close();
    if (stream.has_error()) {
      reason =
          "Cannot write symbolic helper input: " + stream.error().message();
      stream.clear_error();
      return failure();
    }
  }

  llvm::StringRef args[] = {*program, "-m", "triton.tools.rolling_update"};
  std::optional<llvm::StringRef> redirects[] = {paths[0].str(), paths[1].str(),
                                                paths[2].str()};
  std::string executionError;
  int status =
      llvm::sys::ExecuteAndWait(*program, args, std::nullopt, redirects,
                                /*SecondsToWait=*/10,
                                /*MemoryLimit=*/0, &executionError);
  if (status != 0) {
    reason = "Symbolic helper failed (status " + std::to_string(status) + ")";
    if (!executionError.empty())
      reason += ": " + executionError;
    uint64_t size = 0;
    if (!llvm::sys::fs::file_size(paths[2], size) && size != 0) {
      auto diagnostic = llvm::MemoryBuffer::getFileSlice(
          paths[2], std::min<uint64_t>(size, 4096), 0);
      if (diagnostic)
        reason += ": " + (*diagnostic)->getBuffer().str();
    }
    return failure();
  }

  uint64_t size = 0;
  if (auto error = llvm::sys::fs::file_size(paths[1], size)) {
    reason = "Cannot inspect symbolic helper output: " + error.message();
    return failure();
  }
  if (size > 65536) {
    reason = "Symbolic correction response exceeds 64 KiB";
    return failure();
  }
  auto output = llvm::MemoryBuffer::getFile(paths[1]);
  if (!output) {
    reason =
        "Cannot read symbolic helper output: " + output.getError().message();
    return failure();
  }
  auto response = llvm::json::parse((*output)->getBuffer());
  if (!response) {
    reason = "Invalid symbolic helper response: " +
             llvm::toString(response.takeError());
    return failure();
  }
  auto *object = response->getAsObject();
  if (!object || !object->getBoolean("ok").value_or(false)) {
    reason = object
                 ? object->getString("reason")
                       .value_or("Symbolic helper did not prove a correction")
                       .str()
                 : "Expected a symbolic helper response object";
    return failure();
  }
  auto *correction = object->get("correction");
  if (!correction || !correction->getAsObject()) {
    reason = "Symbolic helper omitted the correction expression";
    return failure();
  }
  auto *parameters = request.getArray("parameters");
  auto *domainObjects = object->getObject("domains");
  if (!parameters || !domainObjects ||
      parameters->size() != domainObjects->size()) {
    reason = "Symbolic helper domains do not match the requested parameters";
    return failure();
  }
  llvm::StringMap<ReferenceDomain> domains;
  for (const auto &parameter : *parameters) {
    auto name = parameter.getAsString();
    auto *domain = name ? domainObjects->getObject(*name) : nullptr;
    auto kind = domain ? domain->getString("kind") : std::nullopt;
    auto fallback = domain ? domain->getNumber("fallback") : std::nullopt;
    if (!name || name->empty() || !kind || !fallback) {
      reason = "Symbolic helper omitted a parameter domain or fallback";
      return failure();
    }
    ReferenceDomain value;
    if (*kind == "finite" && *fallback == 0)
      value = ReferenceDomain::Finite;
    else if (*kind == "nonzero" && *fallback == 1)
      value = ReferenceDomain::Nonzero;
    else if (*kind == "positive" && *fallback == 1)
      value = ReferenceDomain::Positive;
    else {
      reason =
          "Unsupported symbolic helper domain or fallback for " + name->str();
      return failure();
    }
    if (!domains.try_emplace(*name, value).second) {
      reason = "Duplicate symbolic correction parameter: " + name->str();
      return failure();
    }
  }
  if (failed(validateCorrectionExpression(*correction, [&](StringRef name) {
        if (name == "acc")
          return true;
        return (name.consume_front("old_") || name.consume_front("new_")) &&
               domains.contains(name);
      }))) {
    reason = "Symbolic helper produced an unsupported correction expression";
    return failure();
  }
  return AlgebraicCorrection{std::move(*correction), std::move(domains)};
}

// Supported powers have direct TTIR lowerings. SymPy encodes rational half
// powers as div(constant(+/-1), constant(2)), not as floating-point constants.
static std::optional<double>
getSupportedPower(const llvm::json::Value &expression) {
  auto *object = expression.getAsObject();
  if (!object)
    return std::nullopt;
  std::optional<double> power;
  if (object->getString("op") == "constant") {
    power = object->getNumber("value");
  } else if (object->getString("op") == "div") {
    auto *args = object->getArray("args");
    if (!args || args->size() != 2)
      return std::nullopt;
    auto *numerator = (*args)[0].getAsObject();
    auto *denominator = (*args)[1].getAsObject();
    if (!numerator || !denominator ||
        numerator->getString("op") != "constant" ||
        denominator->getString("op") != "constant")
      return std::nullopt;
    auto n = numerator->getNumber("value");
    auto d = denominator->getNumber("value");
    if (n && d && std::abs(*n) == 1 && *d == 2)
      power = *n / 2;
  }
  if (power && std::isfinite(*power) &&
      (std::abs(*power) == 0.5 ||
       (*power >= -64 && *power <= 64 && std::trunc(*power) == *power)))
    return power;
  return std::nullopt;
}

static LogicalResult validateCorrectionExpression(
    const llvm::json::Value &expression,
    llvm::function_ref<bool(llvm::StringRef)> validSymbol) {
  unsigned remaining = 256;
  auto validate = [&](auto &&self, const llvm::json::Value &node,
                      unsigned depth) -> LogicalResult {
    if (remaining == 0 || depth > 32)
      return failure();
    --remaining;
    auto *object = node.getAsObject();
    auto op = object ? object->getString("op") : std::nullopt;
    if (!op)
      return failure();
    if (*op == "symbol") {
      auto name = object->getString("name");
      return success(name && validSymbol(*name));
    }
    if (*op == "constant") {
      auto value = object->getNumber("value");
      return success(value && std::isfinite(*value));
    }
    unsigned arity = (*op == "exp" || *op == "exp2" || *op == "log") ? 1
                     : (*op == "add" || *op == "sub" || *op == "mul" ||
                        *op == "div" || *op == "pow")
                         ? 2
                         : 0;
    auto *args = object->getArray("args");
    if (!arity || !args || args->size() != arity)
      return failure();
    for (const auto &arg : *args)
      if (failed(self(self, arg, depth + 1)))
        return failure();
    if (*op == "pow" && !getSupportedPower((*args)[1]))
      return failure();
    return success();
  };
  return validate(validate, expression, 0);
}

FailureOr<Value> emitAlgebraicCorrection(
    OpBuilder &builder, Location loc, const llvm::json::Value &expression,
    Type resultType, llvm::function_ref<Value(llvm::StringRef)> lookupSymbol) {
  Type elementType = resultType;
  auto tensorType = dyn_cast<RankedTensorType>(resultType);
  if (tensorType) {
    if (!tensorType.hasStaticShape())
      return failure();
    elementType = tensorType.getElementType();
  }
  if (!isa<FloatType>(elementType))
    return failure();

  llvm::StringMap<Value> symbols;
  if (failed(validateCorrectionExpression(expression, [&](StringRef name) {
        if (symbols.contains(name))
          return true;
        Value value = lookupSymbol(name);
        if (!value || value.getType() != resultType)
          return false;
        symbols[name] = value;
        return true;
      })))
    return failure();

  auto constant = [&](double value) -> Value {
    auto scalar = builder.getFloatAttr(elementType, value);
    TypedAttr attr = scalar;
    if (tensorType)
      attr = DenseElementsAttr::get(tensorType, scalar);
    return arith::ConstantOp::create(builder, loc, resultType, attr);
  };
  auto emit = [&](auto &&self, const llvm::json::Value &node) -> Value {
    const auto &object = *node.getAsObject();
    auto op = *object.getString("op");
    if (op == "symbol")
      return symbols.lookup(*object.getString("name"));
    if (op == "constant")
      return constant(*object.getNumber("value"));
    const auto &args = *object.getArray("args");
    Value lhs = self(self, args[0]);
    if (op == "exp")
      return math::ExpOp::create(builder, loc, lhs);
    if (op == "exp2")
      return math::Exp2Op::create(builder, loc, lhs);
    if (op == "log")
      return math::LogOp::create(builder, loc, lhs);
    // Avoid math.powf, which has no TTIR-to-TTGIR conversion. Half powers
    // preserve their real domain through sqrt; no log/exp identity is assumed.
    if (op == "pow") {
      double exponent = *getSupportedPower(args[1]);
      Value result;
      if (std::abs(exponent) == 0.5) {
        result = math::SqrtOp::create(builder, loc, lhs);
      } else {
        unsigned power = static_cast<unsigned>(std::abs(exponent));
        result = constant(1);
        while (power) {
          if (power & 1)
            result = arith::MulFOp::create(builder, loc, result, lhs);
          power >>= 1;
          if (power)
            lhs = arith::MulFOp::create(builder, loc, lhs, lhs);
        }
      }
      return exponent < 0
                 ? arith::DivFOp::create(builder, loc, constant(1), result)
                 : result;
    }
    Value rhs = self(self, args[1]);
    if (op == "add")
      return arith::AddFOp::create(builder, loc, lhs, rhs);
    if (op == "sub")
      return arith::SubFOp::create(builder, loc, lhs, rhs);
    if (op == "mul")
      return arith::MulFOp::create(builder, loc, lhs, rhs);
    if (op == "div")
      return arith::DivFOp::create(builder, loc, lhs, rhs);
    llvm_unreachable("Expression was validated before emission");
  };
  return emit(emit, expression);
}

} // namespace mlir::triton
