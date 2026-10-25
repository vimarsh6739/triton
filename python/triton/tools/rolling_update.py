"""Derive reduction repairs for the TTIR rolling-update pass.

This implements the separable case of Neptune's algebraic correction analysis
(https://arxiv.org/html/2510.08726v2#S4): for g(r, c) = A(r) * B(c), invert
the multiplication and substitute the new parameters, then prove that the
repair distributes over the consumer's reducer. B may be an opaque payload.

Proofs use real arithmetic on finite inputs where the original expression is
defined. They do not establish floating-point equivalence or authorize crossing
casts. The caller must permit reassociation and handle empty reduction states.
The command-line protocol is one JSON request and one JSON response. Its
parameter domains describe safe effective references for partial reductions.
Consumers must use those same references for their tile computation and repair,
then repair to finalized producer values after the loop, in dependency order.
"""

import json
import math
import re
import sys

_MAX_NODES = 256
_MAX_DEPTH = 32
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}\Z")


def _valid_name(name):
    return (isinstance(name, str) and _NAME.fullmatch(name) is not None and name != "acc" and not name.startswith(
        ("old_", "new_")))


def _derive_correction(expression, parameters, reducer, *, include_domains=False):
    import sympy as sp

    if reducer != "add":
        raise ValueError("Only additive consumer reductions are supported")
    if (not isinstance(parameters, list) or len(parameters) > 16 or not all(_valid_name(name) for name in parameters)
            or len(set(parameters)) != len(parameters)):
        raise ValueError("Expected distinct parameter names without reserved prefixes")
    symbols = {name: sp.Symbol(name, real=True) for name in parameters}
    params = set(symbols.values())
    requirements = []
    remaining = _MAX_NODES

    def decode(node, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > _MAX_DEPTH:
            raise ValueError("Expression exceeds the supported size")
        if not isinstance(node, dict):
            raise ValueError("Expected an expression object")
        op = node.get("op")
        if op == "symbol":
            name = node.get("name")
            if not _valid_name(name):
                raise ValueError("Invalid or reserved symbol name")
            return symbols.setdefault(name, sp.Symbol(name, real=True))
        if op == "constant":
            value = node.get("value")
            if (type(value) not in (int, float) or (isinstance(value, int) and value.bit_length() > 256)
                    or not math.isfinite(value)):
                raise ValueError("Expected a finite numeric constant")
            return sp.Rational(value)
        arity = {"add": 2, "sub": 2, "mul": 2, "div": 2, "pow": 2, "exp": 1, "exp2": 1, "log": 1}
        args = node.get("args")
        if op not in arity or not isinstance(args, list) or len(args) != arity[op]:
            raise ValueError("Unsupported operation or operand count")
        args = [decode(arg, depth + 1) for arg in args]
        if op in ("pow", "exp2") and args[-1].is_number:
            if args[-1].is_real is not True or not -64 <= args[-1] <= 64:
                raise ValueError("Constant exponent exceeds the supported range")
        # Keep domain restrictions before SymPy can cancel a pole, e.g. r/r.
        if op == "div":
            requirements.append((args[1], "nonzero"))
        elif op == "log":
            requirements.append((args[0], "positive"))
        elif op == "pow":
            if args[1].is_integer is not True:
                requirements.append((args[0], "positive"))
            elif args[1].is_negative:
                requirements.append((args[0], "nonzero"))
        operations = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: a / b,
            "pow": lambda a, b: a**b,
            "exp": sp.exp,
            "exp2": lambda a: sp.Pow(2, a),
            "log": sp.log,
        }
        return operations[op](*args)

    expr = decode(expression)
    if expr.has(sp.oo, -sp.oo, sp.zoo, sp.nan, sp.I):
        raise ValueError("Expression has a non-finite or non-real constant")
    # Expand exponential sums in the exponent, but never expand polynomial
    # powers or products of sums: those can make a small input exponentially big.
    separated = sp.factor_terms(sp.expand_power_exp(expr))
    parameter_factors = []
    for factor in sp.Mul.make_args(separated):
        if factor.free_symbols & params:
            if factor.free_symbols - params:
                raise ValueError("Producer parameters cannot be separated from the payload")
            parameter_factors.append(factor)
    factor = sp.Mul(*parameter_factors)
    # A zero A loses information about B. A reciprocal is nonzero wherever the
    # original expression is finite and defined; exp and exp2 are also nonzero.
    numerator, _ = sp.fraction(factor)
    if numerator.is_zero is not False:
        raise ValueError("Cannot prove the parameter factor is nonzero")
    old = {symbols[name]: sp.Symbol("old_" + name, real=True) for name in parameters}
    new = {symbols[name]: sp.Symbol("new_" + name, real=True) for name in parameters}
    acc = sp.Symbol("acc", real=True)
    repair = sp.simplify(acc * factor.xreplace(new) / factor.xreplace(old))
    if sp.simplify(repair.subs(acc, expr.xreplace(old)) - expr.xreplace(new)) != 0:
        raise ValueError("Could not prove that the repair updates each term")
    x, y = sp.Dummy("x", real=True), sp.Dummy("y", real=True)
    if sp.simplify(repair.subs(acc, x + y) - repair.subs(acc, x) - repair.subs(acc, y)) != 0:
        raise ValueError("Could not prove compatibility with addition")
    allowed = {acc, *old.values(), *new.values()}
    if repair.free_symbols - allowed:
        raise ValueError("Repair still depends on individual payload values")
    remaining = _MAX_NODES

    class Quotient(sp.Function):
        # Keep a chosen division intact during serialization; ordinary SymPy
        # division would expand it back into multiplication by a reciprocal.
        nargs = 2

    def pair_reference_powers(value):
        powers = value.as_powers_dict()
        paired = []
        for name in parameters:
            before, after = old[symbols[name]], new[symbols[name]]
            exponent = powers.get(before, sp.S.Zero)
            if not exponent or not exponent.is_Rational or powers.get(after) != -exponent:
                continue
            del powers[before], powers[after]
            if exponent < 0:
                before, after, exponent = after, before, -exponent
            if exponent.is_Integer:
                # (old/new)**k avoids overflowing or underflowing old**k and
                # new**k separately when the reference magnitudes are similar.
                ratio = Quotient(before, after)
                paired.append(sp.Pow(ratio, exponent, evaluate=False) if exponent != 1 else ratio)
            else:
                # For half powers keep sqrt(old)/sqrt(new): old/new can
                # underflow even while the square-root ratio is representable.
                paired.append(Quotient(before**exponent, after**exponent))
        if not paired:
            return value
        return sp.Mul(*(base**power for base, power in powers.items()), *paired, evaluate=False)

    def encode(value, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > _MAX_DEPTH:
            raise ValueError("Derived correction exceeds the supported size")
        if value.is_Symbol:
            return {"op": "symbol", "name": str(value)}
        if value.is_Integer:
            return {"op": "constant", "value": int(value)}
        if value.is_Rational:
            return {
                "op": "div", "args": [encode(sp.Integer(value.p), depth + 1),
                                      encode(sp.Integer(value.q), depth + 1)]
            }
        if value.func == Quotient:
            return {"op": "div", "args": [encode(arg, depth + 1) for arg in value.args]}
        if value.func == sp.Mul:
            value = pair_reference_powers(value)
        if value.func in (sp.Add, sp.Mul):
            op = "add" if value.func == sp.Add else "mul"
            middle = len(value.args) // 2
            left = value.func(*value.args[:middle])
            right = value.func(*value.args[middle:])
            return {"op": op, "args": [encode(left, depth + 1), encode(right, depth + 1)]}
        if value.func == sp.Pow:
            base, exponent = value.args
            if base == 2:
                return {"op": "exp2", "args": [encode(exponent, depth + 1)]}
            if exponent == -1:
                return {"op": "div", "args": [encode(sp.Integer(1), depth + 1), encode(base, depth + 1)]}
            return {"op": "pow", "args": [encode(base, depth + 1), encode(exponent, depth + 1)]}
        if value.func in (sp.exp, sp.log):
            return {"op": "exp" if value.func == sp.exp else "log", "args": [encode(value.args[0], depth + 1)]}
        raise ValueError("Derived correction contains an unsupported operation")

    correction = encode(repair)
    if not include_domains:
        return correction
    if params - expr.free_symbols or any(old[param] not in repair.free_symbols or new[param] not in repair.free_symbols
                                         for param in params):
        # A canceled parameter can retain a singularity in the source, e.g.
        # c*r/r at final r=0. An identity repair cannot restore that behavior.
        raise ValueError("Parameter cancellation would discard source-domain behavior")

    domains = {name: {"kind": "finite", "fallback": 0} for name in parameters}
    strength = {"finite": 0, "nonzero": 1, "positive": 2}

    def require(value, kind):
        if not value.free_symbols & params:
            return
        if value.free_symbols - params:
            raise ValueError("Cannot guard a domain that also depends on payload values")
        if kind == "nonzero" and value.is_zero is False:
            return
        if kind == "positive" and value.is_positive is True:
            return
        if value in params:
            domain = domains[str(value)]
            if strength[kind] > strength[domain["kind"]]:
                domain.update(kind=kind, fallback=1)
            return
        if kind == "nonzero" and value.func == sp.Mul:
            for arg in value.args:
                require(arg, "nonzero")
            return
        if value.func == sp.Pow:
            base, exponent = value.args
            if kind == "nonzero" or exponent.is_even is True:
                require(base, "nonzero")
                return
        # Independent guards cannot repair coupled poles such as r0 - r1.
        # Accept only proved domains; do not silently replace all negative
        # exponential shifts, which would destroy stable max-shifted sums.
        raise ValueError("Domain needs an expression-level guard")

    for value, kind in requirements:
        require(value, kind)
    return correction, domains


def derive_correction(expression, parameters, reducer="add"):
    """Return a proved repair AST, or None when no supported proof is found.

    Input symbols must not use ``acc``, ``old_``, or ``new_`` prefixes. The
    result refers only to ``acc`` and ``old_<parameter>``/``new_<parameter>``.
    """
    try:
        return _derive_correction(expression, parameters, reducer)
    except (ValueError, TypeError, NotImplementedError):
        return None


def main():
    try:
        data = sys.stdin.read(65537)
        if len(data) > 65536:
            raise ValueError("Request exceeds the supported size")
        request = json.loads(data)
        if not isinstance(request, dict):
            raise ValueError("Expected a request object")
        correction, domains = _derive_correction(request["expression"], request["parameters"],
                                                 request.get("reducer", "add"), include_domains=True)
        response = {"ok": True, "correction": correction, "domains": domains}
    except (ValueError, TypeError, KeyError, NotImplementedError, ImportError) as error:
        response = {"ok": False, "reason": str(error)}
    print(json.dumps(response, allow_nan=False))


if __name__ == "__main__":
    main()
