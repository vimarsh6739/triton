"""Experimental Enzyme-based automatic differentiation for Triton kernels.

This module is intentionally under ``triton.experimental``. Import paths,
argument conventions, generated IR, and runtime behavior may change without
the stability guarantees of the top-level Triton Python API.
"""

from __future__ import annotations

import functools
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, cast

from .. import knobs

__all__ = ["Const", "Duplicated", "autodiff", "fwddiff"]


_FORWARD_MODE = "ForwardMode"
_SUPPORTED_MODES = {"forward": _FORWARD_MODE, "fwd": _FORWARD_MODE, _FORWARD_MODE: _FORWARD_MODE}
_ENZYME_OPT_ENV = "TRITON_ENZYME_OPT"
_ENZYME_OPT_FALLBACKS = (
    "/home/vimarsh6739/hiord/Enzyme-JAX/bazel-bin/enzymexlamlir-opt",
    "/mnt/vimarsh6739/hiord/Enzyme-JAX/bazel-bin/enzymexlamlir-opt",
)


@dataclass(frozen=True)
class Duplicated:
    val: Any
    dval: Any


@dataclass(frozen=True)
class Const:
    val: Any


@dataclass(frozen=True)
class _DiffSpec:
    mode: str
    arg_activities: tuple[str, ...]
    enzyme_opt: str
    tensor_shape: tuple[int, ...] | None
    keep_temps: bool

    @property
    def key_material(self) -> str:
        return repr((self.mode, self.arg_activities, self.enzyme_opt, self.tensor_shape))


def _normalize_mode(mode: str) -> str:
    try:
        return _SUPPORTED_MODES[mode]
    except KeyError as exc:
        supported = ", ".join(sorted(_SUPPORTED_MODES))
        raise ValueError(f"Unsupported Triton autodiff mode {mode!r}; supported modes: {supported}") from exc


def _find_enzyme_opt(enzyme_opt: str | None = None) -> str:
    candidates = []
    if enzyme_opt is not None:
        candidates.append(enzyme_opt)
    if env_path := os.environ.get(_ENZYME_OPT_ENV):
        candidates.append(env_path)
    if path_opt := shutil.which("enzymexlamlir-opt"):
        candidates.append(path_opt)
    candidates.extend(_ENZYME_OPT_FALLBACKS)

    for candidate in candidates:
        if candidate and os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise RuntimeError(
        "Unable to find enzymexlamlir-opt. Set TRITON_ENZYME_OPT to the Enzyme-JAX optimizer binary.")


def _is_pointer_signature_type(ty: str) -> bool:
    return ty.startswith("*")


def _pointer_element_type(ty: str) -> str:
    if not _is_pointer_signature_type(ty):
        raise TypeError(f"Expected a pointer signature type, got {ty!r}")
    return ty[1:]


def _tensor_shape_text(shape: Sequence[int]) -> str:
    return "".join(f"{dim}x" for dim in shape)


def _infer_tensor_shape(ttir: str, element_ty: str, explicit_shape: tuple[int, ...] | None) -> tuple[int, ...]:
    if explicit_shape is not None:
        return explicit_shape

    pattern = re.compile(rf"tensor<([0-9]+(?:x[0-9]+)*)x{re.escape(element_ty)}>")
    if match := pattern.search(ttir):
        return tuple(int(dim) for dim in match.group(1).split("x"))

    raise RuntimeError(
        f"Unable to infer tensor shape for pointer element type {element_ty!r}; pass tensor_shape=... to fwddiff().")


def _stablehlo_type(signature_ty: str, ttir: str, tensor_shape: tuple[int, ...] | None) -> str:
    if _is_pointer_signature_type(signature_ty):
        element_ty = _pointer_element_type(signature_ty)
        shape = _infer_tensor_shape(ttir, element_ty, tensor_shape)
        return f"tensor<{_tensor_shape_text(shape)}{element_ty}>"
    return f"tensor<{signature_ty}>"


def _module_body(ttir: str) -> str:
    module_match = re.search(r"\bmodule\b[^{]*\{", ttir)
    if module_match is None:
        raise RuntimeError("Expected TTIR to contain a top-level module")

    open_brace = ttir.find("{", module_match.start())
    depth = 0
    for idx in range(open_brace, len(ttir)):
        char = ttir[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return ttir[open_brace + 1:idx]
    raise RuntimeError("Could not find the end of the top-level TTIR module")


def _build_enzyme_module(ttir: str, entry_name: str, signature: Sequence[str], spec: _DiffSpec) -> tuple[str, str]:
    stablehlo_types = [_stablehlo_type(ty, ttir, spec.tensor_shape) for ty in signature]
    pointer_arg_indices = [idx for idx, ty in enumerate(signature) if _is_pointer_signature_type(ty)]
    if not pointer_arg_indices:
        raise RuntimeError("fwddiff expected at least one pointer argument to model Triton memory results")

    ret_activities = [spec.arg_activities[idx] for idx in pointer_arg_indices]
    arg_activities = ",".join(spec.arg_activities)
    ret_activity_text = ",".join(ret_activities)
    pass_arg = f"infn=main outfn= argTys={arg_activities} retTys={ret_activity_text} mode={spec.mode}"

    args = ", ".join(f"%arg{idx}: {ty}" for idx, ty in enumerate(stablehlo_types))
    operands = ", ".join(f"%arg{idx}" for idx in range(len(signature)))
    result_types = [stablehlo_types[idx] for idx in pointer_arg_indices]
    result_type_text = ", ".join(result_types)
    result_bind = f"%0:{len(result_types)}" if len(result_types) != 1 else "%0"
    result_refs = ", ".join(f"%0#{idx}" for idx in range(len(result_types)))
    if len(result_types) == 1:
        result_refs = "%0"

    aliases = ", ".join(
        "#stablehlo.output_operand_alias<"
        f"output_tuple_indices = [{result_idx}], "
        f"operand_index = {operand_idx}, operand_tuple_indices = []>"
        for result_idx, operand_idx in enumerate(pointer_arg_indices))

    body = _module_body(ttir)
    wrapper = f"""module {{
  enzymexla_tt_ext.module @{entry_name}_tt {{
    builtin.module @{entry_name}_inner {{
{body}
    }}
  }}
  func.func @main({args}) -> ({result_type_text}) {{
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    {result_bind} = enzymexla_tt_ext.call @{entry_name}_tt::@{entry_name}_inner::@{entry_name} clusters in(%c1, %c1, %c1) blocks in(%c1, %c1, %c1) ({operands}) {{output_operand_aliases = [{aliases}]}} : ({", ".join(stablehlo_types)}) -> ({result_type_text})
    return {result_refs} : {result_type_text}
  }}
}}
"""
    return wrapper, pass_arg


def _find_tt_func(text: str, func_name: str) -> str:
    pattern = re.compile(rf"\btt\.func(?:\s+(?:public|private))?\s+@{re.escape(func_name)}\b")
    match = pattern.search(text)
    if match is None:
        raise RuntimeError(f"Enzyme output did not contain expected differentiated function @{func_name}")

    line_end = text.find("\n", match.start())
    if line_end == -1:
        raise RuntimeError(f"Could not parse differentiated function @{func_name}")
    body_open = text.rfind("{", match.start(), line_end)
    if body_open == -1:
        raise RuntimeError(f"Could not find body for differentiated function @{func_name}")

    depth = 0
    for idx in range(body_open, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[match.start():idx + 1]
    raise RuntimeError(f"Could not find end of differentiated function @{func_name}")


def _make_public(func_text: str) -> str:
    first_line_end = func_text.find("\n")
    if first_line_end == -1:
        first_line_end = len(func_text)
    first_line = func_text[:first_line_end]
    rest = func_text[first_line_end:]
    first_line = re.sub(r"\btt\.func\s+private\s+@", "tt.func public @", first_line, count=1)
    first_line = re.sub(r"\btt\.func\s+@", "tt.func public @", first_line, count=1)
    return first_line + rest


def _derivative_func_name(mode: str, entry_name: str) -> str:
    if mode == _FORWARD_MODE:
        return f"fwddiffe{entry_name}"
    raise ValueError(f"Unsupported Triton autodiff mode {mode!r}")


def _write_temp_mlir(text: str, keep: bool, prefix: str) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".mlir", prefix=prefix, delete=False)
    try:
        handle.write(text)
        return handle.name
    finally:
        handle.close()
        if not keep:
            # The caller may still need the path briefly, so unlinking is handled after use.
            pass


def _differentiate_ttir_module(mod, spec: _DiffSpec):
    from triton._C.libtriton import ir  # pyright: ignore[reportAttributeAccessIssue]

    ttir = mod.str_nodebug()
    entry_name = mod.get_entry_func_name()
    signature = mod.get_function_signature(mod.get_function(entry_name))
    wrapper, pass_arg = _build_enzyme_module(ttir, entry_name, signature, spec)
    wrapper_path = _write_temp_mlir(wrapper, spec.keep_temps, "triton-enzyme-in-")
    out_path = None
    try:
        cmd = [spec.enzyme_opt, wrapper_path, f"--enzyme-wrap={pass_arg}", "--canonicalize"]
        result = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(
                "enzymexlamlir-opt failed while differentiating Triton TTIR:\n"
                f"command: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")

        diff_name = _derivative_func_name(spec.mode, entry_name)
        diff_func = _make_public(_find_tt_func(result.stdout, diff_name))
        diff_module = f"module {{\n{diff_func}\n}}\n"
        out_path = _write_temp_mlir(diff_module, spec.keep_temps, "triton-enzyme-out-")
        parsed = ir.parse_mlir_module(out_path, mod.context)
        parsed.context = mod.context
        if not parsed.verify():
            raise RuntimeError("fwddiff produced invalid TTIR")
        return parsed
    finally:
        if not spec.keep_temps:
            for path in (wrapper_path, out_path):
                if path:
                    try:
                        os.unlink(path)
                    except FileNotFoundError:
                        pass


def _pipeline_cache_key(spec: _DiffSpec) -> tuple[str, str]:
    source = Path(__file__).read_text()
    key = f"triton-autodiff:{spec.key_material}:{source}"
    return key, hashlib.sha256(key.encode("utf-8")).hexdigest()


def _compose_pipeline_hook(previous_hook: knobs.PipelineStagesHook | None,
                           spec: _DiffSpec) -> knobs.PipelineStagesHook:

    def autodiff_stages_hook(
        self: object | None = None,
        stages: knobs.PipelineStages | None = None,
        options: Any = None,
        language: Any = None,
        capability: Any = None,
    ) -> tuple[str, str] | None:
        key, digest = _pipeline_cache_key(spec)
        if stages is None:
            if not all(arg is None for arg in (options, language, capability)):
                raise TypeError("Triton autodiff stage hook expected stages when pipeline metadata is provided")
            if previous_hook is None:
                return key, digest
            previous_key, previous_digest = previous_hook()
            combined_key = previous_key + key
            combined_digest = hashlib.sha256((previous_digest + digest).encode("utf-8")).hexdigest()
            return combined_key, combined_digest

        if previous_hook is not None:
            previous_hook(self, stages, options, language, capability)

        original_ttir = stages.get("ttir")
        if original_ttir is None:
            return key, digest

        def custom_ttir(src: Any, metadata: dict[str, Any]) -> Any:
            mod = original_ttir(src, metadata)
            return _differentiate_ttir_module(mod, spec)

        stages["ttir"] = custom_ttir
        return key, digest

    return cast(knobs.PipelineStagesHook, autodiff_stages_hook)


@contextmanager
def _autodiff_pipeline(spec: _DiffSpec):
    previous_hook = knobs.runtime.add_stages_inspection_hook
    knobs.runtime.add_stages_inspection_hook = _compose_pipeline_hook(previous_hook, spec)
    try:
        yield
    finally:
        knobs.runtime.add_stages_inspection_hook = previous_hook


def _flatten_arg(arg: Any, configured_activity: str | None) -> tuple[Any, tuple[Any, ...], str]:
    if isinstance(arg, Duplicated):
        return arg.val, (arg.val, arg.dval), "enzyme_dup"
    if isinstance(arg, Const):
        return arg.val, (arg.val, ), "enzyme_const"

    if configured_activity == "enzyme_const":
        return arg, (arg, ), "enzyme_const"
    if configured_activity == "enzyme_dup":
        raise TypeError(
            "Active Triton autodiff arguments must be wrapped as "
            "triton.experimental.Duplicated(primal, tangent)")

    if hasattr(arg, "data_ptr") and hasattr(arg, "dtype"):
        raise TypeError(
            "Tensor arguments to triton.experimental.fwddiff must be wrapped as "
            "triton.experimental.Duplicated(primal, tangent)")
    return arg, (arg, ), "enzyme_const"


def _normalize_activity(activity: str | None) -> str | None:
    if activity is None:
        return None
    aliases = {
        "dup": "enzyme_dup",
        "duplicated": "enzyme_dup",
        "active": "enzyme_dup",
        "enzyme_dup": "enzyme_dup",
        "const": "enzyme_const",
        "constant": "enzyme_const",
        "enzyme_const": "enzyme_const",
    }
    try:
        return aliases[activity]
    except KeyError as exc:
        raise ValueError(f"Unknown Triton autodiff activity {activity!r}") from exc


def _normalize_activities(arg_activities: Iterable[str | None] | None) -> tuple[str | None, ...] | None:
    if arg_activities is None:
        return None
    return tuple(_normalize_activity(activity) for activity in arg_activities)


class _AutodiffKernel:

    def __init__(self, kernel, *, mode: str, arg_activities: Iterable[str | None] | None, enzyme_opt: str | None,
                 tensor_shape: int | Sequence[int] | None, keep_temps: bool):
        self._kernel = kernel
        self._mode = _normalize_mode(mode)
        self._configured_arg_activities = _normalize_activities(arg_activities)
        self._enzyme_opt = enzyme_opt
        if isinstance(tensor_shape, int):
            self._tensor_shape = (tensor_shape, )
        elif tensor_shape is None:
            self._tensor_shape = None
        else:
            self._tensor_shape = tuple(tensor_shape)
        self._keep_temps = keep_temps
        functools.update_wrapper(self, kernel, updated=())  # pyright: ignore[reportArgumentType]

    @property
    def primal(self):
        return self._kernel

    def __getattr__(self, name):
        return getattr(self._kernel, name)

    def _flatten_args(self, args: tuple[Any, ...]) -> tuple[list[Any], list[Any], tuple[str, ...]]:
        if self._configured_arg_activities is not None and len(self._configured_arg_activities) != len(args):
            raise TypeError(
                f"Expected {len(self._configured_arg_activities)} runtime arguments for configured autodiff "
                f"activities, got {len(args)}")

        compile_args = []
        launch_args = []
        activities = []
        for idx, arg in enumerate(args):
            configured = None if self._configured_arg_activities is None else self._configured_arg_activities[idx]
            compile_arg, flat_args, activity = _flatten_arg(arg, configured)
            compile_args.append(compile_arg)
            launch_args.extend(flat_args)
            activities.append(activity)
        return compile_args, launch_args, tuple(activities)

    @staticmethod
    def _patch_launcher_signature(kernel, activities: tuple[str, ...]) -> None:
        primal_signature = getattr(kernel.src, "_triton_autodiff_primal_signature", None)
        if primal_signature is None:
            primal_signature = dict(kernel.src.signature)
            kernel.src._triton_autodiff_primal_signature = primal_signature

        expanded = []
        for ty, activity in zip(primal_signature.values(), activities):
            expanded.append(ty)
            if activity == "enzyme_dup":
                expanded.append(ty)
        kernel.src.signature = {idx: ty for idx, ty in enumerate(expanded)}

    def run(self, *args, grid, warmup, **kwargs):
        from ..runtime.driver import driver

        compile_args, launch_args, activities = self._flatten_args(args)
        spec = _DiffSpec(
            mode=self._mode,
            arg_activities=activities,
            enzyme_opt=_find_enzyme_opt(self._enzyme_opt),
            tensor_shape=self._tensor_shape,
            keep_temps=self._keep_temps,
        )

        with _autodiff_pipeline(spec):
            kernel = self._kernel.run(*compile_args, grid=grid, warmup=True, **kwargs)
        self._patch_launcher_signature(kernel, activities)

        if warmup:
            return kernel

        bound_args = {name: value for name, value in zip(self._kernel.arg_names, compile_args)}
        bound_args.update(kwargs)
        if callable(grid):
            grid = grid(bound_args)
        grid = cast(Sequence[int], grid)
        grid_size = len(grid)
        grid_0 = grid[0]
        grid_1 = grid[1] if grid_size > 1 else 1
        grid_2 = grid[2] if grid_size > 2 else 1

        active_driver = cast(Any, driver.active)
        device = active_driver.get_current_device()
        stream = active_driver.get_current_stream(device)
        launch_metadata = kernel.launch_metadata(grid, stream, *compile_args)
        kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                   knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *launch_args)
        return kernel

    def warmup(self, *args, grid, **kwargs):
        return self.run(*args, grid=grid, warmup=True, **kwargs)

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(*args, grid=grid, warmup=False, **kwargs)


def autodiff(kernel=None, *, mode: str = "forward", arg_activities: Iterable[str | None] | None = None,
             enzyme_opt: str | None = None, tensor_shape: int | Sequence[int] | None = None,
             keep_temps: bool = False):
    def decorator(fn):
        return _AutodiffKernel(
            fn,
            mode=mode,
            arg_activities=arg_activities,
            enzyme_opt=enzyme_opt,
            tensor_shape=tensor_shape,
            keep_temps=keep_temps,
        )

    if kernel is None:
        return decorator
    return decorator(kernel)


def fwddiff(kernel=None, **kwargs):
    return autodiff(kernel, mode="forward", **kwargs)

