from __future__ import annotations

import warnings

from .experimental.autodiff import Const, Duplicated, _find_enzyme_opt, autodiff, fwddiff

warnings.warn(
    "triton.autodiff is experimental and has moved to triton.experimental.autodiff.",
    FutureWarning,
    stacklevel=2,
)

__all__ = ["Const", "Duplicated", "autodiff", "fwddiff"]
