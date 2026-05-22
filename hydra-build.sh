#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${CONDA_DEFAULT_ENV:-}" != "triton-dev" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then
    eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
  else
    echo "Could not find conda; activate triton-dev manually." >&2
    exit 1
  fi
  conda activate triton-dev
fi

: "${CUDA_HOME:?CUDA_HOME must be set before running buildp.sh}"

export TRITON_PTXAS_BLACKWELL_PATH="${CUDA_HOME}/bin/ptxas"
export TRITON_BUILD_WITH_CLANG_LLD=1
export TRITON_BUILD_WITH_CCACHE=1

# Only set these when you actually want Triton to use a custom LLVM build.
# Example:
#   LLVM_BUILD_DIR=$HOME/llvm-project/build ./buildp.sh
if [[ -n "${LLVM_BUILD_DIR:-}" ]]; then
  export LLVM_INCLUDE_DIRS="${LLVM_BUILD_DIR}/include"
  export LLVM_LIBRARY_DIR="${LLVM_BUILD_DIR}/lib"
  export LLVM_SYSPATH="${LLVM_BUILD_DIR}"
fi

cd "$ROOT_DIR"
python -m pip install -e . --no-build-isolation -v
