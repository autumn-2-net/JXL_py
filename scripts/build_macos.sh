#!/usr/bin/env sh
set -eu

export CMAKE_C_COMPILER="${CMAKE_C_COMPILER:-clang}"
export CMAKE_CXX_COMPILER="${CMAKE_CXX_COMPILER:-clang++}"
exec "$(dirname -- "$0")/build_unix.sh" "${1:-${TARGET:-jxlpy_native}}"
