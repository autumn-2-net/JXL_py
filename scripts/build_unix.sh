#!/usr/bin/env sh
set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
TARGET="${1:-${TARGET:-jxlpy_native}}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS:-OFF}"
CMAKE_EXE="${CMAKE_EXE:-cmake}"
CMAKE_GENERATOR="${CMAKE_GENERATOR:-Ninja}"

SYSTEM_NAME="$(uname -s)"
case "$SYSTEM_NAME" in
  Darwin)
    DEFAULT_BUILD_DIR="$ROOT/out/build/macos-clang-python"
    DEFAULT_C_COMPILER="${CMAKE_C_COMPILER:-clang}"
    DEFAULT_CXX_COMPILER="${CMAKE_CXX_COMPILER:-clang++}"
    ;;
  Linux)
    DEFAULT_BUILD_DIR="$ROOT/out/build/linux-clang-python"
    DEFAULT_C_COMPILER="${CMAKE_C_COMPILER:-clang}"
    DEFAULT_CXX_COMPILER="${CMAKE_CXX_COMPILER:-clang++}"
    ;;
  *)
    DEFAULT_BUILD_DIR="$ROOT/out/build/unix-python"
    DEFAULT_C_COMPILER="${CMAKE_C_COMPILER:-cc}"
    DEFAULT_CXX_COMPILER="${CMAKE_CXX_COMPILER:-c++}"
    ;;
esac

BUILD_DIR="${BUILD_DIR:-$DEFAULT_BUILD_DIR}"

echo "[jxlpy] root: $ROOT"
echo "[jxlpy] system: $SYSTEM_NAME"
echo "[jxlpy] target: $TARGET"
echo "[jxlpy] build dir: $BUILD_DIR"

"$CMAKE_EXE" \
  -S "$ROOT" \
  -B "$BUILD_DIR" \
  -G "$CMAKE_GENERATOR" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_C_COMPILER="$DEFAULT_C_COMPILER" \
  -DCMAKE_CXX_COMPILER="$DEFAULT_CXX_COMPILER" \
  -DBUILD_SHARED_LIBS="$BUILD_SHARED_LIBS" \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5

"$CMAKE_EXE" --build "$BUILD_DIR" --target "$TARGET" --config "$BUILD_TYPE"

echo "[jxlpy] done"
