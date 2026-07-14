#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(cd "$(dirname -- "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_EXE="${PYTHON_EXE:-python3}"
SYSTEM_NAME="$(uname)"

if [ "$SYSTEM_NAME" = "Darwin" ]; then
    BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/out/build/macos-clang-python}"
    NATIVE_NAME="libjxlpy_native.dylib"
else
    BUILD_DIR="${BUILD_DIR:-$PROJECT_DIR/out/build/linux-clang-python}"
    NATIVE_NAME="libjxlpy_native.so"
fi
JXLPY_NATIVE_LIB="${JXLPY_NATIVE_LIB:-$BUILD_DIR/$NATIVE_NAME}"
export BUILD_DIR JXLPY_NATIVE_LIB

echo "[jxlpy] Step 1: Building native library..."
if [ "$SYSTEM_NAME" = "Darwin" ]; then
    "$SCRIPT_DIR/build_macos.sh" jxlpy_native
else
    "$SCRIPT_DIR/build_linux.sh" jxlpy_native
fi

echo "[jxlpy] Step 2: Building wheel..."
cd "$PROJECT_DIR"
"$PYTHON_EXE" -m pip wheel . --no-deps --wheel-dir dist

echo "[jxlpy] Step 3: Verifying wheel..."
for whl in dist/jxlpy-*.whl; do
    echo "[jxlpy] Built: $whl"
    if "$PYTHON_EXE" -m zipfile -l "$whl" | grep -qi "jxlpy_native"; then
        echo "[jxlpy] OK: native library included in wheel"
    else
        echo "[jxlpy] ERROR: native library not found in wheel"
        exit 1
    fi
done

echo "[jxlpy] Done. Output: dist/"
