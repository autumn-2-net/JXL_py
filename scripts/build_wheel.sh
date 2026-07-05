#!/usr/bin/env sh
set -eu

PYTHON_EXE="${PYTHON_EXE:-python3}"

"$(dirname -- "$0")/build_unix.sh" jxlpy_native
"$PYTHON_EXE" setup.py bdist_wheel

echo "[jxlpy] wheel output: dist"
