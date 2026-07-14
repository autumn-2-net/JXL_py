@echo off
setlocal

if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"
set "PROJECT_DIR=%~dp0.."
for %%I in ("%PROJECT_DIR%") do set "PROJECT_DIR=%%~fI"
if "%BUILD_DIR%"=="" set "BUILD_DIR=%PROJECT_DIR%\out\build\windows-clang-cl-cli"
if "%JXLPY_NATIVE_LIB%"=="" set "JXLPY_NATIVE_LIB=%BUILD_DIR%\jxlpy_native.dll"

echo [jxlpy] Step 1: Building native library...
call "%~dp0build_windows.cmd" jxlpy_native
if errorlevel 1 (
    echo [jxlpy] ERROR: native build failed
    exit /b %errorlevel%
)

echo [jxlpy] Step 2: Building wheel...
"%PYTHON_EXE%" -m pip wheel . --no-deps --wheel-dir dist
if errorlevel 1 (
    echo [jxlpy] ERROR: wheel build failed
    exit /b %errorlevel%
)

echo [jxlpy] Step 3: Verifying wheel contents...
for %%f in (dist\jxlpy-*.whl) do (
    echo [jxlpy] Built: %%f
    "%PYTHON_EXE%" -m zipfile -l "%%f" | findstr /i "jxlpy_native"
    if errorlevel 1 (
        echo [jxlpy] ERROR: native library not found in wheel
        exit /b 1
    ) else (
        echo [jxlpy] OK: native library included in wheel
    )
)

echo [jxlpy] Done. Output: dist\
