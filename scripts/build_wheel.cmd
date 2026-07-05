@echo off
setlocal

if "%PYTHON_EXE%"=="" set "PYTHON_EXE=C:\Users\autumn\.conda\envs\py10\python.exe"

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
        echo [jxlpy] WARNING: native library not found in wheel
    ) else (
        echo [jxlpy] OK: native library included in wheel
    )
)

echo [jxlpy] Done. Output: dist\

