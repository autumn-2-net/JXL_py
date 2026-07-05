@echo off
setlocal

if "%PYTHON_EXE%"=="" set "PYTHON_EXE=C:\Users\autumn\.conda\envs\py10\python.exe"

call "%~dp0build_windows.cmd" jxlpy_native
if errorlevel 1 exit /b %errorlevel%

"%PYTHON_EXE%" setup.py bdist_wheel
if errorlevel 1 exit /b %errorlevel%

echo [jxlpy] wheel output: dist
