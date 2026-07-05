@echo off
setlocal

if "%PYTHON_EXE%"=="" set "PYTHON_EXE=C:\Users\autumn\.conda\envs\py10\python.exe"

"%PYTHON_EXE%" "%~dp0smoke_jxlpy.py"
