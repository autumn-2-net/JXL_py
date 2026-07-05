@echo off
setlocal

if "%PYTHON_EXE%"=="" set "PYTHON_EXE=python"

"%PYTHON_EXE%" "%~dp0smoke_jxlpy.py"
