@echo off
setlocal

set "ROOT=%~dp0.."
for %%I in ("%ROOT%") do set "ROOT=%%~fI"

if "%CMAKE_EXE%"=="" set "CMAKE_EXE=C:\Program Files\JetBrains\CLion 2025.2.4\bin\cmake\win\x64\bin\cmake.exe"
if "%NINJA_EXE%"=="" set "NINJA_EXE=C:\Program Files\JetBrains\CLion 2025.2.4\bin\ninja\win\x64\ninja.exe"
if "%BUILD_DIR%"=="" set "BUILD_DIR=%ROOT%\out\build\windows-clang-cl-cli"
if "%BUILD_TYPE%"=="" set "BUILD_TYPE=Release"
if "%TARGET%"=="" set "TARGET=jxlpy_native"
if not "%~1"=="" set "TARGET=%~1"

echo [jxlpy] root: %ROOT%
echo [jxlpy] target: %TARGET%
echo [jxlpy] build dir: %BUILD_DIR%

"%CMAKE_EXE%" ^
  -S "%ROOT%" ^
  -B "%BUILD_DIR%" ^
  -G Ninja ^
  "-DCMAKE_MAKE_PROGRAM:FILEPATH=%NINJA_EXE%" ^
  -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
  -DCMAKE_C_COMPILER=clang-cl ^
  -DCMAKE_CXX_COMPILER=clang-cl ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
if errorlevel 1 exit /b %errorlevel%

"%CMAKE_EXE%" --build "%BUILD_DIR%" --target "%TARGET%" --config %BUILD_TYPE%
if errorlevel 1 exit /b %errorlevel%

echo [jxlpy] done
