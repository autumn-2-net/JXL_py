# jxl_python

This repository is a local superbuild around `libjxl`. The current milestone is
to build and validate the JPEG XL command line tools, native libraries, and a
first Python/CFFI wrapper.

## Current Status

Working on Windows with:

- `clang-cl` 17.0.6
- CLion bundled CMake 4.2.2
- CLion bundled Ninja 1.13.2
- vendored `libjxl/` source tree with populated `third_party/` dependencies

Built and tested CLI targets:

- `cjxl`
- `djxl`
- `jxlinfo`
- `jxltran`

The first Python wrapper target is also built and smoke-tested:

- `jxlpy_native`
- `jxlpy.encode`
- `jxlpy.decode`
- `jxlpy.encode_multiframe`
- `jxlpy.decode_layer`

The tested Windows CLI build output is:

```text
out/build/windows-clang-cl-cli/libjxl/tools/
```

## Repository Layout

```text
.
+-- CMakeLists.txt          # Thin superbuild wrapper around libjxl
+-- CMakePresets.json      # Windows/Linux/macOS configure and build presets
+-- docs/                  # Project notes and Python wrapper usage
+-- jxlpy/                 # Python CFFI package
+-- libjxl/                # Vendored libjxl source tree
+-- native/                # C ABI shim loaded by CFFI
+-- scripts/               # Build, smoke test, and benchmark helpers
+-- test_img/              # Local image samples used for validation
`-- out/                   # Generated build and test outputs, ignored by Git
```

`libjxl/` is the upstream source tree with submodules already expanded. It is
not a nested Git repository in this workspace.

## Build Design

The root CMake project does not rewrite libjxl. It sets practical defaults,
prepares a few bundled dependencies, then delegates to upstream CMake with:

```cmake
add_subdirectory(libjxl)
```

This keeps the libjxl target graph, symbol exports, Highway dispatch, public
headers, and ABI behavior as close to upstream as possible.

Enabled bundled dependencies:

- Brotli
- Highway
- skcms
- lcms
- libpng
- zlib
- libjpeg-turbo
- sjpeg

Currently not bundled:

- GIF
- OpenEXR
- WebP
- AVIF

Those are still discovered the upstream way through system packages. For a full
format build on Windows, the next clean step is likely vcpkg or additional
vendoring.

## Important Build Notes

`CMAKE_POLICY_VERSION_MINIMUM=3.5` is set because CMake 4.x removed old policy
compatibility needed by bundled projects such as `libjpeg-turbo`.

`libjpeg-turbo` SIMD is disabled for now:

```cmake
WITH_SIMD=OFF
```

This only affects `libjpeg-turbo`'s own JPEG assembly acceleration. It does not
disable libjxl's Highway SIMD. The tested encoder/decoder reported:

```text
[_AVX2_,SSE4,SSE2]
```

Bundled zlib normally renames `third_party/zlib/zconf.h` during out-of-source
CMake configuration. The local patch builds zlib from a copied build-directory
source tree to avoid modifying vendored source files.

## Windows Build

For the Python native shim, use the helper script:

```bat
.\scripts\build_windows.cmd jxlpy_native
```

Then run the smoke test with the known conda environment:

```bat
.\scripts\smoke_jxlpy.cmd
```

Detailed Python wrapper notes are in:

```text
docs/JXLPY_USAGE.md
```

Linux/macOS helper scripts are also present:

```bash
./scripts/build_linux.sh jxlpy_native
./scripts/build_macos.sh jxlpy_native
```

Wheel helper scripts:

```bash
./scripts/build_wheel.sh
```

```bat
.\scripts\build_wheel.cmd
```

If CMake is not on `PATH`, use CLion's bundled CMake explicitly:

```powershell
& 'C:\Program Files\JetBrains\CLion 2025.2.4\bin\cmake\win\x64\bin\cmake.exe' --preset windows-clang-cl-cli
& 'C:\Program Files\JetBrains\CLion 2025.2.4\bin\cmake\win\x64\bin\cmake.exe' --build --preset windows-cli
```

The Windows preset currently hardcodes CLion's bundled Ninja path:

```text
C:/Program Files/JetBrains/CLion 2025.2.4/bin/ninja/win/x64/ninja.exe
```

If CLion is installed somewhere else, update `CMakePresets.json` or pass
`CMAKE_MAKE_PROGRAM` manually.

Equivalent explicit configure command:

```powershell
& 'C:\Program Files\JetBrains\CLion 2025.2.4\bin\cmake\win\x64\bin\cmake.exe' `
  -S . `
  -B out\build\windows-clang-cl-cli `
  -G Ninja `
  '-DCMAKE_MAKE_PROGRAM:FILEPATH=C:/Program Files/JetBrains/CLion 2025.2.4/bin/ninja/win/x64/ninja.exe' `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_C_COMPILER=clang-cl `
  -DCMAKE_CXX_COMPILER=clang-cl `
  -DBUILD_SHARED_LIBS=OFF
```

## Library Build

The shared-library preset is available:

```powershell
& 'C:\Program Files\JetBrains\CLion 2025.2.4\bin\cmake\win\x64\bin\cmake.exe' --preset windows-clang-cl-shared
& 'C:\Program Files\JetBrains\CLion 2025.2.4\bin\cmake\win\x64\bin\cmake.exe' --build --preset windows-libraries
```

The custom aggregate target is:

```text
jxl_libraries
```

It depends on available core library targets:

- `jxl`
- `jxl_dec`
- `jxl_threads`
- `jxl_cms`

## Linux And macOS Presets

The presets are present but not yet validated in this workspace.

Linux clang:

```bash
cmake --preset linux-clang-cli
cmake --build --preset linux-clang-cli
```

Linux GCC:

```bash
cmake --preset linux-gcc-cli
cmake --build --preset linux-gcc-cli
```

macOS clang:

```bash
cmake --preset macos-clang-cli
cmake --build --preset macos-cli
```

Shared-library presets:

```bash
cmake --preset linux-clang-shared
cmake --build --preset linux-clang-libraries

cmake --preset macos-clang-shared
cmake --build --preset macos-libraries
```

## Validation Commands

JPEG lossless transcode:

```powershell
out\build\windows-clang-cl-cli\libjxl\tools\cjxl.exe `
  test_img\wallhaven-vpyekp.jpg `
  out\test-run\wallhaven-vpyekp.jxl `
  --effort=1

out\build\windows-clang-cl-cli\libjxl\tools\djxl.exe `
  out\test-run\wallhaven-vpyekp.jxl `
  out\test-run\wallhaven-vpyekp-roundtrip.jpg

Get-FileHash test_img\wallhaven-vpyekp.jpg,out\test-run\wallhaven-vpyekp-roundtrip.jpg -Algorithm SHA256
```

The round-tripped JPEG hash matched the original.

PNG encode/decode:

```powershell
out\build\windows-clang-cl-cli\libjxl\tools\cjxl.exe `
  test_img\wallhaven-mlzoy1.png `
  out\test-run\wallhaven-mlzoy1.jxl `
  --effort=1 `
  --distance=1

out\build\windows-clang-cl-cli\libjxl\tools\djxl.exe `
  out\test-run\wallhaven-mlzoy1.jxl `
  out\test-run\wallhaven-mlzoy1-roundtrip.png
```

Inspect a JXL file:

```powershell
out\build\windows-clang-cl-cli\libjxl\tools\jxlinfo.exe out\test-run\wallhaven-vpyekp.jxl
```

## Size Test Results

JPEG lossless transcode and PNG tests were run on local samples in `test_img/`.

JPEG settings:

```text
--lossless_jpeg=1 --effort=7
```

PNG pixel-lossless settings:

```text
--distance=0 --effort=10
```

Lossy PNG settings:

```text
--distance=1 --effort=10
```

Summary:

| Input | Files | Original | JXL | Reduction |
|---|---:|---:|---:|---:|
| JPEG lossless transcode | 10 | 27.04 MB | 22.83 MB | 15.59% |
| PNG lossless, effort 10 | 5 | 28.12 MB | 19.48 MB | 30.73% |
| PNG lossy d1/e10 | 5 | 28.12 MB | 5.48 MB | 80.51% |

Notes:

- JPEG lossless transcode can reconstruct the original JPEG bitstream.
- PNG `distance=0` is pixel-lossless, not necessarily byte-identical to the original PNG file.
- PNG `distance=1` is visually high quality but lossy.
- `effort=10` is slow; the 5 PNG lossless test set took several minutes.

Generated test outputs:

```text
out/test-run/
out/size-test/
out/png-max-test/
```

## JPEG XL Feature Notes

The local build keeps core libjxl features enabled, including:

- JPEG reconstruction support
- container boxes
- normal JXL encoder/decoder API targets
- libjxl Highway SIMD dispatch

The local changes do not alter libjxl codec source files. Multi-frame, extra
channel, and container-related functionality should remain available. They have
not yet been validated with dedicated APNG/multi-frame/extra-channel samples.

## Next Steps

Planned next phase:

1. Validate shared library build on Windows.
2. Decide how to handle full optional format support: vcpkg vs extra vendored dependencies.
3. Design a Python package boundary.
4. Use `cffi` to load the built libjxl libraries.
5. Add Python-level encode/decode tests after the C ABI boundary is agreed.
