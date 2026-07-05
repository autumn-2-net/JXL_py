from __future__ import annotations

import os
import sys
from pathlib import Path

from cffi import FFI

ffi = FFI()

ffi.cdef(
    """
    enum {
      JXLPY_DTYPE_UINT8 = 1,
      JXLPY_DTYPE_UINT16 = 2,
      JXLPY_DTYPE_FLOAT16 = 3,
      JXLPY_DTYPE_FLOAT32 = 4,
    };

    typedef struct {
      int lossless;
      float distance;
      float alpha_distance;
      int effort;
      int modular;
      int level;
      int threads;
      int use_container;
      int jpeg_store_metadata;
      uint32_t tps_numerator;
      uint32_t tps_denominator;
    } jxlpy_encode_options;

    typedef struct {
      const void* pixels;
      size_t size;
      uint32_t xsize;
      uint32_t ysize;
      uint32_t dtype;
      uint32_t bits_per_sample;
      uint32_t type;
      const char* name;
      size_t name_size;
    } jxlpy_extra_channel;

    typedef struct {
      const void* pixels;
      size_t size;
      uint32_t xsize;
      uint32_t ysize;
      uint32_t have_crop;
      int32_t crop_x0;
      int32_t crop_y0;
      uint32_t duration;
      uint32_t source_ref;
      uint32_t save_as_ref;
    } jxlpy_frame;

    typedef struct {
      int ok;
      char* error;
      uint8_t* data;
      size_t size;
      uint32_t xsize;
      uint32_t ysize;
      uint32_t num_channels;
      uint32_t dtype;
      uint32_t bits_per_sample;
      uint32_t exponent_bits_per_sample;
      uint32_t num_frames;
      uint32_t frame_index;
      uint32_t have_animation;
      uint32_t layer_have_crop;
      int32_t crop_x0;
      int32_t crop_y0;
      uint32_t layer_xsize;
      uint32_t layer_ysize;
      uint32_t duration;
      uint32_t num_extra_channels;
      uint32_t extra_channel_index;
      uint32_t extra_channel_type;
      char* extra_channel_name;
    } jxlpy_result;

    const char* jxlpy_version(void);
    void jxlpy_free_result(jxlpy_result* result);

    jxlpy_result jxlpy_encode_pixels(
        const void* pixels, size_t size, uint32_t xsize, uint32_t ysize,
        uint32_t num_channels, uint32_t dtype, uint32_t bits_per_sample,
        const jxlpy_encode_options* options);

    jxlpy_result jxlpy_encode_pixels_ex(
        const void* pixels, size_t size, uint32_t xsize, uint32_t ysize,
        uint32_t num_channels, uint32_t dtype, uint32_t bits_per_sample,
        const jxlpy_extra_channel* extra_channels, size_t num_extra_channels,
        const jxlpy_encode_options* options);

    jxlpy_result jxlpy_encode_image_bytes(
        const uint8_t* bytes, size_t size, const jxlpy_encode_options* options);

    jxlpy_result jxlpy_encode_multiframe(
        const jxlpy_frame* frames, size_t num_frames, uint32_t canvas_xsize,
        uint32_t canvas_ysize, uint32_t num_channels, uint32_t dtype,
        uint32_t bits_per_sample, const jxlpy_encode_options* options);

    jxlpy_result jxlpy_encode_multiframe_ex(
        const jxlpy_frame* frames, size_t num_frames, uint32_t canvas_xsize,
        uint32_t canvas_ysize, uint32_t num_channels, uint32_t dtype,
        uint32_t bits_per_sample, const jxlpy_extra_channel* extra_channels,
        size_t num_extra_channels_per_frame,
        const jxlpy_encode_options* options);

    jxlpy_result jxlpy_decode_jxl(
        const uint8_t* bytes, size_t size, int frame_index, int coalesced,
        uint32_t requested_channels, uint32_t requested_dtype);

    jxlpy_result jxlpy_decode_extra_channel_jxl(
        const uint8_t* bytes, size_t size, int frame_index, int coalesced,
        uint32_t extra_channel_index, uint32_t requested_dtype);

    jxlpy_result jxlpy_decode_image_bytes(
        const uint8_t* bytes, size_t size, int frame_index);

    jxlpy_result jxlpy_info(const uint8_t* bytes, size_t size);
    """
)


def _native_name() -> str:
    if sys.platform == "win32":
        return "jxlpy_native.dll"
    if sys.platform == "darwin":
        return "libjxlpy_native.dylib"
    return "libjxlpy_native.so"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_paths() -> list[Path]:
    env = os.environ.get("JXLPY_NATIVE_LIB")
    paths: list[Path] = []
    if env:
        paths.append(Path(env))

    name = _native_name()
    package_dir = Path(__file__).resolve().parent
    paths.append(package_dir / name)

    root = _repo_root()
    paths.extend(root.glob(f"out/build/*/{name}"))
    paths.extend(root.glob(f"out/build/*/*/{name}"))
    return paths


_DLL_DIR_HANDLES = []


def _add_dll_dirs(path: Path) -> None:
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    dirs = {path.parent}
    build_root = path.parent
    for dll in build_root.rglob("*.dll"):
        dirs.add(dll.parent)
    for directory in dirs:
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(directory)))
        except OSError:
            pass


def _load():
    attempted: list[str] = []
    for path in _candidate_paths():
        attempted.append(str(path))
        if path.exists():
            _add_dll_dirs(path)
            return ffi.dlopen(str(path))
    raise RuntimeError(
        "Cannot find jxlpy_native. Build it with CMake first or set "
        "JXLPY_NATIVE_LIB. Tried:\n" + "\n".join(attempted)
    )


lib = _load()
