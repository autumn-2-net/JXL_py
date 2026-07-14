from __future__ import annotations

import os
import sys
from pathlib import Path

from cffi import FFI

ffi = FFI()

ffi.cdef(
    """
    enum {
      JXLPY_NATIVE_ABI_VERSION = 2,
      JXLPY_DTYPE_UINT8 = 1,
      JXLPY_DTYPE_UINT16 = 2,
      JXLPY_DTYPE_FLOAT16 = 3,
      JXLPY_DTYPE_FLOAT32 = 4,
    };

    enum {
      JXLPY_ABI_STRUCT_ENCODE_OPTIONS = 1,
      JXLPY_ABI_STRUCT_EXTRA_CHANNEL = 2,
      JXLPY_ABI_STRUCT_FRAME = 3,
      JXLPY_ABI_STRUCT_RESULT = 4,
      JXLPY_ABI_STRUCT_EXTRA_CHANNEL_RESULT = 5,
      JXLPY_ABI_STRUCT_DECODE_ALL_RESULT = 6,
      JXLPY_ABI_STRUCT_COLOR_ENCODING = 7,
      JXLPY_ABI_STRUCT_ENCODER_SETTING = 8,
    };

    typedef struct {
      int32_t id;
      int32_t is_float;
      int64_t int_value;
      float float_value;
    } jxlpy_encoder_setting;

    typedef struct {
      uint32_t available;
      int32_t color_space;
      int32_t white_point;
      double white_point_xy[2];
      int32_t primaries;
      double primaries_red_xy[2];
      double primaries_green_xy[2];
      double primaries_blue_xy[2];
      int32_t transfer_function;
      double gamma;
      int32_t rendering_intent;
    } jxlpy_color_encoding;

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
      int lossless_jpeg;
      int allow_expert_options;
      int compress_boxes;
      int brotli_effort;
      int keep_invisible;
      int patches;
      int dots;
      int noise;
      int gaborish;
      int group_order;
      int center_x;
      int center_y;
      int progressive_ac;
      int qprogressive_ac;
      int progressive_dc;
      int responsive;
      int epf;
      int faster_decoding;
      int resampling;
      int ec_resampling;
      int already_downsampled;
      int upsampling_mode;
      float photon_noise_iso;
      float intensity_target;
      int premultiply;
      int override_bitdepth;
      int buffering;
      int jpeg_reconstruction_cfl;
      int disable_perceptual_optimizations;
      int modular_group_size;
      int modular_predictor;
      int modular_colorspace;
      float modular_ma_tree_learning_percent;
      int modular_nb_prev_channels;
      int modular_palette_colors;
      int modular_lossy_palette;
      float modular_channel_colors_global_percent;
      float modular_channel_colors_group_percent;
      const jxlpy_encoder_setting* extra_encoder_settings;
      size_t num_extra_encoder_settings;
      int color_encoding_mode;
      jxlpy_color_encoding color_encoding;
      const uint8_t* icc_profile;
      size_t icc_profile_size;
    } jxlpy_encode_options;

    typedef struct {
      const void* pixels;
      size_t size;
      uint32_t xsize;
      uint32_t ysize;
      uint32_t dtype;
      uint32_t bits_per_sample;
      uint32_t exponent_bits_per_sample;
      uint32_t type;
      const char* name;
      size_t name_size;
      uint32_t dim_shift;
      uint32_t alpha_premultiplied;
      float spot_color[4];
      uint32_t cfa_channel;
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
      uint32_t num_frames_known;
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
      uint32_t extra_channel_dim_shift;
      uint32_t extra_channel_alpha_premultiplied;
      float extra_channel_spot_color[4];
      uint32_t extra_channel_cfa_channel;
      jxlpy_color_encoding color_encoding;
      uint32_t color_profile_is_icc;
      uint8_t* icc_profile;
      size_t icc_profile_size;
      jxlpy_color_encoding data_color_encoding;
      uint32_t data_color_profile_is_icc;
      uint8_t* data_icc_profile;
      size_t data_icc_profile_size;
    } jxlpy_result;

    const char* jxlpy_version(void);
    uint32_t jxlpy_abi_version(void);
    size_t jxlpy_abi_struct_size(uint32_t struct_id);
    int jxlpy_supports_frame_settings_passthrough(void);
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
        uint32_t requested_channels, uint32_t requested_dtype, int threads,
        int scan_all_frames, uint64_t max_pixels, size_t max_output_bytes);

    jxlpy_result jxlpy_decode_extra_channel_jxl(
        const uint8_t* bytes, size_t size, int frame_index, int coalesced,
        uint32_t extra_channel_index, uint32_t requested_dtype, int threads,
        int scan_all_frames, uint64_t max_pixels, size_t max_output_bytes);

    typedef struct {
      uint32_t extra_channel_index;
      uint32_t extra_channel_type;
      uint32_t bits_per_sample;
      uint32_t exponent_bits_per_sample;
      uint32_t dtype;
      uint32_t xsize;
      uint32_t ysize;
      uint32_t dim_shift;
      uint32_t alpha_premultiplied;
      float spot_color[4];
      uint32_t cfa_channel;
      char* extra_channel_name;
      uint8_t* data;
      size_t size;
    } jxlpy_extra_channel_result;

    typedef struct {
      int ok;
      char* error;
      uint8_t* color_data;
      size_t color_size;
      uint32_t xsize;
      uint32_t ysize;
      uint32_t num_channels;
      uint32_t dtype;
      uint32_t bits_per_sample;
      uint32_t exponent_bits_per_sample;
      uint32_t num_frames;
      uint32_t num_frames_known;
      uint32_t frame_index;
      uint32_t have_animation;
      uint32_t layer_have_crop;
      int32_t crop_x0;
      int32_t crop_y0;
      uint32_t layer_xsize;
      uint32_t layer_ysize;
      uint32_t duration;
      uint32_t num_extra_channels;
      jxlpy_extra_channel_result* extra_channels;
      jxlpy_color_encoding color_encoding;
      uint32_t color_profile_is_icc;
      uint8_t* icc_profile;
      size_t icc_profile_size;
      jxlpy_color_encoding data_color_encoding;
      uint32_t data_color_profile_is_icc;
      uint8_t* data_icc_profile;
      size_t data_icc_profile_size;
    } jxlpy_decode_all_result;

    jxlpy_decode_all_result jxlpy_decode_all_jxl(
        const uint8_t* bytes, size_t size, int frame_index, int coalesced,
        uint32_t requested_channels, uint32_t requested_dtype, int threads,
        int scan_all_frames, uint64_t max_pixels, size_t max_output_bytes);

    void jxlpy_free_decode_all_result(jxlpy_decode_all_result* result);

    jxlpy_result jxlpy_decode_image_bytes(
        const uint8_t* bytes, size_t size, int frame_index);

    jxlpy_result jxlpy_info(const uint8_t* bytes, size_t size);

    jxlpy_decode_all_result jxlpy_info_all(
        const uint8_t* bytes, size_t size);

    jxlpy_result jxlpy_reconstruct_jpeg(const uint8_t* bytes, size_t size);

    jxlpy_result jxlpy_decode_to_format(
        const uint8_t* bytes, size_t size, const char* extension, int quality);
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
    paths.append(root / "out" / "build" / name)
    paths.extend(root.glob(f"out/build/*/{name}"))
    paths.extend(root.glob(f"out/build/*/*/{name}"))
    return paths


_DLL_DIR_HANDLES = []

_ABI_STRUCTS = {
    1: "jxlpy_encode_options",
    2: "jxlpy_extra_channel",
    3: "jxlpy_frame",
    4: "jxlpy_result",
    5: "jxlpy_extra_channel_result",
    6: "jxlpy_decode_all_result",
    7: "jxlpy_color_encoding",
    8: "jxlpy_encoder_setting",
}


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


def _validate_abi(native, path: Path) -> None:
    try:
        version = int(native.jxlpy_abi_version())
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            f"{path} predates the checked native ABI; rebuild jxlpy_native"
        ) from exc
    expected = int(ffi.cast("uint32_t", 2))
    if version != expected:
        raise RuntimeError(
            f"{path} has native ABI {version}, but Python expects {expected}; "
            "rebuild jxlpy_native"
        )
    mismatches = []
    for struct_id, c_name in _ABI_STRUCTS.items():
        native_size = int(native.jxlpy_abi_struct_size(struct_id))
        python_size = int(ffi.sizeof(c_name))
        if native_size != python_size:
            mismatches.append(f"{c_name}: native={native_size}, cffi={python_size}")
    if mismatches:
        raise RuntimeError(
            f"{path} has incompatible ABI structure sizes: " + "; ".join(mismatches)
        )


def _load():
    attempted: list[str] = []
    errors: list[str] = []
    for path in _candidate_paths():
        attempted.append(str(path))
        if path.exists():
            _add_dll_dirs(path)
            try:
                native = ffi.dlopen(str(path))
                _validate_abi(native, path)
                return native
            except (AttributeError, OSError, RuntimeError) as e:
                errors.append(f"  {path}: {e}")
    msg = "Cannot find or load jxlpy_native. Build it with CMake first or set JXLPY_NATIVE_LIB."
    msg += "\nSearched:\n" + "\n".join(f"  {p}" for p in attempted)
    if errors:
        msg += "\nLoad errors:\n" + "\n".join(errors)
    raise RuntimeError(msg)


lib = _load()
