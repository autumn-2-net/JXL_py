from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ._ffi import ffi, lib
from .metadata import (
    EXTRA_TYPE_TO_NATIVE as _EXTRA_TYPE_TO_NATIVE,
    NATIVE_EXTRA_TYPE as _NATIVE_EXTRA_TYPE,
    color_encoding_to_dict as _color_encoding_to_dict,
    fill_color_encoding as _fill_color_encoding,
    parse_extra_channel as _parse_extra_channel,
    read_icc_profile as _read_icc_profile,
)
from .multiframe import (
    analyze_frame_arrays as _analyze_frame_arrays,
    changed_mask as _changed_mask,
    diff_stats as _diff_stats,
    diff_stats_from_mask as _diff_stats_from_mask,
    select_reference_bbox as _select_reference_bbox,
)


_DTYPE_TO_NATIVE = {
    np.dtype("uint8"): lib.JXLPY_DTYPE_UINT8,
    np.dtype("uint16"): lib.JXLPY_DTYPE_UINT16,
    np.dtype("float16"): lib.JXLPY_DTYPE_FLOAT16,
    np.dtype("float32"): lib.JXLPY_DTYPE_FLOAT32,
}

_NATIVE_TO_DTYPE = {
    lib.JXLPY_DTYPE_UINT8: np.dtype("uint8"),
    lib.JXLPY_DTYPE_UINT16: np.dtype("uint16"),
    lib.JXLPY_DTYPE_FLOAT16: np.dtype("float16"),
    lib.JXLPY_DTYPE_FLOAT32: np.dtype("float32"),
}

_BLEND_MODE_TO_NATIVE = {
    "replace": 0,
    "add": 1,
    "blend": 2,
    "muladd": 3,
    "mul": 4,
}


@dataclass(frozen=True)
class _NativeResult:
    data: bytes
    meta: dict[str, Any]


def _read_bytes(src: Any) -> bytes | bytearray | memoryview:
    if isinstance(src, Path):
        return src.read_bytes()
    if isinstance(src, str):
        return Path(src).read_bytes()
    if isinstance(src, bytes):
        return src
    if isinstance(src, bytearray):
        return src
    if isinstance(src, memoryview):
        if not src.contiguous:
            return memoryview(src.tobytes())
        return src if src.format == "B" and src.ndim == 1 else src.cast("B")
    raise TypeError("expected a path or bytes-like object")


def _is_jxl(data: bytes | bytearray | memoryview) -> bool:
    prefix = bytes(memoryview(data)[:12])
    return prefix.startswith(b"\xff\x0a") or prefix.startswith(
        b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a"
    )


def _torch_to_numpy(value: Any) -> np.ndarray | None:
    try:
        import torch
    except Exception:
        return None
    if not isinstance(value, torch.Tensor):
        return None
    return value.detach().cpu().contiguous().numpy()


def _as_array(value: Any, *, layout: str = "auto") -> np.ndarray:
    tensor_array = _torch_to_numpy(value)
    arr = tensor_array if tensor_array is not None else np.asarray(value)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim != 3:
        raise ValueError("expected a 2D image or a 3D HWC/CHW array")

    if layout == "auto":
        if 1 <= arr.shape[-1] <= 4:
            pass
        elif 1 <= arr.shape[0] <= 4:
            arr = np.moveaxis(arr, 0, -1)
        else:
            raise ValueError("cannot infer channel axis; pass an HWC or CHW array")
    elif layout == "chw":
        arr = np.moveaxis(arr, 0, -1)
    elif layout != "hwc":
        raise ValueError("layout must be 'auto', 'hwc' or 'chw'")

    dtype = np.dtype(arr.dtype)
    if dtype not in _DTYPE_TO_NATIVE:
        raise TypeError("supported dtypes are uint8, uint16, float16 and float32")
    if not 1 <= arr.shape[-1] <= 4:
        raise ValueError("main image channels must be 1, 2, 3 or 4")
    return np.ascontiguousarray(arr)


def _as_extra_array(value: Any, *, layout: str = "auto") -> np.ndarray:
    tensor_array = _torch_to_numpy(value)
    arr = tensor_array if tensor_array is not None else np.asarray(value)
    if arr.ndim == 3:
        if layout == "chw":
            arr = np.moveaxis(arr, 0, -1)
        elif layout == "auto" and arr.shape[0] == 1 and arr.shape[-1] != 1:
            arr = np.moveaxis(arr, 0, -1)
        if arr.shape[-1] != 1:
            raise ValueError("extra channel arrays must be 2D or single-channel")
        arr = arr[:, :, 0]
    elif arr.ndim != 2:
        raise ValueError("extra channel arrays must be 2D or single-channel")

    dtype = np.dtype(arr.dtype)
    if dtype not in _DTYPE_TO_NATIVE:
        raise TypeError("supported extra channel dtypes are uint8, uint16, float16 and float32")
    return np.ascontiguousarray(arr)


def _is_tensor(value: Any) -> bool:
    try:
        import torch
    except Exception:
        return False
    return isinstance(value, torch.Tensor)


def _looks_like_frame_sequence(value: Any, count: int) -> bool:
    if isinstance(value, (str, bytes, bytearray, memoryview, Path, np.ndarray)):
        return False
    if _is_tensor(value):
        return False
    return isinstance(value, (list, tuple)) and len(value) == count


def _optional_int(value: int | None) -> int:
    return -1 if value is None else int(value)


def _optional_float(value: float | None) -> float:
    return -1.0 if value is None else float(value)


def _optional_bool(value: bool | None) -> int:
    return -1 if value is None else (1 if value else 0)


def _pack_source_ref(
    source_ref: int,
    *,
    blend_mode: str = "replace",
    alpha: int = 0,
    clamp: bool = False,
) -> int:
    source = int(source_ref)
    if not 0 <= source <= 3:
        raise ValueError("source_ref must be in 0..3")
    key = blend_mode.lower().replace("-", "").replace("_", "")
    if key not in _BLEND_MODE_TO_NATIVE:
        raise ValueError(f"unknown blend mode: {blend_mode!r}")
    alpha_index = int(alpha)
    if not 0 <= alpha_index <= 255:
        raise ValueError("blend alpha index must be in 0..255")
    return (
        source
        | (_BLEND_MODE_TO_NATIVE[key] << 8)
        | (alpha_index << 16)
        | ((1 if clamp else 0) << 24)
    )


def _resolve_alias(primary_name: str, primary_value: Any, alias_name: str, alias_value: Any):
    if alias_value is None:
        return primary_value
    if primary_value is not None and primary_value != alias_value:
        raise ValueError(f"set either {primary_name} or {alias_name}, not both")
    return alias_value


_FRAME_SETTING_IDS = {
    "effort": 0,
    "decoding_speed": 1,
    "resampling": 2,
    "extra_channel_resampling": 3,
    "already_downsampled": 4,
    "photon_noise": 5,
    "noise": 6,
    "dots": 7,
    "patches": 8,
    "epf": 9,
    "gaborish": 10,
    "modular": 11,
    "keep_invisible": 12,
    "group_order": 13,
    "group_order_center_x": 14,
    "group_order_center_y": 15,
    "responsive": 16,
    "progressive_ac": 17,
    "qprogressive_ac": 18,
    "progressive_dc": 19,
    "channel_colors_global_percent": 20,
    "channel_colors_group_percent": 21,
    "palette_colors": 22,
    "lossy_palette": 23,
    "color_transform": 24,
    "modular_color_space": 25,
    "modular_group_size": 26,
    "modular_predictor": 27,
    "modular_ma_tree_learning_percent": 28,
    "modular_nb_prev_channels": 29,
    "jpeg_recon_cfl": 30,
    "frame_index_box": 31,
    "brotli_effort": 32,
    "jpeg_compress_boxes": 33,
    "buffering": 34,
    "jpeg_keep_exif": 35,
    "jpeg_keep_xmp": 36,
    "jpeg_keep_jumbf": 37,
    "use_full_image_heuristics": 38,
    "disable_perceptual_heuristics": 39,
    "output_mode": 40,
}

_FRAME_SETTING_ALIASES = {
    "faster_decoding": "decoding_speed",
    "ec_resampling": "extra_channel_resampling",
    "photon_noise_iso": "photon_noise",
    "center_x": "group_order_center_x",
    "center_y": "group_order_center_y",
    "pre_compact": "channel_colors_global_percent",
    "post_compact": "channel_colors_group_percent",
    "modular_channel_colors_global_percent": "channel_colors_global_percent",
    "modular_channel_colors_group_percent": "channel_colors_group_percent",
    "modular_palette_colors": "palette_colors",
    "modular_lossy_palette": "lossy_palette",
    "modular_colorspace": "modular_color_space",
    "iterations": "modular_ma_tree_learning_percent",
    "jpeg_reconstruction_cfl": "jpeg_recon_cfl",
    "frame_indexing": "frame_index_box",
    "index_box": "frame_index_box",
    "compress_boxes": "jpeg_compress_boxes",
    "disable_perceptual_optimizations": "disable_perceptual_heuristics",
    "use_full_image": "use_full_image_heuristics",
}

_FLOAT_FRAME_SETTING_IDS = {5, 20, 21, 28}


def _normalize_frame_setting_key(key: Any) -> int:
    if isinstance(key, int):
        return key
    text = str(key).strip()
    if text.isdigit():
        return int(text)
    text = text.lower().replace("-", "_")
    prefix = "jxl_enc_frame_setting_"
    if text.startswith(prefix):
        text = text[len(prefix):]
    prefix = "jxl_enc_frame_"
    if text.startswith(prefix):
        text = text[len(prefix):]
    text = _FRAME_SETTING_ALIASES.get(text, text)
    if text not in _FRAME_SETTING_IDS:
        raise ValueError(f"unknown encoder option or frame setting: {key!r}")
    return _FRAME_SETTING_IDS[text]


def _coerce_frame_setting_value(setting_id: int, value: Any) -> tuple[bool, int, float]:
    force_float = None
    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and str(value[0]).lower() in ("int", "float")
    ):
        force_float = str(value[0]).lower() == "float"
        value = value[1]

    if isinstance(value, bool):
        value = 1 if value else 0

    is_float = (
        force_float
        if force_float is not None
        else setting_id in _FLOAT_FRAME_SETTING_IDS
        or (isinstance(value, float) and not value.is_integer())
    )
    if is_float:
        return True, 0, float(value)
    return False, int(value), 0.0


def _native_supports_frame_settings_passthrough() -> bool:
    try:
        return bool(lib.jxlpy_supports_frame_settings_passthrough())
    except AttributeError:
        return False


def _iter_frame_settings(settings: Any):
    if settings is None:
        return []
    if isinstance(settings, Mapping):
        return list(settings.items())
    return list(settings)


def _make_frame_settings(settings: Any, keepalive: list[Any]):
    entries = _iter_frame_settings(settings)
    if not entries:
        return ffi.NULL, 0
    parsed = []
    for entry in entries:
        if len(entry) != 2:
            raise ValueError("frame_settings entries must be (name_or_id, value)")
        key, value = entry
        setting_id = _normalize_frame_setting_key(key)
        is_float, int_value, float_value = _coerce_frame_setting_value(
            setting_id, value
        )
        parsed.append((setting_id, is_float, int_value, float_value))
    if not _native_supports_frame_settings_passthrough():
        raise RuntimeError(
            "frame_settings passthrough requires rebuilding jxlpy_native"
        )
    c_settings = ffi.new("jxlpy_encoder_setting[]", len(parsed))
    for i, (setting_id, is_float, int_value, float_value) in enumerate(parsed):
        c_settings[i].id = setting_id
        c_settings[i].is_float = 1 if is_float else 0
        c_settings[i].int_value = int_value
        c_settings[i].float_value = float_value
    keepalive.append(c_settings)
    return c_settings, len(entries)


def _merge_encoder_options(
    option_kwargs: dict[str, Any],
    encoder_options: Mapping[str, Any] | None,
    frame_settings: Any,
):
    merged_frame_settings = []
    if frame_settings is not None:
        merged_frame_settings.extend(_iter_frame_settings(frame_settings))
    if encoder_options is None:
        return merged_frame_settings
    if not isinstance(encoder_options, Mapping):
        raise TypeError("encoder_options must be a mapping")
    for key, value in encoder_options.items():
        if key == "frame_settings":
            merged_frame_settings.extend(_iter_frame_settings(value))
        elif key in option_kwargs:
            option_kwargs[key] = value
        else:
            merged_frame_settings.append((key, value))
    return merged_frame_settings


def _make_extra_structs(
    specs: Iterable[Any] | None,
    *,
    expected_hw: tuple[int, int],
    layout: str,
):
    specs = list(specs or [])
    c_extras = ffi.new("jxlpy_extra_channel[]", len(specs))
    buffers: list[np.ndarray] = []
    c_buffers = []
    name_buffers = []

    for i, spec in enumerate(specs):
        parsed = _parse_extra_channel(spec)
        arr = _as_extra_array(parsed.data, layout=layout)
        if arr.shape != expected_hw:
            raise ValueError("extra channel dimensions must match the main image")
        buffers.append(arr)
        c_buf = ffi.from_buffer(arr)
        c_buffers.append(c_buf)
        name_bytes = parsed.name.encode("utf-8")
        c_name = ffi.new("char[]", name_bytes) if name_bytes else ffi.NULL
        name_buffers.append(c_name)
        c_extras[i].pixels = c_buf
        c_extras[i].size = arr.nbytes
        c_extras[i].xsize = arr.shape[1]
        c_extras[i].ysize = arr.shape[0]
        c_extras[i].dtype = _DTYPE_TO_NATIVE[arr.dtype]
        c_extras[i].bits_per_sample = parsed.bits_per_sample
        c_extras[i].exponent_bits_per_sample = parsed.exponent_bits_per_sample
        c_extras[i].type = parsed.type_id
        c_extras[i].name = c_name
        c_extras[i].name_size = len(name_bytes)
        c_extras[i].dim_shift = parsed.dim_shift
        c_extras[i].alpha_premultiplied = 1 if parsed.alpha_premultiplied else 0
        for channel, value in enumerate(parsed.spot_color):
            c_extras[i].spot_color[channel] = value
        c_extras[i].cfa_channel = parsed.cfa_channel

    return c_extras, buffers, c_buffers, name_buffers


def _extra_specs_to_frame_arrays(
    specs: Iterable[Any] | None,
    *,
    frame_count: int,
    expected_hw: tuple[int, int],
    layout: str,
):
    channels = []
    for spec in list(specs or []):
        parsed = _parse_extra_channel(spec)
        if _looks_like_frame_sequence(parsed.data, frame_count):
            arrays = [_as_extra_array(item, layout=layout) for item in parsed.data]
        else:
            one = _as_extra_array(parsed.data, layout=layout)
            arrays = [one for _ in range(frame_count)]
        for arr in arrays:
            if arr.shape != expected_hw:
                raise ValueError("extra channel dimensions must match every frame")
        channels.append(
            {
                "name": parsed.name,
                "type_id": parsed.type_id,
                "bits_per_sample": parsed.bits_per_sample,
                "exponent_bits_per_sample": parsed.exponent_bits_per_sample,
                "dim_shift": parsed.dim_shift,
                "alpha_premultiplied": parsed.alpha_premultiplied,
                "spot_color": parsed.spot_color,
                "cfa_channel": parsed.cfa_channel,
                "arrays": arrays,
            }
        )
    return channels


def _ensure_dim_shift_resampling(
    option_kwargs: dict[str, Any],
    specs: Iterable[Any],
    frame_settings: Iterable[Any],
) -> None:
    max_shift = max(
        (
            int(spec.dim_shift)
            if hasattr(spec, "dim_shift")
            else int(spec.get("dim_shift", 0))
            for spec in specs
        ),
        default=0,
    )
    if max_shift == 0:
        return

    required = 1 << max_shift
    configured = option_kwargs.get("ec_resampling")
    if configured is None or int(configured) == -1:
        option_kwargs["ec_resampling"] = required
    elif int(configured) not in (1, 2, 4, 8):
        raise ValueError("ec_resampling must be one of 1, 2, 4 or 8")
    elif int(configured) < required:
        raise ValueError(
            f"dim_shift={max_shift} requires ec_resampling >= {required}"
        )

    for key, value in frame_settings:
        if _normalize_frame_setting_key(key) != 3:
            continue
        is_float, int_value, _ = _coerce_frame_setting_value(3, value)
        if is_float or int_value not in (2, 4, 8) or int_value < required:
            raise ValueError(
                "extra_channel_resampling in frame_settings conflicts with "
                f"dim_shift={max_shift}; use {required} or greater"
            )


def _options(
    *,
    lossless: bool | None = None,
    distance: float | None = None,
    alpha_distance: float = 0.0,
    effort: int = 7,
    modular: int | None = None,
    level: int = -1,
    threads: int = 0,
    use_container: bool = False,
    jpeg_store_metadata: bool = True,
    lossless_jpeg: bool = True,
    allow_expert_options: bool = False,
    compress_boxes: bool = True,
    brotli_effort: int | None = None,
    keep_invisible: bool | None = None,
    patches: bool | None = None,
    dots: bool | None = None,
    noise: bool | None = None,
    gaborish: bool | None = None,
    group_order: int | None = None,
    center_x: int | None = None,
    center_y: int | None = None,
    progressive: bool = False,
    progressive_ac: bool | None = None,
    qprogressive_ac: bool | None = None,
    progressive_dc: int | None = None,
    responsive: bool | None = None,
    epf: int | None = None,
    faster_decoding: int | None = None,
    resampling: int | None = None,
    ec_resampling: int | None = None,
    already_downsampled: bool | None = None,
    upsampling_mode: int | None = None,
    photon_noise_iso: float = 0.0,
    intensity_target: float = 0.0,
    premultiply: int | None = None,
    override_bitdepth: int = 0,
    buffering: int | None = None,
    jpeg_reconstruction_cfl: bool | None = None,
    disable_perceptual_optimizations: bool = False,
    modular_group_size: int | None = None,
    modular_predictor: int | None = None,
    modular_colorspace: int | None = None,
    modular_ma_tree_learning_percent: float | None = None,
    iterations: float | None = None,
    modular_nb_prev_channels: int | None = None,
    modular_palette_colors: int | None = None,
    modular_lossy_palette: bool | None = None,
    modular_channel_colors_global_percent: float | None = None,
    modular_channel_colors_group_percent: float | None = None,
    pre_compact: float | None = None,
    post_compact: float | None = None,
    color_encoding: str | Mapping[str, Any] | None = None,
    icc_profile: Any = None,
    tps: tuple[int, int] = (1000, 1),
    frame_settings: Any = None,
    _keepalive: list[Any] | None = None,
):
    modular_ma_tree_learning_percent = _resolve_alias(
        "modular_ma_tree_learning_percent",
        modular_ma_tree_learning_percent,
        "iterations",
        iterations,
    )
    modular_channel_colors_global_percent = _resolve_alias(
        "modular_channel_colors_global_percent",
        modular_channel_colors_global_percent,
        "pre_compact",
        pre_compact,
    )
    modular_channel_colors_group_percent = _resolve_alias(
        "modular_channel_colors_group_percent",
        modular_channel_colors_group_percent,
        "post_compact",
        post_compact,
    )
    if lossless is None:
        lossless = distance is None or float(distance) == 0.0
    if distance is None:
        distance = 0.0 if lossless else 1.0
    if progressive:
        if progressive_ac is None:
            progressive_ac = True
        if progressive_dc is None:
            progressive_dc = 1
        if group_order is None:
            group_order = 1
        if patches is None:
            patches = False
        if responsive is None:
            responsive = True
    opts = ffi.new("jxlpy_encode_options *")
    opts.lossless = 1 if lossless else 0
    opts.distance = float(distance)
    opts.alpha_distance = float(alpha_distance)
    opts.effort = int(effort)
    opts.modular = -1 if modular is None else int(modular)
    opts.level = int(level)
    opts.threads = int(threads)
    opts.use_container = 1 if use_container else 0
    opts.jpeg_store_metadata = 1 if jpeg_store_metadata else 0
    opts.tps_numerator = int(tps[0])
    opts.tps_denominator = int(tps[1])
    opts.lossless_jpeg = 1 if lossless_jpeg else 0
    opts.allow_expert_options = 1 if allow_expert_options else 0
    opts.compress_boxes = 1 if compress_boxes else 0
    opts.brotli_effort = _optional_int(brotli_effort)
    opts.keep_invisible = _optional_bool(keep_invisible)
    opts.patches = _optional_bool(patches)
    opts.dots = _optional_bool(dots)
    opts.noise = _optional_bool(noise)
    opts.gaborish = _optional_bool(gaborish)
    opts.group_order = _optional_int(group_order)
    opts.center_x = _optional_int(center_x)
    opts.center_y = _optional_int(center_y)
    opts.progressive_ac = _optional_bool(progressive_ac)
    opts.qprogressive_ac = _optional_bool(qprogressive_ac)
    opts.progressive_dc = _optional_int(progressive_dc)
    opts.responsive = _optional_bool(responsive)
    opts.epf = _optional_int(epf)
    opts.faster_decoding = _optional_int(faster_decoding)
    opts.resampling = _optional_int(resampling)
    opts.ec_resampling = _optional_int(ec_resampling)
    opts.already_downsampled = _optional_bool(already_downsampled)
    opts.upsampling_mode = _optional_int(upsampling_mode)
    opts.photon_noise_iso = float(photon_noise_iso)
    opts.intensity_target = float(intensity_target)
    opts.premultiply = _optional_int(premultiply)
    opts.override_bitdepth = int(override_bitdepth)
    opts.buffering = _optional_int(buffering)
    opts.jpeg_reconstruction_cfl = _optional_bool(jpeg_reconstruction_cfl)
    opts.disable_perceptual_optimizations = (
        1 if disable_perceptual_optimizations else 0
    )
    opts.modular_group_size = _optional_int(modular_group_size)
    opts.modular_predictor = _optional_int(modular_predictor)
    opts.modular_colorspace = _optional_int(modular_colorspace)
    opts.modular_ma_tree_learning_percent = _optional_float(
        modular_ma_tree_learning_percent
    )
    opts.modular_nb_prev_channels = _optional_int(modular_nb_prev_channels)
    opts.modular_palette_colors = _optional_int(modular_palette_colors)
    opts.modular_lossy_palette = _optional_bool(modular_lossy_palette)
    opts.modular_channel_colors_global_percent = _optional_float(
        modular_channel_colors_global_percent
    )
    opts.modular_channel_colors_group_percent = _optional_float(
        modular_channel_colors_group_percent
    )
    keepalive = [] if _keepalive is None else _keepalive
    if color_encoding is not None and icc_profile is not None:
        raise ValueError("set either color_encoding or icc_profile, not both")
    if color_encoding is not None:
        opts.color_encoding_mode = 1
        _fill_color_encoding(opts.color_encoding, color_encoding)
    elif icc_profile is not None:
        icc_data = _read_icc_profile(icc_profile)
        if not icc_data:
            raise ValueError("icc_profile must not be empty")
        c_icc = ffi.from_buffer(icc_data)
        keepalive.extend((icc_data, c_icc))
        opts.color_encoding_mode = 2
        opts.icc_profile = c_icc
        opts.icc_profile_size = len(icc_data)
    c_settings, num_settings = _make_frame_settings(frame_settings, keepalive)
    opts.extra_encoder_settings = c_settings
    opts.num_extra_encoder_settings = num_settings
    return opts


def _meta_from_result(result) -> dict[str, Any]:
    extra_name = (
        ffi.string(result.extra_channel_name).decode("utf-8", "replace")
        if result.extra_channel_name != ffi.NULL
        else ""
    )
    icc_profile = (
        bytes(ffi.buffer(result.icc_profile, result.icc_profile_size))
        if result.icc_profile_size
        else None
    )
    data_icc_profile = (
        bytes(ffi.buffer(result.data_icc_profile, result.data_icc_profile_size))
        if result.data_icc_profile_size
        else None
    )
    return {
        "xsize": int(result.xsize),
        "ysize": int(result.ysize),
        "num_channels": int(result.num_channels),
        "dtype": _NATIVE_TO_DTYPE.get(int(result.dtype)),
        "bits_per_sample": int(result.bits_per_sample),
        "exponent_bits_per_sample": int(result.exponent_bits_per_sample),
        "num_frames": int(result.num_frames),
        "num_frames_known": bool(result.num_frames_known),
        "frame_index": int(result.frame_index),
        "have_animation": bool(result.have_animation),
        "layer_have_crop": bool(result.layer_have_crop),
        "crop_x0": int(result.crop_x0),
        "crop_y0": int(result.crop_y0),
        "layer_xsize": int(result.layer_xsize),
        "layer_ysize": int(result.layer_ysize),
        "duration": int(result.duration),
        "num_extra_channels": int(result.num_extra_channels),
        "extra_channel_index": int(result.extra_channel_index),
        "extra_channel_type": _NATIVE_EXTRA_TYPE.get(
            int(result.extra_channel_type), "unknown"
        ),
        "extra_channel_name": extra_name,
        "extra_channel_dim_shift": int(result.extra_channel_dim_shift),
        "extra_channel_alpha_premultiplied": bool(
            result.extra_channel_alpha_premultiplied
        ),
        "extra_channel_spot_color": tuple(
            float(value) for value in result.extra_channel_spot_color
        ),
        "extra_channel_cfa_channel": int(result.extra_channel_cfa_channel),
        "color_encoding": _color_encoding_to_dict(result.color_encoding),
        "color_profile_is_icc": bool(result.color_profile_is_icc),
        "icc_profile": icc_profile,
        "data_color_encoding": _color_encoding_to_dict(
            result.data_color_encoding
        ),
        "data_color_profile_is_icc": bool(result.data_color_profile_is_icc),
        "data_icc_profile": data_icc_profile,
    }


def _consume_result(result) -> _NativeResult:
    holder = ffi.new("jxlpy_result *", result)
    try:
        if not result.ok:
            message = (
                ffi.string(result.error).decode("utf-8", "replace")
                if result.error != ffi.NULL
                else "native call failed"
            )
            raise RuntimeError(message)
        data = bytes(ffi.buffer(result.data, result.size)) if result.size else b""
        return _NativeResult(data=data, meta=_meta_from_result(result))
    finally:
        lib.jxlpy_free_result(holder)


def _consume_pixels_result(
    result,
    out: str,
    *,
    plane: bool = False,
    max_pixels: int = 0,
    max_output_bytes: int = 0,
) -> tuple[Any, dict[str, Any]]:
    holder = ffi.new("jxlpy_result *", result)
    try:
        if not result.ok:
            message = (
                ffi.string(result.error).decode("utf-8", "replace")
                if result.error != ffi.NULL
                else "native call failed"
            )
            raise RuntimeError(message)
        meta = _meta_from_result(result)
        pixels = meta["xsize"] * meta["ysize"]
        if max_pixels and pixels > max_pixels:
            raise RuntimeError("decoded image exceeds max_pixels")
        if max_output_bytes and int(result.size) > max_output_bytes:
            raise RuntimeError("decoded image exceeds max_output_bytes")
        if out == "raw":
            return bytes(ffi.buffer(result.data, result.size)), meta
        dtype = meta["dtype"]
        if dtype is None:
            raise RuntimeError("native decoder returned an unknown dtype")
        array = np.frombuffer(ffi.buffer(result.data, result.size), dtype=dtype)
        shape = (
            (meta["ysize"], meta["xsize"])
            if plane
            else (meta["ysize"], meta["xsize"], meta["num_channels"])
        )
        array = array.reshape(shape).copy()
        if out == "numpy":
            return array, meta
        if out == "torch":
            import torch

            return torch.from_numpy(array), meta
        raise ValueError("out must be 'numpy', 'torch' or 'raw'")
    finally:
        lib.jxlpy_free_result(holder)


def _write_or_return(data: bytes, output: str | Path | None):
    if output is None:
        return data
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def encode(
    src: Any,
    output: str | Path | None = None,
    *,
    layout: str = "auto",
    extra_channels: Iterable[Any] | None = None,
    color_encoding: str | Mapping[str, Any] | None = None,
    icc_profile: Any = None,
    bits_per_sample: int = 0,
    lossless: bool | None = None,
    distance: float | None = None,
    alpha_distance: float = 0.0,
    effort: int = 7,
    modular: int | None = None,
    level: int = -1,
    threads: int = 0,
    use_container: bool = False,
    jpeg_store_metadata: bool = True,
    lossless_jpeg: bool = True,
    allow_expert_options: bool = False,
    compress_boxes: bool = True,
    brotli_effort: int | None = None,
    keep_invisible: bool | None = None,
    patches: bool | None = None,
    dots: bool | None = None,
    noise: bool | None = None,
    gaborish: bool | None = None,
    group_order: int | None = None,
    center_x: int | None = None,
    center_y: int | None = None,
    progressive: bool = False,
    progressive_ac: bool | None = None,
    qprogressive_ac: bool | None = None,
    progressive_dc: int | None = None,
    responsive: bool | None = None,
    epf: int | None = None,
    faster_decoding: int | None = None,
    resampling: int | None = None,
    ec_resampling: int | None = None,
    already_downsampled: bool | None = None,
    upsampling_mode: int | None = None,
    photon_noise_iso: float = 0.0,
    intensity_target: float = 0.0,
    premultiply: int | None = None,
    override_bitdepth: int = 0,
    buffering: int | None = None,
    jpeg_reconstruction_cfl: bool | None = None,
    disable_perceptual_optimizations: bool = False,
    modular_group_size: int | None = None,
    modular_predictor: int | None = None,
    modular_colorspace: int | None = None,
    modular_ma_tree_learning_percent: float | None = None,
    iterations: float | None = None,
    modular_nb_prev_channels: int | None = None,
    modular_palette_colors: int | None = None,
    modular_lossy_palette: bool | None = None,
    modular_channel_colors_global_percent: float | None = None,
    modular_channel_colors_group_percent: float | None = None,
    pre_compact: float | None = None,
    post_compact: float | None = None,
    encoder_options: Mapping[str, Any] | None = None,
    frame_settings: Any = None,
):
    """Encode a path, encoded image bytes, numpy array or torch tensor to JXL."""
    extra_channels = list(extra_channels or [])
    parsed_extra_channels = [
        _parse_extra_channel(spec) for spec in extra_channels
    ]
    option_kwargs = dict(
        lossless=lossless,
        distance=distance,
        alpha_distance=alpha_distance,
        effort=effort,
        modular=modular,
        level=level,
        threads=threads,
        use_container=use_container,
        jpeg_store_metadata=jpeg_store_metadata,
        lossless_jpeg=lossless_jpeg,
        allow_expert_options=allow_expert_options,
        compress_boxes=compress_boxes,
        brotli_effort=brotli_effort,
        keep_invisible=keep_invisible,
        patches=patches,
        dots=dots,
        noise=noise,
        gaborish=gaborish,
        group_order=group_order,
        center_x=center_x,
        center_y=center_y,
        progressive=progressive,
        progressive_ac=progressive_ac,
        qprogressive_ac=qprogressive_ac,
        progressive_dc=progressive_dc,
        responsive=responsive,
        epf=epf,
        faster_decoding=faster_decoding,
        resampling=resampling,
        ec_resampling=ec_resampling,
        already_downsampled=already_downsampled,
        upsampling_mode=upsampling_mode,
        photon_noise_iso=photon_noise_iso,
        intensity_target=intensity_target,
        premultiply=premultiply,
        override_bitdepth=override_bitdepth,
        buffering=buffering,
        jpeg_reconstruction_cfl=jpeg_reconstruction_cfl,
        disable_perceptual_optimizations=disable_perceptual_optimizations,
        modular_group_size=modular_group_size,
        modular_predictor=modular_predictor,
        modular_colorspace=modular_colorspace,
        modular_ma_tree_learning_percent=modular_ma_tree_learning_percent,
        iterations=iterations,
        modular_nb_prev_channels=modular_nb_prev_channels,
        modular_palette_colors=modular_palette_colors,
        modular_lossy_palette=modular_lossy_palette,
        modular_channel_colors_global_percent=modular_channel_colors_global_percent,
        modular_channel_colors_group_percent=modular_channel_colors_group_percent,
        pre_compact=pre_compact,
        post_compact=post_compact,
        color_encoding=color_encoding,
        icc_profile=icc_profile,
    )
    merged_frame_settings = _merge_encoder_options(
        option_kwargs, encoder_options, frame_settings
    )
    _ensure_dim_shift_resampling(
        option_kwargs, parsed_extra_channels, merged_frame_settings
    )
    option_keepalive: list[Any] = []
    opts = _options(
        **option_kwargs,
        frame_settings=merged_frame_settings,
        _keepalive=option_keepalive,
    )

    if isinstance(src, (str, Path, bytes, bytearray, memoryview)):
        if extra_channels:
            raise ValueError("extra_channels are only supported for array/tensor input")
        data = _read_bytes(src)
        c_data = ffi.from_buffer(data)
        result = lib.jxlpy_encode_image_bytes(c_data, len(data), opts)
        return _write_or_return(_consume_result(result).data, output)

    arr = _as_array(src, layout=layout)
    h, w, channels = arr.shape
    c_pixels = ffi.from_buffer(arr)
    c_extras, extra_buffers, extra_c_buffers, extra_names = _make_extra_structs(
        parsed_extra_channels,
        expected_hw=(h, w),
        layout=layout,
    )
    result = lib.jxlpy_encode_pixels_ex(
        c_pixels,
        arr.nbytes,
        w,
        h,
        channels,
        _DTYPE_TO_NATIVE[arr.dtype],
        int(bits_per_sample),
        c_extras,
        len(c_extras),
        opts,
    )
    return _write_or_return(_consume_result(result).data, output)


def _copy_native_output(pointer, size: int, dtype, shape, out: str):
    if out == "raw":
        return bytes(ffi.buffer(pointer, size)) if size else b""
    if dtype is None:
        raise RuntimeError("native decoder returned an unknown dtype")
    array = np.frombuffer(ffi.buffer(pointer, size), dtype=dtype).reshape(shape).copy()
    if out == "numpy":
        return array
    if out == "torch":
        import torch

        return torch.from_numpy(array)
    raise ValueError("out must be 'numpy', 'torch' or 'raw'")


def _consume_decode_all_result(result, out: str) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    holder = ffi.new("jxlpy_decode_all_result *", result)
    try:
        if not result.ok:
            message = (
                ffi.string(result.error).decode("utf-8", "replace")
                if result.error != ffi.NULL
                else "native call failed"
            )
            raise RuntimeError(message)
        meta = {
            "xsize": int(result.xsize),
            "ysize": int(result.ysize),
            "num_channels": int(result.num_channels),
            "dtype": _NATIVE_TO_DTYPE.get(int(result.dtype)),
            "bits_per_sample": int(result.bits_per_sample),
            "exponent_bits_per_sample": int(result.exponent_bits_per_sample),
            "num_frames": int(result.num_frames),
            "num_frames_known": bool(result.num_frames_known),
            "frame_index": int(result.frame_index),
            "have_animation": bool(result.have_animation),
            "layer_have_crop": bool(result.layer_have_crop),
            "crop_x0": int(result.crop_x0),
            "crop_y0": int(result.crop_y0),
            "layer_xsize": int(result.layer_xsize),
            "layer_ysize": int(result.layer_ysize),
            "duration": int(result.duration),
            "num_extra_channels": int(result.num_extra_channels),
            "color_encoding": _color_encoding_to_dict(result.color_encoding),
            "color_profile_is_icc": bool(result.color_profile_is_icc),
            "icc_profile": (
                bytes(ffi.buffer(result.icc_profile, result.icc_profile_size))
                if result.icc_profile_size
                else None
            ),
            "data_color_encoding": _color_encoding_to_dict(
                result.data_color_encoding
            ),
            "data_color_profile_is_icc": bool(
                result.data_color_profile_is_icc
            ),
            "data_icc_profile": (
                bytes(
                    ffi.buffer(
                        result.data_icc_profile, result.data_icc_profile_size
                    )
                )
                if result.data_icc_profile_size
                else None
            ),
        }
        color_data = _copy_native_output(
            result.color_data,
            int(result.color_size),
            meta["dtype"],
            (meta["ysize"], meta["xsize"], meta["num_channels"]),
            out,
        )
        extras: list[dict[str, Any]] = []
        for i in range(int(result.num_extra_channels)):
            ec = result.extra_channels[i]
            ec_name = (
                ffi.string(ec.extra_channel_name).decode("utf-8", "replace")
                if ec.extra_channel_name != ffi.NULL
                else ""
            )
            ec_dtype = _NATIVE_TO_DTYPE.get(int(ec.dtype))
            ec_data = _copy_native_output(
                ec.data,
                int(ec.size),
                ec_dtype,
                (int(ec.ysize), int(ec.xsize)),
                out,
            )
            extras.append({
                "index": int(ec.extra_channel_index),
                "type": _NATIVE_EXTRA_TYPE.get(
                    int(ec.extra_channel_type), "unknown"
                ),
                "name": ec_name,
                "bits_per_sample": int(ec.bits_per_sample),
                "exponent_bits_per_sample": int(ec.exponent_bits_per_sample),
                "dtype": ec_dtype,
                "xsize": int(ec.xsize),
                "ysize": int(ec.ysize),
                "dim_shift": int(ec.dim_shift),
                "alpha_premultiplied": bool(ec.alpha_premultiplied),
                "spot_color": tuple(float(value) for value in ec.spot_color),
                "cfa_channel": int(ec.cfa_channel),
                "data": ec_data,
            })
        return color_data, meta, extras
    finally:
        lib.jxlpy_free_decode_all_result(holder)


def decode_extra_channel(
    src: Any,
    index: int,
    *,
    frame: int = 0,
    out: str = "numpy",
    coalesced: bool = True,
    return_info: bool = True,
    threads: int = 0,
    scan_all_frames: bool = False,
    max_pixels: int = 0,
    max_output_bytes: int = 0,
):
    """Decode one JPEG XL extra channel as a 2D plane."""
    data = _read_bytes(src)
    if not _is_jxl(data):
        raise ValueError("extra channel decode requires JPEG XL input")
    if threads < 0 or max_pixels < 0 or max_output_bytes < 0:
        raise ValueError("threads and decode limits must be non-negative")
    c_data = ffi.from_buffer(data)
    value, meta = _consume_pixels_result(
        lib.jxlpy_decode_extra_channel_jxl(
            c_data,
            len(data),
            int(frame),
            1 if coalesced else 0,
            int(index),
            0,
            int(threads),
            1 if scan_all_frames else 0,
            int(max_pixels),
            int(max_output_bytes),
        ),
        out,
        plane=True,
        max_pixels=max_pixels,
        max_output_bytes=max_output_bytes,
    )
    return (value, meta) if return_info else value


def decode(
    src: Any,
    *,
    frame: int = 0,
    out: str = "numpy",
    coalesced: bool = True,
    return_info: bool = False,
    return_extra_channels: bool = False,
    include_alpha_extra: bool = False,
    threads: int = 0,
    scan_all_frames: bool = False,
    max_pixels: int = 0,
    max_output_bytes: int = 0,
):
    """Decode JXL, PNG or JPEG bytes/path to numpy, torch or raw bytes."""
    data = _read_bytes(src)
    c_data = ffi.from_buffer(data)
    is_jxl = _is_jxl(data)
    if threads < 0 or max_pixels < 0 or max_output_bytes < 0:
        raise ValueError("threads and decode limits must be non-negative")

    if return_extra_channels and is_jxl:
        all_result = lib.jxlpy_decode_all_jxl(
            c_data,
            len(data),
            int(frame),
            1 if coalesced else 0,
            0,
            0,
            int(threads),
            1 if scan_all_frames else 0,
            int(max_pixels),
            int(max_output_bytes),
        )
        value, meta, extras = _consume_decode_all_result(all_result, out)

        ec_list: list[dict[str, Any]] = []
        for ec in extras:
            if not include_alpha_extra and ec["type"] == "alpha":
                continue
            ec_list.append({
                "index": ec["index"],
                "name": ec["name"],
                "type": ec["type"],
                "bits_per_sample": ec["bits_per_sample"],
                "exponent_bits_per_sample": ec["exponent_bits_per_sample"],
                "dtype": ec["dtype"],
                "xsize": ec["xsize"],
                "ysize": ec["ysize"],
                "dim_shift": ec["dim_shift"],
                "alpha_premultiplied": ec["alpha_premultiplied"],
                "spot_color": ec["spot_color"],
                "cfa_channel": ec["cfa_channel"],
                "data": ec["data"],
            })
        meta["extra_channels"] = ec_list
        return (value, meta) if (return_info or return_extra_channels) else value

    if is_jxl:
        result = lib.jxlpy_decode_jxl(
            c_data,
            len(data),
            int(frame),
            1 if coalesced else 0,
            0,
            0,
            int(threads),
            1 if scan_all_frames else 0,
            int(max_pixels),
            int(max_output_bytes),
        )
    else:
        result = lib.jxlpy_decode_image_bytes(c_data, len(data), int(frame))
    value, meta = _consume_pixels_result(
        result,
        out,
        max_pixels=max_pixels,
        max_output_bytes=max_output_bytes,
    )

    if return_extra_channels:
        meta["extra_channels"] = []

    return (value, meta) if (return_info or return_extra_channels) else value


def decode_layer(
    src: Any,
    *,
    layer: int = 0,
    out: str = "numpy",
    return_info: bool = True,
    return_extra_channels: bool = False,
    include_alpha_extra: bool = False,
    threads: int = 0,
    scan_all_frames: bool = False,
    max_pixels: int = 0,
    max_output_bytes: int = 0,
):
    """Decode a non-coalesced JXL layer/crop instead of the full composed frame."""
    return decode(
        src,
        frame=layer,
        out=out,
        coalesced=False,
        return_info=return_info,
        return_extra_channels=return_extra_channels,
        include_alpha_extra=include_alpha_extra,
        threads=threads,
        scan_all_frames=scan_all_frames,
        max_pixels=max_pixels,
        max_output_bytes=max_output_bytes,
    )


def _frame_arrays(frames: Iterable[Any], *, layout: str) -> list[np.ndarray]:
    arrays = []
    for frame in frames:
        if isinstance(frame, (str, Path, bytes, bytearray, memoryview)):
            frame = decode(frame, out="numpy")
        arrays.append(_as_array(frame, layout=layout))
    if not arrays:
        raise ValueError("frames must not be empty")
    first_shape = arrays[0].shape
    first_dtype = arrays[0].dtype
    for arr in arrays:
        if arr.shape != first_shape:
            raise ValueError("all frames must have the same shape")
        if arr.dtype != first_dtype:
            raise ValueError("all frames must have the same dtype")
    return arrays


def _durations(value: int | Iterable[int] | None, count: int) -> list[int]:
    if value is None:
        out = [0] * count
    elif isinstance(value, int):
        out = [value] * count
    else:
        out = [int(v) for v in value]
    if len(out) != count:
        raise ValueError("durations length must match frame count")
    if any(duration < 0 or duration > 0xFFFFFFFF for duration in out):
        raise ValueError("durations must be unsigned 32-bit tick counts")
    return out


def _as_float_samples(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / np.float32(255.0)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / np.float32(65535.0)
    return arr.astype(np.float32, copy=False)


def _additive_payloads(arrays: list[np.ndarray]) -> list[np.ndarray]:
    normalized = [_as_float_samples(arr) for arr in arrays]
    payloads = [normalized[0]]
    for i in range(1, len(normalized)):
        payloads.append(normalized[i] - normalized[i - 1])
    return [np.ascontiguousarray(payload, dtype=np.float32) for payload in payloads]


def _additive_extra_specs(extra_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for spec in extra_specs:
        next_spec = dict(spec)
        next_spec["arrays"] = _additive_payloads(spec["arrays"])
        next_spec["bits_per_sample"] = 0
        next_spec["exponent_bits_per_sample"] = 0
        out.append(next_spec)
    return out


def _integer_mask_value(dtype: np.dtype, bits_per_sample: int) -> int:
    dtype = np.dtype(dtype)
    if dtype == np.dtype("uint8"):
        bits = bits_per_sample if bits_per_sample else 8
    elif dtype == np.dtype("uint16"):
        bits = bits_per_sample if bits_per_sample else 16
    else:
        raise TypeError("blend_mask reference mode requires uint8 or uint16 frames")
    if bits <= 0 or bits > dtype.itemsize * 8:
        raise ValueError("bits_per_sample is incompatible with blend_mask dtype")
    return (1 << bits) - 1


def _masked_payload(arr: np.ndarray, changed: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr)
    out[changed] = arr[changed]
    return np.ascontiguousarray(out)


def _masked_plane_payload(arr: np.ndarray, changed: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr)
    out[changed] = arr[changed]
    return np.ascontiguousarray(out)


def encode_multiframe(
    frames: Iterable[Any],
    output: str | Path | None = None,
    *,
    layout: str = "auto",
    extra_channels: Iterable[Any] | None = None,
    durations: int | Iterable[int] | None = 1,
    color_encoding: str | Mapping[str, Any] | None = None,
    icc_profile: Any = None,
    tps: tuple[int, int] = (1000, 1),
    reference: str = "auto",
    min_crop_ratio: float = 0.98,
    bits_per_sample: int = 0,
    lossless: bool | None = None,
    distance: float | None = None,
    alpha_distance: float = 0.0,
    effort: int = 7,
    modular: int | None = None,
    level: int = -1,
    threads: int = 0,
    use_container: bool = False,
    allow_expert_options: bool = False,
    compress_boxes: bool = True,
    brotli_effort: int | None = None,
    keep_invisible: bool | None = None,
    patches: bool | None = None,
    dots: bool | None = None,
    noise: bool | None = None,
    gaborish: bool | None = None,
    group_order: int | None = None,
    center_x: int | None = None,
    center_y: int | None = None,
    progressive: bool = False,
    progressive_ac: bool | None = None,
    qprogressive_ac: bool | None = None,
    progressive_dc: int | None = None,
    responsive: bool | None = None,
    epf: int | None = None,
    faster_decoding: int | None = None,
    resampling: int | None = None,
    ec_resampling: int | None = None,
    already_downsampled: bool | None = None,
    upsampling_mode: int | None = None,
    photon_noise_iso: float = 0.0,
    intensity_target: float = 0.0,
    premultiply: int | None = None,
    override_bitdepth: int = 0,
    buffering: int | None = None,
    disable_perceptual_optimizations: bool = False,
    modular_group_size: int | None = None,
    modular_predictor: int | None = None,
    modular_colorspace: int | None = None,
    modular_ma_tree_learning_percent: float | None = None,
    iterations: float | None = None,
    modular_nb_prev_channels: int | None = None,
    modular_palette_colors: int | None = None,
    modular_lossy_palette: bool | None = None,
    modular_channel_colors_global_percent: float | None = None,
    modular_channel_colors_group_percent: float | None = None,
    pre_compact: float | None = None,
    post_compact: float | None = None,
    encoder_options: Mapping[str, Any] | None = None,
    frame_settings: Any = None,
):
    """Encode multiple frames, with optional exact REPLACE+crop delta frames."""
    if reference not in (
        "auto",
        "first",
        "previous",
        "none",
        "full",
        "add",
        "additive",
        "blend_mask",
        "mask",
        "masked",
        "blend_mask8",
        "mask8",
        "masked8",
    ):
        raise ValueError(
            "reference must be 'auto', 'first', 'previous', 'none', 'full', "
            "'add' or 'blend_mask'"
        )
    additive = reference in ("add", "additive")
    blend_mask_8bit = reference in ("blend_mask8", "mask8", "masked8")
    blend_mask = reference in ("blend_mask", "mask", "masked") or blend_mask_8bit
    arrays = _frame_arrays(frames, layout=layout)
    durs = _durations(durations, len(arrays))
    h, w, channels = arrays[0].shape
    dtype_id = _DTYPE_TO_NATIVE[arrays[0].dtype]
    native_bits_per_sample = int(bits_per_sample)
    full_area = w * h
    extra_specs = _extra_specs_to_frame_arrays(
        extra_channels,
        frame_count=len(arrays),
        expected_hw=(h, w),
        layout=layout,
    )
    reference_extra_specs = extra_specs
    if additive:
        arrays = _additive_payloads(arrays)
        extra_specs = _additive_extra_specs(extra_specs)
        dtype_id = lib.JXLPY_DTYPE_FLOAT32
        native_bits_per_sample = 0
        if modular is None:
            modular = 1
    mask_value = 0
    mask_alpha_index = 0
    if blend_mask:
        if additive:
            raise ValueError("blend_mask and add reference modes are mutually exclusive")
        _integer_mask_value(arrays[0].dtype, native_bits_per_sample)
        mask_dtype = np.dtype(arrays[0].dtype) if blend_mask_8bit else np.dtype("uint8")
        mask_bits_per_sample = (
            native_bits_per_sample
            if blend_mask_8bit and native_bits_per_sample
            else (arrays[0].dtype.itemsize * 8 if blend_mask_8bit else 1)
        )
        mask_value = (
            _integer_mask_value(arrays[0].dtype, native_bits_per_sample)
            if blend_mask_8bit
            else np.iinfo(mask_dtype).max
        )
        mask_alpha_index = (1 if channels in (2, 4) else 0) + len(extra_specs)
        mask_arrays = [
            np.zeros((h, w), dtype=mask_dtype) for _ in range(len(arrays))
        ]
        extra_specs = [
            *extra_specs,
            {
                "name": "jxlpy_blend_mask",
                "type_id": _EXTRA_TYPE_TO_NATIVE["selection_mask"],
                "bits_per_sample": mask_bits_per_sample,
                "arrays": mask_arrays,
            },
        ]
        if modular is None:
            modular = 1

    option_kwargs = dict(
        lossless=lossless,
        distance=distance,
        alpha_distance=alpha_distance,
        effort=effort,
        modular=modular,
        level=level,
        threads=threads,
        use_container=use_container,
        jpeg_store_metadata=False,
        allow_expert_options=allow_expert_options,
        compress_boxes=compress_boxes,
        brotli_effort=brotli_effort,
        keep_invisible=keep_invisible,
        patches=patches,
        dots=dots,
        noise=noise,
        gaborish=gaborish,
        group_order=group_order,
        center_x=center_x,
        center_y=center_y,
        progressive=progressive,
        progressive_ac=progressive_ac,
        qprogressive_ac=qprogressive_ac,
        progressive_dc=progressive_dc,
        responsive=responsive,
        epf=epf,
        faster_decoding=faster_decoding,
        resampling=resampling,
        ec_resampling=ec_resampling,
        already_downsampled=already_downsampled,
        upsampling_mode=upsampling_mode,
        photon_noise_iso=photon_noise_iso,
        intensity_target=intensity_target,
        premultiply=premultiply,
        override_bitdepth=override_bitdepth,
        buffering=buffering,
        jpeg_reconstruction_cfl=None,
        disable_perceptual_optimizations=disable_perceptual_optimizations,
        modular_group_size=modular_group_size,
        modular_predictor=modular_predictor,
        modular_colorspace=modular_colorspace,
        modular_ma_tree_learning_percent=modular_ma_tree_learning_percent,
        iterations=iterations,
        modular_nb_prev_channels=modular_nb_prev_channels,
        modular_palette_colors=modular_palette_colors,
        modular_lossy_palette=modular_lossy_palette,
        modular_channel_colors_global_percent=modular_channel_colors_global_percent,
        modular_channel_colors_group_percent=modular_channel_colors_group_percent,
        pre_compact=pre_compact,
        post_compact=post_compact,
        color_encoding=color_encoding,
        icc_profile=icc_profile,
        tps=tps,
    )
    merged_frame_settings = _merge_encoder_options(
        option_kwargs, encoder_options, frame_settings
    )
    _ensure_dim_shift_resampling(option_kwargs, extra_specs, merged_frame_settings)
    option_keepalive: list[Any] = []
    opts = _options(
        **option_kwargs,
        frame_settings=merged_frame_settings,
        _keepalive=option_keepalive,
    )

    c_frames = ffi.new("jxlpy_frame[]", len(arrays))
    c_extras = ffi.new("jxlpy_extra_channel[]", len(arrays) * len(extra_specs))
    buffers = []
    c_buffers = []
    extra_buffers = []
    extra_c_buffers = []
    extra_name_buffers = []
    refs: dict[int, tuple[np.ndarray, list[np.ndarray]]] = {}

    for i, arr in enumerate(arrays):
        reference_full_extras = [
            spec["arrays"][i] for spec in reference_extra_specs
        ]
        if blend_mask:
            full_extras = reference_full_extras + [
                np.full((h, w), mask_value, dtype=mask_dtype)
                if i == 0
                else np.zeros((h, w), dtype=mask_dtype)
            ]
        else:
            full_extras = [spec["arrays"][i] for spec in extra_specs]
        have_crop = False
        x0 = y0 = 0
        crop = arr
        crop_extras = full_extras
        source_ref = 0
        save_ref = 0

        if i == 0:
            if (additive or blend_mask) and len(arrays) > 1:
                save_ref = 1
            elif reference in ("auto", "first") and len(arrays) > 1:
                save_ref = 2
            elif reference == "previous" and len(arrays) > 1:
                save_ref = 1
        elif reference in ("none", "full"):
            save_ref = 0
        elif additive:
            source_ref = _pack_source_ref(1, blend_mode="add")
            save_ref = 1 if i + 1 < len(arrays) else 0
        elif blend_mask:
            ref_main, ref_extras = refs[1]
            changed = _changed_mask(
                arr, ref_main, reference_full_extras, ref_extras
            )
            diff = _diff_stats_from_mask(changed)
            x0, y0, x1, y1 = diff.bbox
            source_ref = _pack_source_ref(
                1, blend_mode="blend", alpha=mask_alpha_index
            )
            save_ref = 1 if i + 1 < len(arrays) else 0
            if diff.bbox_area < full_area * float(min_crop_ratio):
                have_crop = True
                changed_crop = changed[y0:y1, x0:x1]
                source_crop = arr[y0:y1, x0:x1, :]
                crop = _masked_payload(source_crop, changed_crop)
                crop_extras = [
                    _masked_plane_payload(extra[y0:y1, x0:x1], changed_crop)
                    for extra in reference_full_extras
                ]
            else:
                changed_crop = changed
                crop = _masked_payload(arr, changed_crop)
                crop_extras = [
                    _masked_plane_payload(extra, changed_crop)
                    for extra in reference_full_extras
                ]
            mask_payload = np.zeros(changed_crop.shape, dtype=mask_dtype)
            mask_payload[changed_crop] = mask_value
            crop_extras.append(np.ascontiguousarray(mask_payload))
        else:
            source_ref, diff, _ = _select_reference_bbox(
                index=i,
                current=arr,
                arrays=arrays,
                current_extras=reference_full_extras,
                extra_specs=reference_extra_specs,
                refs=refs,
                reference=reference,
            )
            bbox = diff.bbox
            x0, y0, x1, y1 = bbox
            if diff.bbox_area < full_area * float(min_crop_ratio):
                have_crop = True
                crop = np.ascontiguousarray(arr[y0:y1, x0:x1, :])
                crop_extras = [
                    np.ascontiguousarray(extra[y0:y1, x0:x1])
                    for extra in full_extras
                ]
            else:
                source_ref = 0
                crop = arr
                crop_extras = full_extras
            if reference in ("previous", "auto"):
                save_ref = 1

        crop_buffer = np.ascontiguousarray(crop)
        buffers.append(crop_buffer)
        c_buf = ffi.from_buffer(crop_buffer)
        c_buffers.append(c_buf)
        c_frames[i].pixels = c_buf
        c_frames[i].size = crop_buffer.nbytes
        c_frames[i].xsize = crop.shape[1]
        c_frames[i].ysize = crop.shape[0]
        c_frames[i].have_crop = 1 if have_crop else 0
        c_frames[i].crop_x0 = x0
        c_frames[i].crop_y0 = y0
        c_frames[i].duration = durs[i]
        c_frames[i].source_ref = source_ref
        c_frames[i].save_as_ref = save_ref

        if save_ref:
            refs[save_ref] = (
                arr,
                reference_full_extras if blend_mask else full_extras,
            )

        for extra_i, (spec, extra_arr) in enumerate(zip(extra_specs, crop_extras)):
            flat_i = i * len(extra_specs) + extra_i
            extra_buffer = np.ascontiguousarray(extra_arr)
            extra_buffers.append(extra_buffer)
            c_extra_buf = ffi.from_buffer(extra_buffer)
            extra_c_buffers.append(c_extra_buf)
            name_bytes = spec["name"].encode("utf-8")
            c_name = ffi.new("char[]", name_bytes) if name_bytes else ffi.NULL
            extra_name_buffers.append(c_name)
            c_extras[flat_i].pixels = c_extra_buf
            c_extras[flat_i].size = extra_buffer.nbytes
            c_extras[flat_i].xsize = extra_arr.shape[1]
            c_extras[flat_i].ysize = extra_arr.shape[0]
            c_extras[flat_i].dtype = _DTYPE_TO_NATIVE[extra_arr.dtype]
            c_extras[flat_i].bits_per_sample = int(spec["bits_per_sample"])
            c_extras[flat_i].exponent_bits_per_sample = int(
                spec.get("exponent_bits_per_sample", 0)
            )
            c_extras[flat_i].type = int(spec["type_id"])
            c_extras[flat_i].name = c_name
            c_extras[flat_i].name_size = len(name_bytes)
            c_extras[flat_i].dim_shift = int(spec.get("dim_shift", 0))
            c_extras[flat_i].alpha_premultiplied = (
                1 if spec.get("alpha_premultiplied", False) else 0
            )
            for channel, value in enumerate(spec.get("spot_color", (0, 0, 0, 0))):
                c_extras[flat_i].spot_color[channel] = float(value)
            c_extras[flat_i].cfa_channel = int(spec.get("cfa_channel", 0))

    result = lib.jxlpy_encode_multiframe_ex(
        c_frames,
        len(arrays),
        w,
        h,
        channels,
        dtype_id,
        native_bits_per_sample,
        c_extras,
        len(extra_specs),
        opts,
    )
    return _write_or_return(_consume_result(result).data, output)


def info(src: Any) -> dict[str, Any]:
    """Return metadata dict for a JXL, PNG or JPEG file/bytes."""
    data = _read_bytes(src)
    c_data = ffi.from_buffer(data)
    if _is_jxl(data):
        _, meta, extras = _consume_decode_all_result(
            lib.jxlpy_info_all(c_data, len(data)), "raw"
        )
        meta["extra_channels"] = [
            {key: value for key, value in channel.items() if key != "data"}
            for channel in extras
        ]
        return meta
    native = _consume_result(lib.jxlpy_decode_image_bytes(c_data, len(data), 0))
    native.meta["extra_channels"] = []
    return native.meta


def convert(
    src: Any,
    output: str | Path | None = None,
    *,
    format: str = "png",
    quality: int = -1,
) -> bytes | Path:
    """Decode any supported input and re-encode to a target format.

    Supported output formats: png, jpeg/jpg, ppm, pgm, pam, pfm, pgx.
    Quality (0-100) only applies to JPEG output.
    """
    data = _read_bytes(src)
    ext = format.lower()
    if not ext.startswith("."):
        ext = "." + ext
    c_data = ffi.from_buffer(data)
    c_ext = ffi.new("char[]", ext.encode("utf-8"))
    result = lib.jxlpy_decode_to_format(c_data, len(data), c_ext, int(quality))
    return _write_or_return(_consume_result(result).data, output)


def decode_to_png(src: Any, output: str | Path | None = None) -> bytes | Path:
    """Decode JXL/PNG/JPEG to PNG bytes or write to file."""
    return convert(src, output, format="png")


def reconstruct_jpeg(src: Any, output: str | Path | None = None) -> bytes | Path:
    """Extract the original JPEG bitstream from a JXL file (lossless transcode).

    Raises RuntimeError if the JXL does not contain JPEG reconstruction data.
    """
    data = _read_bytes(src)
    c_data = ffi.from_buffer(data)
    result = lib.jxlpy_reconstruct_jpeg(c_data, len(data))
    return _write_or_return(_consume_result(result).data, output)


def decode_to_jpeg(
    src: Any, output: str | Path | None = None, *, quality: int = 95
) -> bytes | Path:
    """Decode to JPEG. Tries JPEG reconstruction first (bit-exact); falls back to re-encode."""
    data = _read_bytes(src)
    if _is_jxl(data):
        try:
            return _write_or_return(
                _consume_result(
                    lib.jxlpy_reconstruct_jpeg(ffi.from_buffer(data), len(data))
                ).data,
                output,
            )
        except RuntimeError:
            pass
    return convert(data, output, format="jpg", quality=quality)


def analyze_multiframe(
    frames: Iterable[Any],
    *,
    layout: str = "auto",
    extra_channels: Iterable[Any] | None = None,
    reference: str = "auto",
    min_crop_ratio: float = 0.98,
) -> dict[str, Any]:
    """Analyze frames using the same exact reference/crop policy as encoding."""
    arrays = _frame_arrays(frames, layout=layout)
    h, w, _ = arrays[0].shape
    extra_specs = _extra_specs_to_frame_arrays(
        extra_channels,
        frame_count=len(arrays),
        expected_hw=(h, w),
        layout=layout,
    )
    return _analyze_frame_arrays(
        arrays,
        extra_specs=extra_specs,
        reference=reference,
        min_crop_ratio=min_crop_ratio,
    )
