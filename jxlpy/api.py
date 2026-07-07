from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ._ffi import ffi, lib


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

_EXTRA_TYPE_TO_NATIVE = {
    "alpha": 0,
    "depth": 1,
    "spot_color": 2,
    "selection_mask": 3,
    "black": 4,
    "cfa": 5,
    "thermal": 6,
    "unknown": 15,
    "optional": 16,
}

_NATIVE_EXTRA_TYPE = {value: key for key, value in _EXTRA_TYPE_TO_NATIVE.items()}


@dataclass(frozen=True)
class _NativeResult:
    data: bytes
    meta: dict[str, Any]


def _read_bytes(src: Any) -> bytes:
    if isinstance(src, Path):
        return src.read_bytes()
    if isinstance(src, str):
        return Path(src).read_bytes()
    if isinstance(src, bytes):
        return src
    if isinstance(src, bytearray):
        return bytes(src)
    if isinstance(src, memoryview):
        return src.tobytes()
    raise TypeError("expected a path or bytes-like object")


def _is_jxl(data: bytes) -> bool:
    return data.startswith(b"\xff\x0a") or data.startswith(
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


def _extra_type_id(value: Any) -> int:
    if value is None:
        return _EXTRA_TYPE_TO_NATIVE["unknown"]
    if isinstance(value, str):
        key = value.lower().replace("-", "_")
        if key not in _EXTRA_TYPE_TO_NATIVE:
            raise ValueError(f"unknown extra channel type: {value!r}")
        return _EXTRA_TYPE_TO_NATIVE[key]
    return int(value)


def _optional_int(value: int | None) -> int:
    return -1 if value is None else int(value)


def _optional_float(value: float | None) -> float:
    return -1.0 if value is None else float(value)


def _optional_bool(value: bool | None) -> int:
    return -1 if value is None else (1 if value else 0)


def _parse_extra_spec(spec: Any) -> tuple[str, int, int, Any]:
    name = ""
    type_id = _EXTRA_TYPE_TO_NATIVE["unknown"]
    bits_per_sample = 0
    data = spec

    if isinstance(spec, dict):
        data = spec["data"]
        name = str(spec.get("name", ""))
        type_id = _extra_type_id(spec.get("type", "unknown"))
        bits_per_sample = int(spec.get("bits_per_sample", 0))
    elif isinstance(spec, tuple):
        if len(spec) == 2:
            name = str(spec[0])
            data = spec[1]
        elif len(spec) == 3:
            name = str(spec[0])
            type_id = _extra_type_id(spec[1])
            data = spec[2]
        else:
            raise ValueError("extra channel tuple must be (name, data) or (name, type, data)")

    return name, type_id, bits_per_sample, data


def _make_extra_structs(
    specs: Iterable[Any] | None,
    *,
    expected_hw: tuple[int, int],
    layout: str,
):
    specs = list(specs or [])
    c_extras = ffi.new("jxlpy_extra_channel[]", len(specs))
    buffers: list[bytes] = []
    c_buffers = []
    name_buffers = []

    for i, spec in enumerate(specs):
        name, type_id, bits_per_sample, data = _parse_extra_spec(spec)
        arr = _as_extra_array(data, layout=layout)
        if arr.shape != expected_hw:
            raise ValueError("extra channel dimensions must match the main image")
        raw = arr.tobytes()
        buffers.append(raw)
        c_buf = ffi.from_buffer(raw)
        c_buffers.append(c_buf)
        name_bytes = name.encode("utf-8")
        c_name = ffi.new("char[]", name_bytes) if name_bytes else ffi.NULL
        name_buffers.append(c_name)
        c_extras[i].pixels = c_buf
        c_extras[i].size = len(raw)
        c_extras[i].xsize = arr.shape[1]
        c_extras[i].ysize = arr.shape[0]
        c_extras[i].dtype = _DTYPE_TO_NATIVE[arr.dtype]
        c_extras[i].bits_per_sample = bits_per_sample
        c_extras[i].type = type_id
        c_extras[i].name = c_name
        c_extras[i].name_size = len(name_bytes)

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
        name, type_id, bits_per_sample, data = _parse_extra_spec(spec)
        if _looks_like_frame_sequence(data, frame_count):
            arrays = [_as_extra_array(item, layout=layout) for item in data]
        else:
            one = _as_extra_array(data, layout=layout)
            arrays = [one for _ in range(frame_count)]
        for arr in arrays:
            if arr.shape != expected_hw:
                raise ValueError("extra channel dimensions must match every frame")
        channels.append(
            {
                "name": name,
                "type_id": type_id,
                "bits_per_sample": bits_per_sample,
                "arrays": arrays,
            }
        )
    return channels


def _bbox_changed_with_extras(
    current: np.ndarray,
    reference: np.ndarray,
    current_extras: list[np.ndarray],
    reference_extras: list[np.ndarray],
):
    changed = np.any(current != reference, axis=2)
    for cur_extra, ref_extra in zip(current_extras, reference_extras):
        changed |= cur_extra != ref_extra
    if not np.any(changed):
        return 0, 0, 1, 1
    ys, xs = np.nonzero(changed)
    x0 = int(xs.min())
    y0 = int(ys.min())
    x1 = int(xs.max()) + 1
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


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
    modular_nb_prev_channels: int | None = None,
    modular_palette_colors: int | None = None,
    modular_lossy_palette: bool | None = None,
    modular_channel_colors_global_percent: float | None = None,
    modular_channel_colors_group_percent: float | None = None,
    tps: tuple[int, int] = (1000, 1),
):
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
    return opts


def _meta_from_result(result) -> dict[str, Any]:
    extra_name = (
        ffi.string(result.extra_channel_name).decode("utf-8", "replace")
        if result.extra_channel_name != ffi.NULL
        else ""
    )
    return {
        "xsize": int(result.xsize),
        "ysize": int(result.ysize),
        "num_channels": int(result.num_channels),
        "dtype": _NATIVE_TO_DTYPE.get(int(result.dtype)),
        "bits_per_sample": int(result.bits_per_sample),
        "exponent_bits_per_sample": int(result.exponent_bits_per_sample),
        "num_frames": int(result.num_frames),
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
    modular_nb_prev_channels: int | None = None,
    modular_palette_colors: int | None = None,
    modular_lossy_palette: bool | None = None,
    modular_channel_colors_global_percent: float | None = None,
    modular_channel_colors_group_percent: float | None = None,
):
    """Encode a path, encoded image bytes, numpy array or torch tensor to JXL."""
    opts = _options(
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
        modular_nb_prev_channels=modular_nb_prev_channels,
        modular_palette_colors=modular_palette_colors,
        modular_lossy_palette=modular_lossy_palette,
        modular_channel_colors_global_percent=modular_channel_colors_global_percent,
        modular_channel_colors_group_percent=modular_channel_colors_group_percent,
    )

    if isinstance(src, (str, Path, bytes, bytearray, memoryview)):
        if extra_channels is not None and list(extra_channels):
            raise ValueError("extra_channels are only supported for array/tensor input")
        data = _read_bytes(src)
        c_data = ffi.from_buffer(data)
        result = lib.jxlpy_encode_image_bytes(c_data, len(data), opts)
        return _write_or_return(_consume_result(result).data, output)

    arr = _as_array(src, layout=layout)
    h, w, channels = arr.shape
    c_pixels = ffi.from_buffer(arr)
    c_extras, extra_buffers, extra_c_buffers, extra_names = _make_extra_structs(
        extra_channels,
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


def _decode_to_array(native: _NativeResult) -> np.ndarray:
    dtype = native.meta["dtype"]
    if dtype is None:
        raise RuntimeError("native decoder returned an unknown dtype")
    arr = np.frombuffer(native.data, dtype=dtype)
    return arr.reshape(
        native.meta["ysize"], native.meta["xsize"], native.meta["num_channels"]
    )


def _plane_to_output(native: _NativeResult, out: str):
    arr = _decode_to_array(native)[:, :, 0]
    if out == "numpy":
        return arr.copy()
    if out == "torch":
        import torch

        return torch.from_numpy(arr.copy())
    if out == "raw":
        return native.data
    raise ValueError("out must be 'numpy', 'torch' or 'raw'")


def _consume_decode_all_result(result) -> tuple[bytes, dict[str, Any], list[dict[str, Any]]]:
    holder = ffi.new("jxlpy_decode_all_result *", result)
    try:
        if not result.ok:
            message = (
                ffi.string(result.error).decode("utf-8", "replace")
                if result.error != ffi.NULL
                else "native call failed"
            )
            raise RuntimeError(message)
        color_data = (
            bytes(ffi.buffer(result.color_data, result.color_size))
            if result.color_size
            else b""
        )
        meta = {
            "xsize": int(result.xsize),
            "ysize": int(result.ysize),
            "num_channels": int(result.num_channels),
            "dtype": _NATIVE_TO_DTYPE.get(int(result.dtype)),
            "bits_per_sample": int(result.bits_per_sample),
            "exponent_bits_per_sample": int(result.exponent_bits_per_sample),
            "num_frames": int(result.num_frames),
            "frame_index": int(result.frame_index),
            "have_animation": bool(result.have_animation),
            "layer_have_crop": bool(result.layer_have_crop),
            "crop_x0": int(result.crop_x0),
            "crop_y0": int(result.crop_y0),
            "layer_xsize": int(result.layer_xsize),
            "layer_ysize": int(result.layer_ysize),
            "duration": int(result.duration),
            "num_extra_channels": int(result.num_extra_channels),
        }
        extras: list[dict[str, Any]] = []
        for i in range(int(result.num_extra_channels)):
            ec = result.extra_channels[i]
            ec_name = (
                ffi.string(ec.extra_channel_name).decode("utf-8", "replace")
                if ec.extra_channel_name != ffi.NULL
                else ""
            )
            ec_data = (
                bytes(ffi.buffer(ec.data, ec.size)) if ec.size else b""
            )
            extras.append({
                "index": int(ec.extra_channel_index),
                "type": _NATIVE_EXTRA_TYPE.get(
                    int(ec.extra_channel_type), "unknown"
                ),
                "name": ec_name,
                "bits_per_sample": int(ec.bits_per_sample),
                "exponent_bits_per_sample": int(ec.exponent_bits_per_sample),
                "dtype": _NATIVE_TO_DTYPE.get(int(ec.dtype)),
                "data": ec_data,
            })
        return color_data, meta, extras
    finally:
        lib.jxlpy_free_decode_all_result(holder)


def _ec_data_to_output(ec_data: bytes, ec_meta: dict[str, Any], out: str):
    dtype = ec_meta["dtype"]
    if dtype is None:
        raise RuntimeError("native decoder returned an unknown dtype for extra channel")
    arr = np.frombuffer(ec_data, dtype=dtype)
    arr = arr.reshape(ec_meta["ysize"], ec_meta["xsize"])
    if out == "numpy":
        return arr.copy()
    if out == "torch":
        import torch

        return torch.from_numpy(arr.copy())
    if out == "raw":
        return ec_data
    raise ValueError("out must be 'numpy', 'torch' or 'raw'")


def decode_extra_channel(
    src: Any,
    index: int,
    *,
    frame: int = 0,
    out: str = "numpy",
    coalesced: bool = True,
    return_info: bool = True,
):
    """Decode one JPEG XL extra channel as a 2D plane."""
    data = _read_bytes(src)
    if not _is_jxl(data):
        raise ValueError("extra channel decode requires JPEG XL input")
    c_data = ffi.from_buffer(data)
    native = _consume_result(
        lib.jxlpy_decode_extra_channel_jxl(
            c_data, len(data), int(frame), 1 if coalesced else 0, int(index), 0
        )
    )
    value = _plane_to_output(native, out)
    return (value, native.meta) if return_info else value


def decode(
    src: Any,
    *,
    frame: int = 0,
    out: str = "numpy",
    coalesced: bool = True,
    return_info: bool = False,
    return_extra_channels: bool = False,
    include_alpha_extra: bool = False,
):
    """Decode JXL, PNG or JPEG bytes/path to numpy, torch or raw bytes."""
    data = _read_bytes(src)
    c_data = ffi.from_buffer(data)
    is_jxl = _is_jxl(data)

    if return_extra_channels and is_jxl:
        all_result = lib.jxlpy_decode_all_jxl(
            c_data, len(data), int(frame), 1 if coalesced else 0, 0, 0
        )
        color_data, meta, extras = _consume_decode_all_result(all_result)

        if out == "raw":
            value: Any = color_data
        else:
            dtype = meta["dtype"]
            if dtype is None:
                raise RuntimeError("native decoder returned an unknown dtype")
            arr = np.frombuffer(color_data, dtype=dtype)
            arr = arr.reshape(meta["ysize"], meta["xsize"], meta["num_channels"])
            if out == "numpy":
                value = arr.copy()
            elif out == "torch":
                import torch

                value = torch.from_numpy(arr.copy())
            else:
                raise ValueError("out must be 'numpy', 'torch' or 'raw'")

        ec_list: list[dict[str, Any]] = []
        for ec in extras:
            if not include_alpha_extra and ec["type"] == "alpha":
                continue
            ec_meta = {**ec, "xsize": meta["xsize"], "ysize": meta["ysize"]}
            ec_out = _ec_data_to_output(ec["data"], ec_meta, out if out in ("numpy", "torch") else "raw")
            ec_list.append({
                "index": ec["index"],
                "name": ec["name"],
                "type": ec["type"],
                "bits_per_sample": ec["bits_per_sample"],
                "dtype": ec["dtype"],
                "data": ec_out,
            })
        meta["extra_channels"] = ec_list
        return (value, meta) if (return_info or return_extra_channels) else value

    if is_jxl:
        result = lib.jxlpy_decode_jxl(
            c_data, len(data), int(frame), 1 if coalesced else 0, 0, 0
        )
    else:
        result = lib.jxlpy_decode_image_bytes(c_data, len(data), int(frame))
    native = _consume_result(result)

    if out == "raw":
        value = native.data
    else:
        arr = _decode_to_array(native)
        if out == "numpy":
            value = arr.copy()
        elif out == "torch":
            import torch

            value = torch.from_numpy(arr.copy())
        else:
            raise ValueError("out must be 'numpy', 'torch' or 'raw'")

    if return_extra_channels:
        native.meta["extra_channels"] = []

    return (value, native.meta) if (return_info or return_extra_channels) else value


def decode_layer(
    src: Any,
    *,
    layer: int = 0,
    out: str = "numpy",
    return_info: bool = True,
    return_extra_channels: bool = False,
    include_alpha_extra: bool = False,
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


def _durations(value: int | Iterable[int], count: int) -> list[int]:
    if isinstance(value, int):
        return [value] * count
    out = [int(v) for v in value]
    if len(out) != count:
        raise ValueError("durations length must match frame count")
    return out


def encode_multiframe(
    frames: Iterable[Any],
    output: str | Path | None = None,
    *,
    layout: str = "auto",
    extra_channels: Iterable[Any] | None = None,
    durations: int | Iterable[int] = 1,
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
    modular_nb_prev_channels: int | None = None,
    modular_palette_colors: int | None = None,
    modular_lossy_palette: bool | None = None,
    modular_channel_colors_global_percent: float | None = None,
    modular_channel_colors_group_percent: float | None = None,
):
    """Encode multiple frames, with optional exact REPLACE+crop delta frames."""
    arrays = _frame_arrays(frames, layout=layout)
    durs = _durations(durations, len(arrays))
    h, w, channels = arrays[0].shape
    dtype_id = _DTYPE_TO_NATIVE[arrays[0].dtype]
    full_area = w * h
    extra_specs = _extra_specs_to_frame_arrays(
        extra_channels,
        frame_count=len(arrays),
        expected_hw=(h, w),
        layout=layout,
    )

    opts = _options(
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
        disable_perceptual_optimizations=disable_perceptual_optimizations,
        modular_group_size=modular_group_size,
        modular_predictor=modular_predictor,
        modular_colorspace=modular_colorspace,
        modular_ma_tree_learning_percent=modular_ma_tree_learning_percent,
        modular_nb_prev_channels=modular_nb_prev_channels,
        modular_palette_colors=modular_palette_colors,
        modular_lossy_palette=modular_lossy_palette,
        modular_channel_colors_global_percent=modular_channel_colors_global_percent,
        modular_channel_colors_group_percent=modular_channel_colors_group_percent,
        tps=tps,
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
        full_extras = [spec["arrays"][i] for spec in extra_specs]
        have_crop = False
        x0 = y0 = 0
        crop = arr
        crop_extras = full_extras
        source_ref = 0
        save_ref = 0

        if i == 0:
            if reference in ("auto", "first") and len(arrays) > 1:
                save_ref = 2
            elif reference == "previous" and len(arrays) > 1:
                save_ref = 1
        elif reference in ("none", "full"):
            save_ref = 0
        else:
            candidates = []
            if reference in ("previous", "auto") and 1 in refs:
                ref_main, ref_extras = refs[1]
                candidates.append(
                    (
                        1,
                        _bbox_changed_with_extras(
                            arr, ref_main, full_extras, ref_extras
                        ),
                    )
                )
            if reference in ("first", "auto") and 2 in refs:
                ref_main, ref_extras = refs[2]
                candidates.append(
                    (
                        2,
                        _bbox_changed_with_extras(
                            arr, ref_main, full_extras, ref_extras
                        ),
                    )
                )
            if not candidates:
                previous_extras = [spec["arrays"][i - 1] for spec in extra_specs]
                candidates.append(
                    (
                        1,
                        _bbox_changed_with_extras(
                            arr, arrays[i - 1], full_extras, previous_extras
                        ),
                    )
                )

            source_ref, bbox = min(
                candidates,
                key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]),
            )
            x0, y0, x1, y1 = bbox
            crop_area = (x1 - x0) * (y1 - y0)
            if crop_area < full_area * float(min_crop_ratio):
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

        raw = np.ascontiguousarray(crop).tobytes()
        buffers.append(raw)
        c_buf = ffi.from_buffer(raw)
        c_buffers.append(c_buf)
        c_frames[i].pixels = c_buf
        c_frames[i].size = len(raw)
        c_frames[i].xsize = crop.shape[1]
        c_frames[i].ysize = crop.shape[0]
        c_frames[i].have_crop = 1 if have_crop else 0
        c_frames[i].crop_x0 = x0
        c_frames[i].crop_y0 = y0
        c_frames[i].duration = durs[i]
        c_frames[i].source_ref = source_ref
        c_frames[i].save_as_ref = save_ref

        if save_ref:
            refs[save_ref] = (arr, full_extras)

        for extra_i, (spec, extra_arr) in enumerate(zip(extra_specs, crop_extras)):
            flat_i = i * len(extra_specs) + extra_i
            raw_extra = np.ascontiguousarray(extra_arr).tobytes()
            extra_buffers.append(raw_extra)
            c_extra_buf = ffi.from_buffer(raw_extra)
            extra_c_buffers.append(c_extra_buf)
            name_bytes = spec["name"].encode("utf-8")
            c_name = ffi.new("char[]", name_bytes) if name_bytes else ffi.NULL
            extra_name_buffers.append(c_name)
            c_extras[flat_i].pixels = c_extra_buf
            c_extras[flat_i].size = len(raw_extra)
            c_extras[flat_i].xsize = extra_arr.shape[1]
            c_extras[flat_i].ysize = extra_arr.shape[0]
            c_extras[flat_i].dtype = _DTYPE_TO_NATIVE[extra_arr.dtype]
            c_extras[flat_i].bits_per_sample = int(spec["bits_per_sample"])
            c_extras[flat_i].type = int(spec["type_id"])
            c_extras[flat_i].name = c_name
            c_extras[flat_i].name_size = len(name_bytes)

    result = lib.jxlpy_encode_multiframe_ex(
        c_frames,
        len(arrays),
        w,
        h,
        channels,
        dtype_id,
        int(bits_per_sample),
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
        return _consume_result(lib.jxlpy_info(c_data, len(data))).meta
    native = _consume_result(lib.jxlpy_decode_image_bytes(c_data, len(data), 0))
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
) -> dict[str, Any]:
    """Analyze a frame sequence and report whether multiframe encoding is beneficial.

    Returns a dict with per-frame diff stats and a recommendation.
    """
    arrays = _frame_arrays(frames, layout=layout)
    h, w, channels = arrays[0].shape
    full_area = w * h
    frame_stats = []

    for i, arr in enumerate(arrays):
        if i == 0:
            frame_stats.append({
                "index": 0,
                "changed_pixels": full_area,
                "changed_pct": 100.0,
                "bbox_area": full_area,
                "bbox_pct": 100.0,
            })
            continue
        changed = np.any(arr != arrays[i - 1], axis=2)
        changed_count = int(np.sum(changed))
        if changed_count == 0:
            frame_stats.append({
                "index": i,
                "changed_pixels": 0,
                "changed_pct": 0.0,
                "bbox_area": 0,
                "bbox_pct": 0.0,
            })
        else:
            ys, xs = np.nonzero(changed)
            bbox_w = int(xs.max()) - int(xs.min()) + 1
            bbox_h = int(ys.max()) - int(ys.min()) + 1
            bbox_area = bbox_w * bbox_h
            frame_stats.append({
                "index": i,
                "changed_pixels": changed_count,
                "changed_pct": changed_count / full_area * 100.0,
                "bbox_area": bbox_area,
                "bbox_pct": bbox_area / full_area * 100.0,
            })

    avg_bbox_pct = np.mean([s["bbox_pct"] for s in frame_stats[1:]]) if len(frame_stats) > 1 else 100.0
    avg_changed_pct = np.mean([s["changed_pct"] for s in frame_stats[1:]]) if len(frame_stats) > 1 else 100.0

    if avg_bbox_pct < 30:
        recommendation = "highly_beneficial"
    elif avg_bbox_pct < 70:
        recommendation = "moderately_beneficial"
    else:
        recommendation = "minimal_benefit"

    return {
        "num_frames": len(arrays),
        "canvas_size": (w, h),
        "channels": channels,
        "dtype": arrays[0].dtype,
        "avg_bbox_pct": float(avg_bbox_pct),
        "avg_changed_pct": float(avg_changed_pct),
        "recommendation": recommendation,
        "frames": frame_stats,
    }
