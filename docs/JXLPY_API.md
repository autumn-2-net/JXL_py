# jxlpy API

`jxlpy` is a Python/CFFI wrapper around the local `jxlpy_native` shim and the
vendored `libjxl` tree. It is designed for direct use from Python with paths,
bytes, NumPy arrays, and Torch tensors.

## Import

```python
import jxlpy
```

## Encoding

```python
jxlpy.encode(src, output=None, **options) -> bytes | pathlib.Path
```

`src` can be:

- path string or `Path`
- encoded image bytes, such as PNG/JPEG/JXL bytes
- NumPy array
- Torch tensor

`output=None` returns JXL bytes. Passing `output="file.jxl"` writes the file and
returns the output path.

Examples:

```python
jxl_bytes = jxlpy.encode("input.png", distance=0, effort=9)
jxlpy.encode("input.png", "output.jxl", distance=0, effort=9)
jxl_bytes = jxlpy.encode(array_hwc, distance=0)
```

For array/tensor input, supported main-image shapes are:

- `H x W`
- `H x W x C`
- `C x H x W` with `layout="chw"`

Main-image channel count must be `1`, `2`, `3`, or `4`. Use `extra_channels`
for additional planes.

Supported dtypes:

- `uint8`
- `uint16`
- `float16`
- `float32`

## Decoding

```python
jxlpy.decode(src, frame=0, coalesced=True, return_info=False, **options)
```

`src` can be a path or bytes. By default this returns a NumPy array.

Common examples:

```python
image = jxlpy.decode("image.jxl")
image, meta = jxlpy.decode("image.jxl", return_info=True)
frame2 = jxlpy.decode(jxl_bytes, frame=2, coalesced=True)
```

`coalesced=True` returns the reconstructed full frame. Use `decode_layer` to
inspect the stored crop/layer itself.

Target-frame decoding stops immediately after the requested frame. For
coalesced animation frames, libjxl's frame-skip API avoids producing pixel
buffers for earlier frames while retaining any reference dependencies required
by the codestream. Non-coalesced layers still require sequential dependency
processing.

```python
frame, meta = jxlpy.decode(
    jxl_bytes,
    frame=20,
    return_info=True,
    threads=4,
    max_pixels=100_000_000,
    max_output_bytes=1_000_000_000,
)
```

After an early stop, `meta["num_frames_known"]` is false and `num_frames` is
the number reached so far. Pass `scan_all_frames=True` when an exact total is
required, or call `jxlpy.info()` to scan frame headers without decoding pixels.

## Extra Channels

Extra channels are non-color planes such as depth, masks, or heat maps.
When `type` is omitted, jxlpy uses `optional`, the JXL type intended for
application-defined planes. `unknown` may appear in decoder metadata but cannot
be emitted by libjxl.

```python
jxl = jxlpy.encode(
    rgb,
    extra_channels=[
        ("mask", "selection_mask", mask),
        {"name": "depth", "type": "depth", "data": depth_u16},
    ],
)
```

The mapping form exposes the complete public extra-channel metadata:

```python
{
    "name": "depth",
    "type": "depth",
    "data": depth_u16,            # full-resolution plane
    "bits_per_sample": 16,
    "exponent_bits_per_sample": 0,
    "dim_shift": 1,               # codestream downsampling metadata
    "alpha_premultiplied": False, # alpha channels only
    "spot_color": (0, 0, 0, 0),  # linear RGBA, spot channels only
    "cfa_channel": 0,             # CFA channels only
}
```

Decoded channel dictionaries return these fields plus `xsize`, `ysize`, and
`exponent_bits_per_sample`. The public libjxl decoder outputs full-resolution
extra planes even when `dim_shift` is nonzero. `jxlpy.info()` returns the same
metadata under `extra_channels` without decoding pixel planes.

`dim_shift` accepts `0..3`. A nonzero value instructs libjxl to downsample that
plane in the codestream, so it is not byte-exact even when the main encode is
lossless. jxlpy automatically raises `ec_resampling` to at least
`2 ** dim_shift`; an explicitly smaller value is rejected instead of failing
later in `JxlEncoderProcessOutput`.

## Color Metadata

Array and multi-frame encoding can declare a structured color encoding or an
ICC profile. They are mutually exclusive.

```python
jxl = jxlpy.encode(rgb, color_encoding="linear_srgb")
jxl = jxlpy.encode(rgb, icc_profile="display.icc")
```

Structured presets are `srgb`, `linear_srgb`, `gray_srgb`, `linear_gray`,
`display_p3`, `rec2100_pq`, and `rec2100_hlg`. A mapping can directly set the
color-space, white-point, primaries, transfer-function, gamma, rendering intent,
and custom xy coordinates. `info()` and decode metadata expose
`color_encoding`, `color_profile_is_icc`, and `icc_profile` for the original
profile. The corresponding decoder-output profile is exposed as
`data_color_encoding`, `data_color_profile_is_icc`, and `data_icc_profile`.
Overriding color metadata on JPEG input disables JPEG bitstream reconstruction
and uses pixel-lossless/lossy encoding as requested, because an exact JPEG
transcode cannot also change its color profile.

Decode one channel:

```python
mask, meta = jxlpy.decode_extra_channel(jxl, index=0)
```

Decode main image plus all extra channels:

```python
image, meta = jxlpy.decode(
    jxl,
    return_info=True,
    return_extra_channels=True,
)
```

## Multi-Frame Encoding

```python
jxlpy.encode_multiframe(frames, output=None, **options)
```

By default the wrapper stores exact delta layers using `JXL_BLEND_REPLACE` plus
crop. It also has experimental masked BLEND and ADD residual modes for testing
scattered changes.

Reference modes:

| `reference` | Meaning |
|---|---|
| `"auto"` | Compare previous-frame and first-frame references, choose smaller bbox. |
| `"previous"` | Compare only against previous reconstructed frame. |
| `"first"` | Compare only against the first frame. |
| `"none"` / `"full"` | Store every frame full-size. |
| `"blend_mask"` / `"mask"` | Experimental: store changed pixels plus an internal binary mask extra channel, then compose with `JXL_BLEND_BLEND` against the previous reference. |
| `"blend_mask8"` / `"mask8"` | Experimental comparison mode: same as `blend_mask`, but store the mask at the main image integer bit depth instead of declaring it as 1-bit. |
| `"add"` / `"additive"` | Experimental: store full-frame float32 residuals and compose with `JXL_BLEND_ADD`. |

Example:

```python
jxl = jxlpy.encode_multiframe(
    frames,
    durations=None,
    distance=0,
    effort=7,
    reference="auto",
    min_crop_ratio=0.98,
)
```

`durations=None` or `durations=0` creates true zero-duration layers and leaves
`have_animation` false. Positive durations create animation frames; a list may
mix zero-duration layers and timed frames. The default remains `durations=1`
for backward compatibility.

Experimental masked BLEND mode:

```python
jxl = jxlpy.encode_multiframe(
    frames,
    distance=0,
    effort=3,
    reference="blend_mask",
)
```

`reference="blend_mask"` keeps integer samples, uses the previous frame as the
reference, and adds an internal 1-bit `selection_mask` extra channel named
`jxlpy_blend_mask`. Within the selected bbox, changed pixels contain current
samples and mask value 1; unchanged pixels are zeroed and mask value 0. The
decoder composites with `JXL_BLEND_BLEND`, so unchanged pixels come from the
previous reference. Main-image decoding remains exact for `uint8`/`uint16`
inputs, including RGBA, because the mask is separate from the real alpha
channel. If all extra channels are requested during decode, the internal mask is
visible as an extra channel. `reference="blend_mask8"` keeps the older full
integer bit-depth mask for comparison; it can be smaller on some modular inputs.

Experimental ADD residual mode:

```python
jxl = jxlpy.encode_multiframe(
    frames,
    distance=0,
    effort=3,
    reference="add",
)
```

`reference="add"` converts integer frames to normalized `float32`, stores the
first frame directly, and stores later frames as `current - previous` with
`JXL_BLEND_ADD`. This can help scattered changes avoid a huge crop bbox, but it
is an experiment: decoded frames are float samples, and exact byte-for-byte
RGBA preservation must be validated for the target data.

Read the reconstructed frame:

```python
frame = jxlpy.decode(jxl, frame=3, coalesced=True)
```

Read the stored crop/layer:

```python
layer, meta = jxlpy.decode_layer(jxl, layer=3)
```

Analyze the crop/reference plan without encoding:

```python
report = jxlpy.analyze_multiframe(
    frames,
    extra_channels=[("mask", "selection_mask", masks)],
    reference="auto",
    min_crop_ratio=0.98,
)
```

The analyzer mirrors `encode_multiframe` reference and crop decisions.

## Encoder Options

The recommended API is explicit keyword arguments:

```python
jxlpy.encode("input.png", distance=0, effort=9, patches=False)
```

For preset dictionaries, use `encoder_options`. This is applied after the
regular keyword defaults, so it can override defaults:

```python
preset = {
    "distance": 0,
    "effort": 9,
    "modular_group_size": 3,
    "iterations": 100,
    "modular_predictor": 0,
    "modular_palette_colors": 10000,
    "patches": False,
    "post_compact": 0,
}

jxlpy.encode("content.png", "content.jxl", encoder_options=preset)
```

Unknown keys in `encoder_options` are treated as low-level frame settings. For
clarity, prefer the explicit `frame_settings` argument for low-level settings.

## Low-Level Frame Setting Passthrough

`frame_settings` passes values to libjxl `JXL_ENC_FRAME_SETTING_*`. This is for
advanced settings that are not first-class Python arguments.

```python
jxlpy.encode(
    "input.png",
    distance=0,
    effort=9,
    frame_settings={
        "use_full_image_heuristics": 0,
        "color_transform": 1,
    },
)
```

Accepted setting keys:

- short names, such as `"color_transform"`
- full enum names, such as `"JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM"`
- numeric enum ids, such as `24`

Most values are integers. Float-valued settings are detected for known ids:

- `photon_noise`
- `channel_colors_global_percent`
- `channel_colors_group_percent`
- `modular_ma_tree_learning_percent`

For an unknown numeric id that needs a float value, force it with:

```python
jxlpy.encode(img, frame_settings={123: ("float", 50.0)})
```

`frame_settings` requires a native shim built with passthrough support. If the
loaded `jxlpy_native` is older, `jxlpy` raises a clear rebuild error instead of
silently ignoring the setting.

## Screenshot/Document Preset

This `cjxl` command:

```bash
cjxl content.png content.jxl -d 0 -e 9 -g 3 -I 100 -P 0 \
  --modular_palette_colors=10000 --patches 0 -Y 0
```

maps to:

```python
jxlpy.encode(
    "content.png",
    "content.jxl",
    distance=0,
    effort=9,
    modular_group_size=3,
    iterations=100,
    modular_predictor=0,
    modular_palette_colors=10000,
    patches=False,
    post_compact=0,
)
```

Short-option traps:

- `-g` means `modular_group_size`, not `group_order`.
- `-I` means `iterations`, not `intensity_target`.
- `-P` means `modular_predictor`, not patches.
- `-Y` means `post_compact`.

## Current Boundaries

The wrapper does not try to be a complete `cjxl` command-line clone.

Still not first-class API:

- `--quality`: use `distance` directly.
- `--dec-hints` for ambiguous input formats.
- Exif/XMP/JUMBF file override helpers.
- `--frame_indexing`.
- CLI benchmark flags such as `--num_reps`, `--disable_output`, and verbose
  diagnostics.

Some libjxl settings require more than a frame setting. For example, metadata
box editing needs additional input data plumbing, not only
`JXL_ENC_FRAME_SETTING_*`.

## Native ABI Compatibility

The CFFI/native boundary is versioned. Import validates ABI version 2 and the
native size of every shared structure before exposing the library. A stale or
wrong-architecture `jxlpy_native` fails immediately with a rebuild message.
After changing `native/jxlpy_native.h`, rebuild the shim before running Python:

```bat
.\scripts\build_windows.cmd jxlpy_native
```
