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

## Extra Channels

Extra channels are non-color planes such as depth, masks, or heat maps.

```python
jxl = jxlpy.encode(
    rgb,
    extra_channels=[
        ("mask", "selection_mask", mask),
        {"name": "depth", "type": "depth", "data": depth_u16},
    ],
)
```

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

The wrapper stores exact delta layers using `JXL_BLEND_REPLACE` plus crop. It
does not use transparent-black `BLEND` tricks for exact preservation.

Reference modes:

| `reference` | Meaning |
|---|---|
| `"auto"` | Compare previous-frame and first-frame references, choose smaller bbox. |
| `"previous"` | Compare only against previous reconstructed frame. |
| `"first"` | Compare only against the first frame. |
| `"none"` / `"full"` | Store every frame full-size. |

Example:

```python
jxl = jxlpy.encode_multiframe(
    frames,
    distance=0,
    effort=7,
    reference="auto",
    min_crop_ratio=0.98,
)
```

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
- `--dec-hints` and ICC/color-hint helpers.
- Exif/XMP/JUMBF file override helpers.
- `--frame_indexing`.
- CLI benchmark flags such as `--num_reps`, `--disable_output`, and verbose
  diagnostics.
- Decode-side thread count.

Some libjxl settings require more than a frame setting. For example, metadata
box editing and custom ICC handling need additional input data plumbing, not
only `JXL_ENC_FRAME_SETTING_*`.
