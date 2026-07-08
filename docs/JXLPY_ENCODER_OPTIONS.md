# jxlpy Encoder Option Coverage

`jxlpy` is not a full `cjxl` command line clone. It is a Python/native API for
in-memory arrays, bytes, files, extra channels, and multi-frame JXL. The current
binding exposes most core libjxl encoder knobs that map directly to
`JXL_ENC_FRAME_SETTING_*` or `JXLCompressParams`.

For the full libjxl parameter map, including every public frame setting id,
CLI-only wrapper options, and external-search guidance, see
`docs/JXL_ENCODER_PARAMETER_REFERENCE.md`.

## Common Options

| Python option | Rough `cjxl` option | Notes |
|---|---|---|
| `distance` | `-d`, `--distance` | `None` defaults to lossless. `0.0` is pixel lossless. |
| `lossless` | derived from distance | `None` auto-detects from `distance`. |
| `lossless_jpeg` | `--lossless_jpeg` | JPEG input only. `False` allows decoded-pixel lossless JXL. |
| `alpha_distance` | `--alpha_distance` | Defaults to `0.0`. |
| `effort` | `-e`, `--effort` | Normal range `1..10`; `11` requires `allow_expert_options=True`. |
| `allow_expert_options` | `--allow_expert_options` | Currently only needed for effort 11. |
| `threads` | `--num_threads` | Encode side. `0` means libjxl default in this binding. |
| `use_container` | `--container=1` | Forces container mode. |
| `level` | `--codestream_level` | `-1`, `5`, or `10`. |
| `jpeg_store_metadata` | `--allow_jpeg_reconstruction` | Controls JPEG reconstruction metadata for lossless JPEG transcode. |
| `compress_boxes` | `--compress_boxes` | Metadata/JPEG box Brotli compression. |
| `brotli_effort` | `--brotli_effort` | `None` keeps encoder default. |

## Density And Tool Toggles

| Python option | Rough `cjxl` option |
|---|---|
| `patches` | `--patches` |
| `dots` | `--dots` |
| `noise` | `--noise` |
| `gaborish` | `--gaborish` |
| `epf` | `--epf` |
| `faster_decoding` | `--faster_decoding` |
| `keep_invisible` | `--keep_invisible` |
| `photon_noise_iso` | `--photon_noise_iso` |
| `disable_perceptual_optimizations` | `--disable_perceptual_optimizations` |

For tri-state booleans, use:

```python
None   # encoder default
False  # force 0
True   # force 1
```

Example:

```python
jxlpy.encode("in.png", "out.jxl", distance=0, effort=9, patches=True)
jxlpy.encode("in.png", "out.jxl", distance=0, effort=10, patches=False)
```

## Preset Dictionaries And Passthrough

All first-class encoder options can also be passed through `encoder_options`.
This is useful for reusable presets and avoids adding a new Python parameter for
every experiment:

```python
screenshot_preset = {
    "distance": 0,
    "effort": 9,
    "modular_group_size": 3,
    "iterations": 100,
    "modular_predictor": 0,
    "modular_palette_colors": 10000,
    "patches": False,
    "post_compact": 0,
}

jxlpy.encode("content.png", "content.jxl", encoder_options=screenshot_preset)
```

Unknown keys in `encoder_options` are interpreted as low-level
`JXL_ENC_FRAME_SETTING_*` names. Prefer the explicit `frame_settings` argument
when using low-level settings:

```python
jxlpy.encode(
    "input.png",
    "output.jxl",
    distance=0,
    effort=9,
    frame_settings={
        "use_full_image_heuristics": 0,
        "JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM": 1,
    },
)
```

`frame_settings` accepts short names, full enum names, or numeric enum ids. It
requires a rebuilt native shim with passthrough support; older `jxlpy_native`
builds raise a rebuild error instead of silently ignoring the options.

## Progressive Options

| Python option | Rough `cjxl` option |
|---|---|
| `progressive` | `-p`, `--progressive` |
| `group_order` | `--group_order` |
| `center_x`, `center_y` | `--center_x`, `--center_y` |
| `progressive_ac` | `--progressive_ac` |
| `qprogressive_ac` | `--qprogressive_ac` |
| `progressive_dc` | `--progressive_dc` |
| `responsive` | `--responsive` |

`progressive=True` follows the same high-level intent as `cjxl -p`: it enables
progressive AC, sets `progressive_dc=1`, uses center-first group order, disables
patches unless explicitly overridden, and enables responsive mode unless
explicitly overridden.

Do not enable this by default for lossless archive use. On some low-entropy
document screenshots, progressive lossless output can be much larger than the
normal lossless path.

## Resampling And Alpha

| Python option | Rough `cjxl` option |
|---|---|
| `resampling` | `--resampling` |
| `ec_resampling` | `--ec_resampling` |
| `already_downsampled` | `--already_downsampled` |
| `upsampling_mode` | `--upsampling_mode` |
| `premultiply` | `--premultiply` |
| `override_bitdepth` | `--override_bitdepth` |
| `intensity_target` | `--intensity_target` |
| `buffering` | `--buffering` |

## Modular Options

| Python option | Rough `cjxl` option |
|---|---|
| `modular` | `--modular` |
| `modular_group_size` | `--modular_group_size` |
| `modular_predictor` | `--modular_predictor` |
| `modular_colorspace` | `--modular_colorspace` |
| `modular_ma_tree_learning_percent`, `iterations` | `-I`, `--iterations` |
| `modular_nb_prev_channels` | `--modular_nb_prev_channels` |
| `modular_palette_colors` | `--modular_palette_colors` |
| `modular_lossy_palette` | `--modular_lossy_palette` |
| `modular_channel_colors_global_percent`, `pre_compact` | `-X`, `--pre-compact` |
| `modular_channel_colors_group_percent`, `post_compact` | `-Y`, `--post-compact` |

Important `cjxl` short-option mapping:

- `-g` is `modular_group_size`, not progressive `group_order`.
- `-I` is `iterations` / `modular_ma_tree_learning_percent`, not
  `intensity_target`.
- `-P` is `modular_predictor`, not patches/progressive.
- `-Y` is `post_compact` / `modular_channel_colors_group_percent`.

The community screenshot/document preset:

```bash
cjxl content.png content.jxl -d 0 -e 9 -g 3 -I 100 -P 0 \
  --modular_palette_colors=10000 --patches 0 -Y 0
```

is equivalent to:

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

The fully explicit low-level spelling is also supported:

```python
jxlpy.encode(
    "content.png",
    "content.jxl",
    distance=0,
    effort=9,
    modular_group_size=3,
    modular_ma_tree_learning_percent=100,
    modular_predictor=0,
    modular_palette_colors=10000,
    patches=False,
    modular_channel_colors_group_percent=0,
)
```

## JPEG-Specific Options

| Python option | Rough `cjxl` option |
|---|---|
| `lossless_jpeg` | `--lossless_jpeg` |
| `jpeg_store_metadata` | `--allow_jpeg_reconstruction` |
| `jpeg_reconstruction_cfl` | `--jpeg_reconstruction_cfl` |
| `compress_boxes` | `--compress_boxes` |
| `brotli_effort` | `--brotli_effort` |

Examples:

```python
# Bit-exact JPEG reconstruction path.
jxlpy.encode("in.jpg", "out.jxl", lossless=True, lossless_jpeg=True)

# Decoded-pixel lossless path; cannot reconstruct original JPEG bytes.
jxlpy.encode("in.jpg", "out.jxl", distance=0, lossless_jpeg=False)
```

## Still Not Aiming To Mirror

These `cjxl` features are still intentionally not exposed as first-class Python
parameters:

- `--quality`: use `distance` directly for now.
- `--dec-hints` and ICC/Exif/XMP/JUMBF file override helpers.
- Metadata strip keys such as `strip=exif`; only JPEG reconstruction storage is
  exposed at the moment.
- `--frame_indexing`.
- `--streaming_input`, `--streaming_output`, and direct output processor
  behavior.
- Benchmark/developer flags such as `--num_reps`, `--disable_output`, verbose
  mode, and command-line-only diagnostics.

Decode-side thread count is also not exposed yet. Encoder calls support
`threads=...`.

Some non-first-class `JXL_ENC_FRAME_SETTING_*` values, such as
`output_mode`, `color_transform`, `jpeg_keep_exif`, `jpeg_keep_xmp`,
`jpeg_keep_jumbf`, and `use_full_image_heuristics`, can be passed through
`frame_settings`. This only covers libjxl frame settings; features that need
extra input data or CLI-only behavior still need dedicated wrapper code.

## Multi-Frame Analysis

`jxlpy.analyze_multiframe(...)` now mirrors the `encode_multiframe(...)`
reference and crop decision logic:

```python
report = jxlpy.analyze_multiframe(
    frames,
    extra_channels=[("mask", "selection_mask", masks)],
    reference="auto",
    min_crop_ratio=0.98,
)
```

Supported analysis options:

| Python option | Meaning |
|---|---|
| `reference` | Crop/replace analysis modes: `auto`, `first`, `previous`, `none`, `full`. The experimental `blend_mask`, `add`, and `patch_add` encoder paths are not modeled by the analyzer. |
| `min_crop_ratio` | Same crop cutoff as encoding. |
| `extra_channels` | Optional extra channel planes; changes are included in the diff bbox. |

The returned report keeps the older aggregate fields:

- `num_frames`
- `canvas_size`
- `avg_bbox_pct`
- `avg_changed_pct`
- `recommendation`
- `frames`

Each frame entry also includes encoder-oriented fields:

- `source`: `none`, `first`, or `previous`
- `source_ref`: JXL reference id used by the frame
- `save_as_ref`: JXL reference id saved by the frame
- `use_crop`
- `bbox_x0`, `bbox_y0`, `bbox_x1`, `bbox_y1`
- `bbox_xsize`, `bbox_ysize`, `bbox_pct`
- `crop_x0`, `crop_y0`, `crop_x1`, `crop_y1`
- `crop_xsize`, `crop_ysize`
- `encoded_pct`

For `reference="auto"`, the analyzer picks the smaller bbox between the saved
previous-frame reference and first-frame reference, matching the encoder path.
This means reports can differ from older versions that only compared every frame
against the immediately previous frame.

`bbox_*` describes the candidate difference region that was detected. `crop_*`
describes the layer that will actually be encoded. They are the same when
`use_crop=True`; if `min_crop_ratio` rejects a crop, `crop_*` is the full canvas
while `bbox_*` still shows the detected region.
