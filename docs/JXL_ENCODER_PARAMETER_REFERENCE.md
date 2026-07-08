# JPEG XL Encoder Parameter Reference

This document is based on the vendored `libjxl` source in this repository:

- `libjxl/lib/include/jxl/encode.h`
- `libjxl/lib/jxl/encode.cc`
- `libjxl/tools/cjxl_main.cc`

It focuses on parameters that matter when building an external search layer
around `cjxl` / `jxlpy`.

## Mental Model

JPEG XL has several parameter layers:

1. **Global encoder / wrapper parameters**
   - Not `JXL_ENC_FRAME_SETTING_*`.
   - Examples: input bytes, output path, JPEG reconstruction, container mode,
     codestream level, threads, ICC/metadata plumbing, frame crop/blend.
2. **Frame settings**
   - Public C API enum `JxlEncoderFrameSettingId`.
   - Current vendored libjxl exposes ids `0..40`, plus `65535` fill enum.
   - These control effort, coding tools, modular options, progressive options,
     buffering, JPEG metadata keep flags, and similar encoder choices.
3. **`cjxl` command-line conveniences**
   - Some options directly map to frame settings.
   - Some are convenience flags that set multiple options.
   - Some are pure CLI behavior, such as repeated benchmarking, verbose output,
     file metadata override helpers, and stream/file I/O behavior.
4. **`jxlpy` convenience layer**
   - First-class keyword options for common settings.
   - `encoder_options={...}` for reusable presets.
   - `frame_settings={...}` for low-level passthrough to frame settings.

The important practical point: the encoder effort levels do internal heuristic
search, but they are not an exhaustive search over all meaningful combinations.
External search over selected knobs can beat the default strategy on screenshots,
documents, synthetic images, and multi-frame data.

## How Many Parameters Are There?

In this vendored libjxl:

- Public frame settings: **41 ids** from `0` through `40`.
- Special fill enum: `65535`, not a usable option.
- `cjxl` exposes roughly another layer of wrapper parameters around input,
  output, metadata, JPEG reconstruction, threading, and benchmarking.
- `jxlpy` exposes most common encoder knobs as keywords and lets you pass raw
  frame settings for the rest.

Frame settings are not the whole encoder API. For example, crop/blend/reference
for multi-frame JXL is frame header metadata, not a frame setting. Likewise ICC
profiles and metadata boxes require actual byte payloads, not just an integer
setting.

## `jxlpy` Passing Styles

Prefer explicit keyword arguments for normal use:

```python
jxlpy.encode("in.png", "out.jxl", distance=0, effort=9, patches=False)
```

Use `encoder_options` for reusable presets:

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

Use `frame_settings` for direct `JXL_ENC_FRAME_SETTING_*` passthrough:

```python
jxlpy.encode(
    "in.png",
    "out.jxl",
    distance=0,
    effort=9,
    frame_settings={
        "use_full_image_heuristics": 0,
        "JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM": 1,
        38: 0,
    },
)
```

Accepted low-level keys:

- short names: `"color_transform"`
- full enum names: `"JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM"`
- numeric ids: `24`

Most settings are integer settings. Float settings are detected for known ids:

- `photon_noise`
- `channel_colors_global_percent`
- `channel_colors_group_percent`
- `modular_ma_tree_learning_percent`

For a numeric id that should be sent through the float API:

```python
jxlpy.encode(img, frame_settings={123: ("float", 50.0)})
```

## High-Level / Non-Frame Options

These are not all `JXL_ENC_FRAME_SETTING_*`, but they matter for real output.

| Concept | `cjxl` | `jxlpy` | Range / values | Notes |
|---|---|---|---|---|
| Distance | `-d`, `--distance` | `distance` | `0..25` | `0` is mathematically lossless. Main lossy size/quality knob. |
| Quality | `-q`, `--quality` | not first-class | `0..100` | CLI maps this to distance. Use `distance` directly in `jxlpy`. |
| Alpha distance | `-a`, `--alpha_distance` | `alpha_distance` | `0..25` | Defaults to exact alpha in `jxlpy`. |
| Effort | `-e`, `--effort` | `effort` | `1..10`, `11` expert | Also frame setting id 0. Effort 11 needs `allow_expert_options`. |
| Expert options | `--allow_expert_options` | `allow_expert_options` | bool | Required for effort 11. Extreme cost. |
| Threads | `--num_threads` | `threads` | int | `jxlpy` encode side only. |
| Modular mode | `-m`, `--modular` | `modular` | `-1/0/1` | Also frame setting id 11. Usually lossless uses modular. |
| JPEG transcode | `-j`, `--lossless_jpeg` | `lossless_jpeg` | `0/1` | For JPEG input. `1` preserves JPEG reconstruction. |
| JPEG reconstruction metadata | `--allow_jpeg_reconstruction` | `jpeg_store_metadata` | `0/1` | Controls whether exact JPEG bytes can be reconstructed. |
| Container | `--container` | `use_container` | `0/1` | Forces BMFF container. Needed automatically for some metadata/JPEG cases. |
| Codestream level | `--codestream_level` | `level` | `-1/5/10` | Level 10 permits larger/more complex features. |
| Brotli boxes | `--compress_boxes` | `compress_boxes` | `0/1` | Metadata/JPEG boxes. Also frame setting id 33 for JPEG boxes. |
| Brotli effort | `--brotli_effort` | `brotli_effort` | `-1..11` | Helps JPEG recompression / metadata boxes. |
| Intensity target | `--intensity_target` | `intensity_target` | nits, `0` default | HDR/tone metadata. Not `-I`. |
| Premultiply alpha | `--premultiply` | `premultiply` | `-1/0/1` | Alpha association metadata. Be careful for exact alpha workflows. |
| Override bit depth | `--override_bitdepth` | `override_bitdepth` | bit depth | Metadata/input bit-depth override. |
| Upsampling mode | `--upsampling_mode` | `upsampling_mode` | `-1/0/1` | Used with resampling/already-downsampled. |
| Frame durations | no simple still-image flag | `durations`, `tps` | ints | Multi-frame animation timing in `jxlpy`. |
| Crop/reference/blend | frame header API | `reference`, `min_crop_ratio` | strings / float | `jxlpy` owns exact multi-frame delta crop logic. |
| Extra channels | input dependent | `extra_channels` | planes | Depth/mask/thermal/etc. |
| Metadata/ICC hints | `-x`, `--dec-hints` | not first-class | key/value | Needs additional byte/file plumbing. |
| Benchmark repeats | `--num_reps` | not exposed | int | CLI-only benchmarking wrapper. |

## `cjxl` Short Options That Are Easy To Misread

| Short | Long | Meaning |
|---|---|---|
| `-d` | `--distance` | Quality/visual distance. |
| `-q` | `--quality` | CLI convenience mapped to distance. |
| `-e` | `--effort` | Encoder effort. |
| `-a` | `--alpha_distance` | Alpha-channel distance. |
| `-p` | `--progressive` | Convenience flag for progressive/responsive behavior. |
| `-m` | `--modular` | Force modular/VarDCT. |
| `-j` | `--lossless_jpeg` | JPEG lossless transcode mode. |
| `-I` | `--iterations` | Modular MA-tree learning percent. Not intensity. |
| `-C` | `--modular_colorspace` | Modular reversible color transform. |
| `-g` | `--modular_group_size` | Modular group size. Not progressive group order. |
| `-P` | `--modular_predictor` | Modular predictor. Not patches. |
| `-E` | `--modular_nb_prev_channels` | Previous-channel MA properties. |
| `-X` | `--pre-compact` | Global channel palette threshold. |
| `-Y` | `--post-compact` | Local channel palette threshold. |
| `-R` | `--responsive` | Modular squeeze/progressive behavior. |

`--patches` has no short `-P` form.

## Public Frame Settings

The following table lists all current public frame setting ids in this vendored
libjxl. `jxlpy` can pass these with `frame_settings`, and many also have
first-class keyword aliases.

| ID | Setting key | C enum | Type/range | `cjxl` / `jxlpy` mapping | Search notes |
|---:|---|---|---|---|---|
| 0 | `effort` | `JXL_ENC_FRAME_SETTING_EFFORT` | int `1..10`, `11` expert | `-e`, `effort` | Primary speed/size knob. Search `7/8/9/10`; `10` can be much slower. |
| 1 | `decoding_speed` | `JXL_ENC_FRAME_SETTING_DECODING_SPEED` | int `0..4` | `--faster_decoding`, `faster_decoding` | Larger improves decode speed but can hurt density/quality. Usually keep `0`. |
| 2 | `resampling` | `JXL_ENC_FRAME_SETTING_RESAMPLING` | `-1/1/2/4/8` | `--resampling`, `resampling` | Lossy only unless intentionally downsampling. Avoid for archival lossless. |
| 3 | `extra_channel_resampling` | `JXL_ENC_FRAME_SETTING_EXTRA_CHANNEL_RESAMPLING` | `-1/1/2/4/8` | `--ec_resampling`, `ec_resampling` | Alpha/mask downsampling. Dangerous for exact masks. |
| 4 | `already_downsampled` | `JXL_ENC_FRAME_SETTING_ALREADY_DOWNSAMPLED` | `0/1` | `--already_downsampled`, `already_downsampled` | Advanced API path; input must already match downsampled dimensions. |
| 5 | `photon_noise` | `JXL_ENC_FRAME_SETTING_PHOTON_NOISE` | float `>=0` | `--photon_noise_iso`, `photon_noise_iso` | Adds synthetic noise. This changes pixels perceptually; not a density trick. |
| 6 | `noise` | `JXL_ENC_FRAME_SETTING_NOISE` | `-1/0/1` | `--noise`, `noise` | Older adaptive noise toggle. Prefer photon noise if intentionally adding noise. |
| 7 | `dots` | `JXL_ENC_FRAME_SETTING_DOTS` | `-1/0/1` | `--dots`, `dots` | Encoder tool toggle. Usually not a first search target. |
| 8 | `patches` | `JXL_ENC_FRAME_SETTING_PATCHES` | `-1/0/1` | `--patches`, `patches` | Intra-frame patch dictionary. Helps some screenshots/text/repeated blocks, can be slow and not always smaller. |
| 9 | `epf` | `JXL_ENC_FRAME_SETTING_EPF` | `-1..3` | `--epf`, `epf` | Edge preserving filter. Mostly lossy/visual tuning. |
| 10 | `gaborish` | `JXL_ENC_FRAME_SETTING_GABORISH` | `-1/0/1` | `--gaborish`, `gaborish` | Filter toggle. Can affect lossy density/quality; test for exact synthetic cases if needed. |
| 11 | `modular` | `JXL_ENC_FRAME_SETTING_MODULAR` | `-1/0/1` | `-m`, `modular` | Force VarDCT or modular. Lossless usually modular. |
| 12 | `keep_invisible` | `JXL_ENC_FRAME_SETTING_KEEP_INVISIBLE` | `-1/0/1` | `--keep_invisible`, `keep_invisible` | Critical if RGB under alpha=0 must roundtrip exactly. |
| 13 | `group_order` | `JXL_ENC_FRAME_SETTING_GROUP_ORDER` | `-1/0/1` | `--group_order`, `group_order` | Progressive group storage order, not modular group size. |
| 14 | `group_order_center_x` | `JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_X` | `-1..xsize` | `--center_x`, `center_x` | Only meaningful with center-first group order. |
| 15 | `group_order_center_y` | `JXL_ENC_FRAME_SETTING_GROUP_ORDER_CENTER_Y` | `-1..ysize` | `--center_y`, `center_y` | Only meaningful with center-first group order. |
| 16 | `responsive` | `JXL_ENC_FRAME_SETTING_RESPONSIVE` | `-1/0/1` | `-R`, `--responsive`, `responsive` | Modular squeeze/progressive behavior. Can enlarge lossless screenshots. |
| 17 | `progressive_ac` | `JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC` | `-1/0/1` | `--progressive_ac`, `progressive_ac` | Progressive VarDCT AC. UX feature, not usually best density. |
| 18 | `qprogressive_ac` | `JXL_ENC_FRAME_SETTING_QPROGRESSIVE_AC` | `-1/0/1` | `--qprogressive_ac`, `qprogressive_ac` | Alternative progressive AC. |
| 19 | `progressive_dc` | `JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC` | `-1/0/1/2` | `--progressive_dc`, `progressive_dc` | Progressive DC passes. Can increase size. |
| 20 | `channel_colors_global_percent` | `JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GLOBAL_PERCENT` | float `-1..100` | `-X`, `pre_compact` | Modular global channel palette threshold. Search for synthetic/text. |
| 21 | `channel_colors_group_percent` | `JXL_ENC_FRAME_SETTING_CHANNEL_COLORS_GROUP_PERCENT` | float `-1..100` | `-Y`, `post_compact` | Modular local palette threshold. `0` helped screenshot preset. |
| 22 | `palette_colors` | `JXL_ENC_FRAME_SETTING_PALETTE_COLORS` | int `-1..70913` | `--modular_palette_colors`, `modular_palette_colors` | Very important for low-color images. Search `0/1024/10000/70913`. |
| 23 | `lossy_palette` | `JXL_ENC_FRAME_SETTING_LOSSY_PALETTE` | `-1/0/1` | `--modular_lossy_palette`, `modular_lossy_palette` | Lossy delta palette. Not for exact archival unless you know the effect. |
| 24 | `color_transform` | `JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM` | `-1/0/1/2` | `frame_settings` | `0=XYB`, `1=none/RGB`, `2=YCbCr marker`. Advanced. |
| 25 | `modular_color_space` | `JXL_ENC_FRAME_SETTING_MODULAR_COLOR_SPACE` | `-1..41` | `-C`, `modular_colorspace` | Modular RCT. `0=none`, `6=YCoCg`. Test for synthetic/illustration. |
| 26 | `modular_group_size` | `JXL_ENC_FRAME_SETTING_MODULAR_GROUP_SIZE` | `-1/0/1/2/3` | `-g`, `modular_group_size` | `0=128`, `1=256`, `2=512`, `3=1024`. Big effect on screenshots. |
| 27 | `modular_predictor` | `JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR` | `-1..15` | `-P`, `modular_predictor` | Core modular predictor. Try `0/5/6/14/15` first. |
| 28 | `modular_ma_tree_learning_percent` | `JXL_ENC_FRAME_SETTING_MODULAR_MA_TREE_LEARNING_PERCENT` | float `-1..100` | `-I`, `iterations` | Higher can improve density but costs memory/time. |
| 29 | `modular_nb_prev_channels` | `JXL_ENC_FRAME_SETTING_MODULAR_NB_PREV_CHANNELS` | `-1..11` | `-E`, `modular_nb_prev_channels` | Useful with many channels/extra channels. Higher slows encode/decode. |
| 30 | `jpeg_recon_cfl` | `JXL_ENC_FRAME_SETTING_JPEG_RECON_CFL` | `-1/0/1` | `--jpeg_reconstruction_cfl`, `jpeg_reconstruction_cfl` | JPEG lossless recompression detail. |
| 31 | `frame_index_box` | `JXL_ENC_FRAME_INDEX_BOX` | `0/1` | low-level only | Not the same as convenient `cjxl --frame_indexing` pattern. Requires valid indexed keyframes. |
| 32 | `brotli_effort` | `JXL_ENC_FRAME_SETTING_BROTLI_EFFORT` | `-1..11` | `--brotli_effort`, `brotli_effort` | JPEG recompression / brob boxes. |
| 33 | `jpeg_compress_boxes` | `JXL_ENC_FRAME_SETTING_JPEG_COMPRESS_BOXES` | `-1/0/1` | `--compress_boxes`, `compress_boxes` | JPEG-derived metadata boxes. |
| 34 | `buffering` | `JXL_ENC_FRAME_SETTING_BUFFERING` | `-1..3` | `--buffering`, `buffering` | Memory/streaming tradeoff. `0` can be densest, more memory. |
| 35 | `jpeg_keep_exif` | `JXL_ENC_FRAME_SETTING_JPEG_KEEP_EXIF` | `-1/0/1` | low-level / `dec-hints strip` in CLI | Cannot discard while storing JPEG reconstruction metadata. |
| 36 | `jpeg_keep_xmp` | `JXL_ENC_FRAME_SETTING_JPEG_KEEP_XMP` | `-1/0/1` | low-level / `dec-hints strip` in CLI | Same metadata caveats. |
| 37 | `jpeg_keep_jumbf` | `JXL_ENC_FRAME_SETTING_JPEG_KEEP_JUMBF` | `-1/0/1` | low-level | JPEG metadata handling. |
| 38 | `use_full_image_heuristics` | `JXL_ENC_FRAME_SETTING_USE_FULL_IMAGE_HEURISTICS` | `0/1` | `frame_settings` | Mostly streaming/test equivalence. Can be searched, but not usually first target. |
| 39 | `disable_perceptual_heuristics` | `JXL_ENC_FRAME_SETTING_DISABLE_PERCEPTUAL_HEURISTICS` | `0/1` | `--disable_perceptual_optimizations`, `disable_perceptual_optimizations` | Advanced. Interacts with original profile / XYB choices. |
| 40 | `output_mode` | `JXL_ENC_FRAME_SETTING_OUTPUT_MODE` | `-1/0/1/2` | `--output_mode`, `frame_settings` | Output streaming/order. Mostly memory/IO, not ordinary density. |

## Parameters Worth External Search

The full combinatorial space is huge. Search only a small candidate set per
image class.

### Lossless Screenshots / Documents

Strong candidates:

```python
[
    # Baseline.
    dict(distance=0, effort=8, patches=False),
    dict(distance=0, effort=9, patches=False),

    # Patch dictionary candidate.
    dict(distance=0, effort=8, patches=True),
    dict(distance=0, effort=9, patches=True),

    # Community screenshot preset.
    dict(
        distance=0,
        effort=9,
        modular_group_size=3,
        iterations=100,
        modular_predictor=0,
        modular_palette_colors=10000,
        patches=False,
        post_compact=0,
    ),
]
```

Then expand only if the file is important:

- `effort`: `8/9/10`
- `modular_group_size`: `2/3`
- `modular_predictor`: `0/5/6/14/15`
- `modular_palette_colors`: `0/1024/10000/70913`
- `post_compact`: `0/80/-1`
- `pre_compact`: `0/95/-1`
- `patches`: `False/True`
- `modular_colorspace`: `0/6/-1`

Observations from this project:

- White paper/doc screenshots can be weird; PNG may still win on some.
- Black UI screenshots and repeated text can benefit heavily from modular
  palette/predictor tuning.
- `patches=True` can help, but can also cost a lot of CPU for small gain.
- Progressive/responsive lossless can make document screenshots much larger.

### Lossless Illustrations / Synthetic PNG

Try:

- baseline `distance=0, effort=8/9`
- `patches=False/True`
- `modular_palette_colors`
- `modular_predictor`
- `modular_colorspace`
- `modular_group_size`

Do not assume the screenshot preset wins for illustrations. It is a candidate,
not a default.

### JPEG Input

First decide whether exact JPEG reconstruction matters.

If yes:

```python
jxlpy.encode("input.jpg", distance=0, lossless_jpeg=True, effort=8)
```

If no, and the decoded pixels are simple/synthetic, compare:

```python
jxlpy.encode("input.jpg", distance=0, lossless_jpeg=True, effort=8)
pixels = jxlpy.decode("input.jpg")
jxlpy.encode(pixels, distance=0, lossless_jpeg=False, effort=8)
```

For ordinary photos, JPEG transcode is usually much smaller than decoded-pixel
lossless JXL.

### Lossy Photos

Primary search:

- `distance`: target quality
- `effort`: `5/7/8/9`
- `epf`: `-1/1/2/3`
- `gaborish`: `-1/0/1`
- `faster_decoding`: if decode speed matters

Avoid palette/modular screenshot presets for normal photos unless metrics show
the image is actually synthetic.

### Multi-Frame / Layered Data

Do not rely on the encoder to discover inter-frame deltas from full-size frames.
In this project, `t1` showed:

```text
manual REPLACE+crop delta: 18732 bytes
full-size multi-frame:     47478 bytes
single-frame sum:          47494 bytes
```

Reference headers alone did not reduce the full-size case.

Useful external search:

- `reference`: `auto`, `previous`, `first`
- `min_crop_ratio`: `0.90`, `0.95`, `0.98`, `1.0`
- compare manual crop delta vs full-size if crop area is large
- then apply normal lossless/search presets to the resulting layers

For exact RGBA preservation, compare all channels, including RGB under
`alpha=0`, and prefer `REPLACE+crop` over `BLEND` tricks.

## Known Good Screenshot Preset

Equivalent `cjxl`:

```bash
cjxl content.png content.jxl -d 0 -e 9 -g 3 -I 100 -P 0 \
  --modular_palette_colors=10000 --patches 0 -Y 0
```

`jxlpy`:

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

In this project, `image.png` measured:

```text
PNG original:           366447 bytes
plain JXL e9 lossless:  425118 bytes
screenshot preset:      133369 bytes
```

Pixel roundtrip was exact.

## Why External Search Can Beat Defaults

The encoder has internal heuristics, and higher effort expands some of them, but
it does not fully search over all combinations of:

- modular group size
- predictor
- palette thresholds
- patch dictionary on/off
- progressive/responsive toggles
- color transforms
- JPEG transcode vs decoded-pixel lossless
- multi-frame crop/reference decisions

That is why a small external search can win on specialized image classes.

The practical approach is not to brute-force everything. Classify the image
first, then run a small preset set for that class.

## Practical Candidate Sets

### Fast Archive Lossless

```python
[
    dict(distance=0, effort=7, patches=False),
    dict(distance=0, effort=8, patches=False),
]
```

### Better Lossless Raster

```python
[
    dict(distance=0, effort=8, patches=False),
    dict(distance=0, effort=8, patches=True),
    dict(distance=0, effort=9, patches=False),
    dict(distance=0, effort=9, patches=True),
]
```

### Screenshot / Document Lossless

```python
[
    dict(distance=0, effort=9, patches=False),
    dict(distance=0, effort=9, patches=True),
    dict(
        distance=0,
        effort=9,
        modular_group_size=3,
        iterations=100,
        modular_predictor=0,
        modular_palette_colors=10000,
        patches=False,
        post_compact=0,
    ),
]
```

### Extreme Lossless

```python
[
    dict(distance=0, effort=9, patches=False),
    dict(distance=0, effort=9, patches=True),
    dict(distance=0, effort=10, patches=False),
    dict(distance=0, effort=10, patches=True),
    dict(
        distance=0,
        effort=9,
        modular_group_size=3,
        iterations=100,
        modular_predictor=0,
        modular_palette_colors=10000,
        patches=False,
        post_compact=0,
    ),
]
```

Effort 11 exists only with expert options and is generally too slow for routine
search.

## Current `jxlpy` Caveats

- `frame_settings` currently applies settings through the native shim's
  `JXLCompressParams` path. Most settings are global for the encode call, not
  per-frame custom schedules.
- `frame_index_box` is exposed as a raw setting, but `cjxl --frame_indexing`
  is a more convenient pattern parser. `jxlpy` does not yet provide an ergonomic
  multi-frame index-box API.
- Metadata/ICC override and stripping are not generic frame settings. They need
  dedicated byte/file payload plumbing.
- Decode-side thread count is not exposed yet.
- libjxl internals are evolving; this reference is for the vendored source in
  this repository, not a promise about every future upstream release.
