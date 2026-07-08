# JXL Compression Heuristics

This note records the current size/speed heuristics from local tests. It is not
a final encoder policy. Treat the thresholds as starting points that should be
updated when more samples are measured.

## Stable Conclusions

- For normal PNG/raster lossless, start with `cjxl input.png output.jxl -d 0 -e 8`.
- For final archive size, `-e 9` can be smaller, but it is often much slower for
  small gains.
- For lossy image output, `-e 5` is the current practical default. Higher efforts
  helped much less than lossless.
- For exact JPEG preservation, use JPEG lossless transcode:

```bash
cjxl input.jpg output.jxl --lossless_jpeg=1 -e 8
```

- JPEG transcode preserves the original JPEG bitstream. Pixel lossless does not.
- `--patches=1` has no observed effect on JPEG lossless transcode outputs.
- Do not rely on encoder patch auto-detection. In tested screenshot/text cases,
  auto was bit-identical to `--patches=0`, while forced `--patches=1` was smaller.
- Forced patches can be 3x to 15x slower. Only force them when the image looks
  structurally simple enough.

## Why PNG Can Win

The paper screenshot `9bde6aaf390109f97622d35ff1e900ee.png` is a good warning
case. It is mostly white document/text content, and the original PNG is already
well optimized.

Measured sizes:

| Method | Size | Ratio vs original | Time |
|---|---:|---:|---:|
| Original PNG | 323,776 | 100.00% | n/a |
| PIL PNG level 9 | 360,945 | 111.48% | 0.364s |
| PIL PNG level 9 optimize | 360,945 | 111.48% | 0.365s |
| JXL e8 auto/off | 462,301 | 142.78% | n/a |
| JXL e8 patches=1 | 302,272 | 93.36% | 2.772s |
| JXL e9 auto/off | 432,908 | 133.71% | n/a |
| JXL e9 patches=1 | 284,799 | 87.96% | 9.019s |

Interpretation:

- This does not look like a JXL decode bug.
- The original PNG was probably saved with better PNG filtering/optimization
  than Pillow's default encoder can reproduce.
- JXL without patches can be bad on document-like screenshots.
- Forced patches are useful here, but the CPU cost is high.

The likely reason is not semantic text information. PNG generally compresses
pixels, not OCR/text structure. A document screenshot benefits because the pixel
data has long white runs, repeated glyph-shaped byte patterns, low color
complexity, and scanlines that become small residuals after PNG row filters. The
Deflate backend can then search repeated byte sequences very effectively. The
fact that Pillow level 9 saved a larger PNG from the same pixels suggests the
original file used a better filter/zlib strategy than Pillow's default writer.

## Preflight Metrics

Compute these on decoded pixels. Sampling is acceptable for large images, but
exact archive decisions should use the full image.

| Metric | Meaning |
|---|---|
| `entropy_gray` | Shannon entropy of grayscale histogram, in bits. Lower means simpler. |
| `unique_per_mpx` | Unique RGBA or RGB colors per megapixel. Lower means palette-like or flat. |
| `flat4_pct` | Percentage of 4x4 blocks where all pixels are identical. |
| `near_white_pct` | Percentage of pixels close to white, useful for documents. |
| `edge_mean` | Mean neighbor difference. Lower usually means flatter regions. |

For exact RGBA processing, always include every channel in comparisons. Do not
ignore RGB values behind `alpha=0`, because training/archive workflows may care
about byte-exact invisible pixels.

Reference metric code:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ImageMetrics:
    width: int
    height: int
    channels: int
    entropy_gray: float
    unique_colors: int
    unique_per_mpx: float
    flat4_pct: float
    near_white_pct: float
    near_black_pct: float
    edge_mean: float


def load_rgba_for_metrics(path: Path) -> np.ndarray:
    img = Image.open(path)
    img.load()
    return np.asarray(img.convert("RGBA"), dtype=np.uint8)


def shannon_entropy_u8(values: np.ndarray) -> float:
    hist = np.bincount(values.reshape(-1), minlength=256)
    prob = hist[hist > 0].astype(np.float64)
    prob /= prob.sum()
    return float(-(prob * np.log2(prob)).sum())


def analyze_pixels(path: Path) -> ImageMetrics:
    arr = load_rgba_for_metrics(path)
    h, w, c = arr.shape

    rgb = arr[..., :3].astype(np.float32)
    gray = np.rint(
        rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    ).astype(np.uint8)
    entropy_gray = shannon_entropy_u8(gray)

    # Exact unique colors are useful for small/medium images. For huge images,
    # replace this with a deterministic sample to keep the preflight cheap.
    unique_colors = int(np.unique(arr.reshape(-1, c), axis=0).shape[0])
    megapixels = (w * h) / 1_000_000.0
    unique_per_mpx = unique_colors / megapixels if megapixels else 0.0

    hh = (h // 4) * 4
    ww = (w // 4) * 4
    if hh and ww:
        blocks = arr[:hh, :ww, :].reshape(hh // 4, 4, ww // 4, 4, c)
        flat_blocks = np.all(blocks == blocks[:, :1, :, :1, :], axis=(1, 3, 4))
        flat4_pct = float(flat_blocks.mean() * 100.0)
    else:
        flat4_pct = 0.0

    near_white_pct = float(np.all(arr[..., :3] >= 245, axis=-1).mean() * 100.0)
    near_black_pct = float(np.all(arr[..., :3] <= 16, axis=-1).mean() * 100.0)

    rgb_i = arr[..., :3].astype(np.int16)
    dx = np.abs(rgb_i[:, 1:, :] - rgb_i[:, :-1, :]).mean() if w > 1 else 0.0
    dy = np.abs(rgb_i[1:, :, :] - rgb_i[:-1, :, :]).mean() if h > 1 else 0.0
    edge_mean = float((dx + dy) / 2.0)

    return ImageMetrics(
        width=w,
        height=h,
        channels=c,
        entropy_gray=entropy_gray,
        unique_colors=unique_colors,
        unique_per_mpx=unique_per_mpx,
        flat4_pct=flat4_pct,
        near_white_pct=near_white_pct,
        near_black_pct=near_black_pct,
        edge_mean=edge_mean,
    )
```

For exact encode/decode validation, do not use this Pillow conversion as the
only source of truth. Use the wrapper/native decoder and compare the original
raw channels, especially when hidden RGB behind alpha matters.

## Patch Decision

Use forced patches for lossless raster images when at least one of these is true:

- `entropy_gray < 3.0`
- `unique_per_mpx < 5,000`
- `flat4_pct > 50`
- `near_white_pct > 50`

Use the normal no-patch path when the image looks natural or illustration-like:

- `entropy_gray > 5.5`
- `unique_per_mpx > 20,000`
- `flat4_pct < 25`
- no dominant background

Current command choices:

```bash
# Fast/practical lossless
cjxl input.png output.jxl -d 0 -e 8 --patches=0

# Archive/document/text/screenshot lossless
cjxl input.png output.jxl -d 0 -e 8 --patches=1

# Maximum size pressure, slow
cjxl input.png output.jxl -d 0 -e 9 --patches=1
```

If build time is acceptable and the image matches the patch candidate rules, the
most reliable size policy is to encode both `--patches=0` and `--patches=1`, then
keep the smaller output. If build time is not acceptable, only force patches for
document/text/screenshot-like images.

## JPEG Decision

Default JPEG path:

```bash
cjxl input.jpg output.jxl --lossless_jpeg=1 -e 8
```

Use `-e 9` only for archive mode. Local results showed slightly smaller output
with much more CPU.

Pixel-lossless JPEG path:

```bash
cjxl input.jpg output.jxl --lossless_jpeg=0 -d 0 -e 8
```

Pixel-lossless can be smaller than JPEG transcode only for unusually simple JPEG
content, such as mostly blank/generated/diff frames. It is often much larger for
ordinary photos or illustrations.

Try pixel-lossless only when byte-exact JPEG reconstruction is not required and
the decoded image matches a "simple" profile:

- `entropy_gray < 5.0`
- and one of:
  - `flat4_pct > 35`
  - `unique_per_mpx < 10,000`
  - `near_white_pct > 25`

If those rules match and CPU budget allows it, run both:

```bash
cjxl input.jpg transcode.jxl --lossless_jpeg=1 -e 8
cjxl input.jpg pixel.jxl --lossless_jpeg=0 -d 0 -e 8
```

Then keep the smaller file only if JPEG byte-exact preservation is not needed.

## Multi-Frame Diff Notes

For exact multi-frame/layer storage, prefer `REPLACE + crop` over `BLEND`.
Transparent-black tricks can work for opaque images, but they are risky for
semi-transparent pixels, premultiplied-alpha assumptions, and invisible RGB.

Diff crop rules:

- Compare all channels, including RGB behind `alpha=0`.
- Use the full first frame.
- For later frames, store only the bounding box of changed pixels.
- Default reference should be the previous reconstructed frame.
- Optionally test first-frame reference for long sequences where every frame is
  a small edit over a fixed base.
- For exact restore, decoding must reconstruct the full frame by applying the
  crops in order.

This is separate from libjxl patches. Current libjxl does not aggressively
optimize inter-frame differences automatically, so the wrapper should own this
logic when exact multi-frame compression matters.

## Proposed Encoder Policy

Pseudo-code:

```text
if input is JPEG and exact_jpeg_required:
    encode JPEG transcode with lossless_jpeg=1, effort=8
    use effort=9 only for archive mode
    return

decode or sample pixels
metrics = analyze_pixels(pixels)

if input is JPEG and not exact_jpeg_required:
    if is_simple_jpeg_candidate(metrics):
        encode JPEG transcode e8
        encode pixel-lossless e8, maybe with patches if document-like
        keep smaller
    else:
        encode JPEG transcode e8
    return

if lossless raster:
    if is_patch_candidate(metrics):
        if archive_or_best_size:
            encode e8 patches=0 and e8 patches=1, keep smaller
        else:
            encode e8 patches=1
    else:
        encode e8 patches=0
    return

if lossy raster:
    encode distance target with effort=5
    use effort=7/8 only when a measured quality/size target requires it
```

## Screenshot/Document Modular Preset

Community-provided settings for screenshots/documents with repeated glyphs,
flat regions, and limited palettes:

```bash
cjxl content.png content.jxl -d 0 -e 9 -g 3 -I 100 -P 0 \
  --modular_palette_colors=10000 --patches 0 -Y 0
```

Python equivalent:

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

Notes:

- `-g 3` means 1024x1024 modular groups.
- `-I 100` uses all pixels for MA-tree learning.
- `-P 0` forces the zero predictor; this can help some text/palette images, but
  it should not be assumed best for photos or illustrations.
- `-Y 0` disables local post-compaction.
- `--patches 0` can beat patch detection when patch search overhead or chosen
  patches are counterproductive.

Treat this as a candidate for document-like screenshots, not a universal
default. Compare against at least plain `distance=0, effort=9` for archive
mode.

Current helper predicates:

```text
is_patch_candidate(m):
    return (
        m.entropy_gray < 3.0
        or m.unique_per_mpx < 5000
        or m.flat4_pct > 50
        or m.near_white_pct > 50
    )

is_simple_jpeg_candidate(m):
    return (
        m.entropy_gray < 5.0
        and (
            m.flat4_pct > 35
            or m.unique_per_mpx < 10000
            or m.near_white_pct > 25
        )
    )
```

Runnable Python version:

```python
def is_patch_candidate(m: ImageMetrics) -> bool:
    return (
        m.entropy_gray < 3.0
        or m.unique_per_mpx < 5000
        or m.flat4_pct > 50.0
        or m.near_white_pct > 50.0
    )


def is_simple_jpeg_candidate(m: ImageMetrics) -> bool:
    return (
        m.entropy_gray < 5.0
        and (
            m.flat4_pct > 35.0
            or m.unique_per_mpx < 10000
            or m.near_white_pct > 25.0
        )
    )
```

## Benchmark Code

The batch benchmark script in this repo uses this command wrapper:

```python
from pathlib import Path
import subprocess
import time


def run_cjxl(
    cjxl: Path,
    source: Path,
    output: Path,
    args: tuple[str, ...],
    timeout: int = 900,
) -> float:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(cjxl), str(source), str(output), *args, "--quiet"]
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    seconds = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(
            "cjxl failed\n"
            + " ".join(cmd)
            + "\nstdout:\n"
            + proc.stdout
            + "\nstderr:\n"
            + proc.stderr
        )
    return seconds
```

Useful commands:

```powershell
# PNG lossless, compare auto/off/on patches at effort 8.
C:\Users\autumn\.conda\envs\py10\python.exe scripts\benchmark_high_effort_images.py `
  --root test_img `
  --diff-root test_img\mt_lay `
  --efforts 8 `
  --patches auto off on `
  --jpeg-modes both `
  --reuse
```

The Pillow PNG recompression check used for the paper screenshot:

```python
from pathlib import Path
import time

from PIL import Image


def pil_png_level9(source: Path, output: Path, optimize: bool) -> dict[str, object]:
    img = Image.open(source)
    img.load()
    output.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    img.save(output, format="PNG", compress_level=9, optimize=optimize)
    seconds = time.perf_counter() - start

    test = Image.open(output)
    test.load()
    exact_pixels = list(img.getdata()) == list(test.getdata())

    original_bytes = source.stat().st_size
    encoded_bytes = output.stat().st_size
    return {
        "source": str(source),
        "output": str(output),
        "optimize": optimize,
        "original_bytes": original_bytes,
        "encoded_bytes": encoded_bytes,
        "ratio": encoded_bytes / original_bytes,
        "seconds": seconds,
        "exact_pixels": exact_pixels,
    }
```

## Local Reference Results

| Sample | Best relevant result | Notes |
|---|---:|---|
| `3dd595a30a79d3ba6d107fa6f60c5d3d.png` | e8 p1: 114,116 vs e8 p0: 133,825 | screenshot, patches useful |
| `7dba5bd64be12f08db25c3a78bcb80d7.png` | e8 p1: 12,139 vs e8 p0: 14,690 | text screenshot, patches useful |
| `test2/2026.6 ... .png` | e8 p1: 2,059,867 vs e8 p0: 2,102,705 | illustration, tiny patch gain, slow |
| `test2/146644643_p1.jpg` | transcode e8: 1,031,345 vs pixel e8 p0: 2,120,314 | normal JPEG, use transcode |
| `test_img/mt_lay/t3/p0.jpg` | pixel e8 p1: 199,847 vs transcode e8: 356,431 | simple JPEG, pixel path can win |
| `9bde6aaf390109f97622d35ff1e900ee.png` | e9 p1: 284,799 vs PNG: 323,776 | paper screenshot, PNG still beats JXL no-patch |

## PNG Recompression Probe

The 2026-07-06 PNG probe compared Pillow PNG level 9, JXL effort 9 without
patches, and JXL effort 9 with patches for screenshot/document samples. Outputs
are under:

```text
out/test-run/png_pil_jxl_e9_compare/
```

Main results:

| Sample | Original PNG | PIL level9 opt | JXL e9 p0 | JXL e9 p1 | Interpretation |
|---|---:|---:|---:|---:|---|
| `1eb7...png` | 591,394 | 552,363 | 469,579 | 450,992 | new paper screenshot; JXL p0 already wins, patch gain is small but slow |
| `9bde...png` | 323,776 | 360,945 | 432,908 | 284,799 | old paper screenshot; PNG beats JXL p0, forced patches are needed |
| `3dd5...png` | 208,958 | 312,928 | 131,375 | 108,465 | UI screenshot; JXL wins, patches help |
| `7dba...png` | 40,671 | 54,060 | 14,136 | 11,609 | text screenshot; JXL wins strongly, patches help |
| `image.png` | 366,447 | not retested | 425,118 | 362,093 | dense paper text; PNG is extremely competitive, p0 loses |
| `wallhaven-n6qzmx.png` | 998,338 | 921,114 | 471,747 | not retested here | normal PNG; JXL wins strongly |
| `test2` illustration PNG | 3,152,128 | 3,120,503 | 2,098,739 | not retested here | illustration; JXL wins, patch not worth default CPU |

Timing highlights:

- `1eb7...png`: PIL opt `0.506s`, JXL e9 p0 `0.989s`, JXL e9 p1 `13.999s`.
- `9bde...png`: PIL opt `0.402s`, JXL e9 p0 `0.620s`, JXL e9 p1 `9.278s`.
- `3dd5...png`: PIL opt `0.272s`, JXL e9 p0 `0.298s`, JXL e9 p1 `3.337s`.
- `7dba...png`: PIL opt `0.077s`, JXL e9 p0 `0.651s`, JXL e9 p1 `0.405s`.
- `image.png`: JXL e9 p0 `0.660s`, JXL e9 p1 `8.162s`.

This supports the current policy: do not force patches for every PNG, but do
force or at least trial-run patches for document/text/screenshot archive mode.

### PNG Chunk And Alpha Check

The document screenshots do not contain semantic text metadata. They have normal
PNG chunks such as `IHDR`, `sRGB`, `gAMA`, `pHYs`, `IDAT`, and `IEND`; no
`tEXt`/`iTXt` OCR-like side data was present in these samples.

Several screenshot PNGs are RGBA with alpha always equal to 255:

| Sample | PNG color type | Alpha |
|---|---|---|
| `1eb7...png` | 8-bit RGBA | 100% opaque |
| `9bde...png` | 8-bit RGBA | 100% opaque |
| `3dd5...png` | 8-bit RGBA | 100% opaque |
| `7dba...png` | 8-bit RGBA | 100% opaque |
| `wallhaven-n6qzmx.png` | 8-bit RGBA | 100% opaque |
| `image.png` | 8-bit RGBA | 100% opaque |

Dropping the all-255 alpha channel can be a valid visual-equivalence
optimization, but only when exact channel preservation is not required.

RGB conversion probe:

| Sample | RGB PNG level9 opt | RGB JXL e9 p0 | RGB JXL e9 p1 | Conclusion |
|---|---:|---:|---:|---|
| `1eb7...png` | 488,696 | 465,967 | 445,861 | alpha removal helps PNG more than JXL, JXL still wins |
| `9bde...png` | 319,814 | 430,945 | 285,966 | alpha is not the main p0 problem; patches still fix it |
| `3dd5...png` | 283,343 | 130,228 | 108,329 | RGB PNG gets worse; JXL still wins |
| `7dba...png` | 49,063 | 15,918 | 11,503 | RGB PNG gets worse; JXL patch is best |
| `wallhaven-n6qzmx.png` | 816,749 | 471,715 | 443,312 | RGB PNG and JXL both improve |
| `image.png` | 311,010 | 423,817 | 356,652 | RGB PNG beats JXL; visual-equivalent alpha drop matters |

So the old paper screenshot is likely a real no-patch JXL lossless weak case:
PNG row filters plus Deflate match repeated low-color text/white-background
bytes better than JXL's default no-patch lossless path. Forced patches change
the result enough to beat the original PNG.

`image.png` strengthens this point. It has `entropy_gray=1.560`,
`unique_per_mpx=2534.9`, `flat4_pct=72.67`, `near_white_pct=86.04`, and an
all-255 alpha channel. Even with those patch-friendly metrics, JXL e9 p0 is
larger than the original PNG and JXL e9 p1 only barely beats the exact RGBA PNG.
If visual equivalence is enough and the opaque alpha channel can be dropped, an
optimized RGB PNG is smaller than JXL for this sample.
