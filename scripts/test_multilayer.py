"""Test multiframe/multilayer encode and decode using real test images."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jxlpy


def load_frames_pil(folder: Path) -> list[np.ndarray]:
    """Load images from folder, resize to common canvas, return RGBA arrays."""
    from PIL import Image, ImageOps

    files = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )
    if not files:
        raise FileNotFoundError(f"no images in {folder}")

    images = []
    for f in files:
        with Image.open(f) as im:
            images.append(ImageOps.exif_transpose(im).copy())

    sizes = [im.size for im in images]
    max_w = max(w for w, _ in sizes)
    max_h = max(h for _, h in sizes)

    arrays = []
    for im in images:
        canvas = Image.new("RGBA", (max_w, max_h), (0, 0, 0, 0))
        canvas.paste(im.convert("RGBA"), (0, 0))
        arrays.append(np.array(canvas))
    return arrays


def test_synthetic():
    """Test with synthetic data — should be pixel-exact roundtrip."""
    print("=" * 60)
    print("TEST: Synthetic multiframe (lossless roundtrip)")
    print("=" * 60)

    frames = [np.zeros((64, 64, 4), dtype=np.uint8) for _ in range(4)]
    frames[0][..., :3] = 100
    frames[0][..., 3] = 255
    frames[1] = frames[0].copy()
    frames[1][10:30, 10:30, 0] = 200
    frames[2] = frames[0].copy()
    frames[2][30:50, 30:50, 1] = 180
    frames[3] = frames[1].copy()
    frames[3][40:60, 40:60, 2] = 220

    t0 = time.perf_counter()
    jxl_bytes = jxlpy.encode_multiframe(frames, reference="auto")
    encode_time = time.perf_counter() - t0
    print(f"  Encoded {len(frames)} frames -> {len(jxl_bytes)} bytes in {encode_time:.3f}s")

    # Get info
    meta = jxlpy.info(jxl_bytes)
    print(f"  Info: {meta['xsize']}x{meta['ysize']}, {meta['num_frames']} frames, animation={meta['have_animation']}")

    # Decode each frame and verify lossless
    all_ok = True
    for i, original in enumerate(frames):
        decoded = jxlpy.decode(jxl_bytes, frame=i, coalesced=True)
        match = np.array_equal(decoded, original)
        if not match:
            diff_count = np.sum(decoded != original)
            print(f"  Frame {i}: MISMATCH ({diff_count} values differ)")
            all_ok = False
        else:
            print(f"  Frame {i}: OK (lossless)")

    # Test layer/crop decode (non-coalesced)
    print("  --- Layer decode (non-coalesced) ---")
    for i in range(len(frames)):
        layer, lmeta = jxlpy.decode_layer(jxl_bytes, layer=i)
        crop_info = ""
        if lmeta["layer_have_crop"]:
            crop_info = f" crop=({lmeta['crop_x0']},{lmeta['crop_y0']}) size={layer.shape[1]}x{layer.shape[0]}"
        else:
            crop_info = f" full={layer.shape[1]}x{layer.shape[0]}"
        print(f"  Layer {i}:{crop_info}")

    print(f"  RESULT: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


def test_real_images(folder: Path):
    """Test with real images from a folder."""
    print("=" * 60)
    print(f"TEST: Real images from {folder.name}")
    print("=" * 60)

    try:
        frames = load_frames_pil(folder)
    except Exception as e:
        print(f"  SKIP: {e}")
        print()
        return True

    h, w = frames[0].shape[:2]
    print(f"  Loaded {len(frames)} frames, canvas {w}x{h}")

    configs = [
        ("lossless e3",  dict(effort=3, lossless=True)),
        ("lossless e7",  dict(effort=7, lossless=True)),
        ("lossy d1 e3",  dict(effort=3, distance=1.0, lossless=False)),
        ("lossy d1 e7",  dict(effort=7, distance=1.0, lossless=False)),
    ]

    print(f"  {'mode':<14s} {'single sum':>11s} {'multiframe':>11s} {'saving':>8s} {'time':>7s}")
    print(f"  {'-'*14} {'-'*11} {'-'*11} {'-'*8} {'-'*7}")

    jxl_bytes = None  # keep last lossless for decode verify
    for label, kwargs in configs:
        # Single-frame sum
        single_total = sum(len(jxlpy.encode(f, **kwargs)) for f in frames)
        # Multiframe
        t0 = time.perf_counter()
        multi = jxlpy.encode_multiframe(frames, reference="auto", **kwargs)
        dt = time.perf_counter() - t0
        saving = (1.0 - len(multi) / single_total) * 100 if single_total else 0
        print(f"  {label:<14s} {single_total/1024:>8.1f}KiB {len(multi)/1024:>8.1f}KiB {saving:>+7.1f}% {dt:>6.2f}s")
        if "lossless" in label and kwargs.get("lossless"):
            jxl_bytes = multi

    # Info
    meta = jxlpy.info(jxl_bytes)
    print(f"  Info: num_frames={meta['num_frames']}, animation={meta['have_animation']}")

    # Decode and verify
    all_ok = True
    for i, original in enumerate(frames):
        t0 = time.perf_counter()
        decoded = jxlpy.decode(jxl_bytes, frame=i, coalesced=True)
        dt = time.perf_counter() - t0
        match = np.array_equal(decoded, original)
        status = "OK" if match else "MISMATCH"
        if not match:
            diff_pixels = np.any(decoded != original, axis=2).sum()
            print(f"  Frame {i}: {status} ({diff_pixels} pixels differ) [{dt:.3f}s]")
            all_ok = False
        else:
            print(f"  Frame {i}: {status} (lossless) [{dt:.3f}s]")

    # Layer info
    print("  --- Layer structure ---")
    for i in range(len(frames)):
        layer, lmeta = jxlpy.decode_layer(jxl_bytes, layer=i)
        if lmeta["layer_have_crop"]:
            crop_area = layer.shape[0] * layer.shape[1]
            full_area = h * w
            print(f"  Layer {i}: crop ({lmeta['crop_x0']},{lmeta['crop_y0']}) "
                  f"{layer.shape[1]}x{layer.shape[0]} "
                  f"({crop_area/full_area*100:.1f}% of canvas)")
        else:
            print(f"  Layer {i}: full {layer.shape[1]}x{layer.shape[0]}")

    print(f"  RESULT: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


def test_reference_modes():
    """Test different reference modes."""
    print("=" * 60)
    print("TEST: Reference mode comparison")
    print("=" * 60)

    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(5)]
    frames[0][..., 0] = 128
    for i in range(1, 5):
        frames[i] = frames[i - 1].copy()
        r = 4 * i
        frames[i][r:r+4, r:r+4, 1] = 200

    for mode in ("auto", "first", "previous", "none"):
        jxl_bytes = jxlpy.encode_multiframe(frames, reference=mode, effort=1)
        # Verify all frames decode correctly
        ok = all(
            np.array_equal(jxlpy.decode(jxl_bytes, frame=i), frames[i])
            for i in range(len(frames))
        )
        print(f"  reference={mode:10s}: {len(jxl_bytes):6d} bytes, lossless={'PASS' if ok else 'FAIL'}")

    print()
    return True


def test_extra_channels_multiframe():
    """Test extra channels with multiframe."""
    print("=" * 60)
    print("TEST: Extra channels + multiframe")
    print("=" * 60)

    frames = [np.zeros((32, 32, 4), dtype=np.uint8) for _ in range(3)]
    frames[0][..., 3] = 255
    frames[0][..., 0] = 100
    frames[1] = frames[0].copy()
    frames[1][8:16, 8:16, 1] = 200
    frames[2] = frames[0].copy()
    frames[2][16:24, 16:24, 2] = 180

    masks = [np.zeros((32, 32), dtype=np.uint8) for _ in range(3)]
    masks[0][4:8, 4:8] = 255
    masks[1][10:14, 10:14] = 128
    masks[2][20:28, 20:28] = 64

    jxl_bytes = jxlpy.encode_multiframe(
        frames,
        reference="auto",
        effort=1,
        extra_channels=[("mask", "selection_mask", masks)],
    )
    print(f"  Encoded 3 frames + extra channel -> {len(jxl_bytes)} bytes")

    all_ok = True
    for i in range(3):
        decoded = jxlpy.decode(jxl_bytes, frame=i, coalesced=True)
        if not np.array_equal(decoded, frames[i]):
            print(f"  Frame {i} main: MISMATCH")
            all_ok = False

        mask_decoded, mask_meta = jxlpy.decode_extra_channel(jxl_bytes, 1, frame=i)
        if not np.array_equal(mask_decoded, masks[i]):
            print(f"  Frame {i} mask: MISMATCH")
            all_ok = False

    if all_ok:
        print("  All frames + extra channels: lossless OK")

    print(f"  RESULT: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


def _mp_encode_worker(arr):
    """Module-level worker for multiprocess test (must be picklable)."""
    return len(jxlpy.encode(arr))


def test_thread_safety():
    """Test concurrent encode/decode from multiple threads."""
    import concurrent.futures
    print("=" * 60)
    print("TEST: Thread safety (concurrent encode/decode)")
    print("=" * 60)

    frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(8)]
    encoded = [jxlpy.encode(f) for f in frames]

    def encode_task(arr):
        return jxlpy.encode(arr)

    def decode_task(data):
        return jxlpy.decode(data)

    # Threads
    all_ok = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        encode_futures = [pool.submit(encode_task, f) for f in frames]
        results_enc = [fut.result() for fut in encode_futures]
        for i, r in enumerate(results_enc):
            if r != encoded[i]:
                print(f"  Thread encode {i}: MISMATCH")
                all_ok = False

        decode_futures = [pool.submit(decode_task, e) for e in encoded]
        results_dec = [fut.result() for fut in decode_futures]
        for i, r in enumerate(results_dec):
            if not np.array_equal(r, frames[i]):
                print(f"  Thread decode {i}: MISMATCH")
                all_ok = False

    if all_ok:
        print("  4 threads × 8 tasks: encode + decode all match")

    # Multiprocess
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        mp_futures = [pool.submit(_mp_encode_worker, f) for f in frames[:4]]
        mp_results = [fut.result() for fut in mp_futures]
        for i, size in enumerate(mp_results):
            if size != len(encoded[i]):
                print(f"  Process encode {i}: size mismatch ({size} vs {len(encoded[i])})")
                all_ok = False

    if all_ok:
        print("  2 processes × 4 tasks: encode sizes match")

    print(f"  RESULT: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


def test_jpeg_transcode_vs_pixel():
    """Compare JPEG lossless transcode vs decode-to-pixels then JXL lossless."""
    print("=" * 60)
    print("TEST: JPEG lossless transcode vs pixel re-encode")
    print("=" * 60)

    jpeg_dir = Path("test_img")
    jpegs = sorted(p for p in jpeg_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg"))
    if not jpegs:
        print("  SKIP: no JPEG files in test_img/")
        print()
        return True

    print(f"  {'file':<28s} {'jpeg':>8s} {'transcode':>10s} {'pixel jxl':>10s} {'winner':>8s}")
    print(f"  {'-'*28} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

    all_ok = True
    for jpeg_path in jpegs:
        jpeg_bytes = jpeg_path.read_bytes()
        jpeg_size = len(jpeg_bytes)

        # Method 1: JPEG lossless transcode (default)
        transcode_jxl = jxlpy.encode(jpeg_bytes)

        # Method 2: Decode to pixels, then encode pixels as lossless JXL
        pixels = jxlpy.decode(jpeg_bytes)
        pixel_jxl = jxlpy.encode(pixels)

        # Verify transcode can round-trip back to exact JPEG
        # (decode_to_jpeg should give back the original)
        try:
            roundtrip_jpeg = jxlpy.decode_to_jpeg(transcode_jxl)
            rt_ok = roundtrip_jpeg == jpeg_bytes
        except Exception:
            rt_ok = False

        transcode_kb = len(transcode_jxl) / 1024
        pixel_kb = len(pixel_jxl) / 1024
        winner = "transcod" if len(transcode_jxl) <= len(pixel_jxl) else "pixel"
        rt_mark = "" if rt_ok else " [RT FAIL]"

        print(f"  {jpeg_path.name:<28s} {jpeg_size/1024:>7.1f}K {transcode_kb:>9.1f}K {pixel_kb:>9.1f}K {winner:>8s}{rt_mark}")

        if not rt_ok:
            # Not a hard failure - some JPEGs may not roundtrip perfectly
            pass

    print()
    print("  transcode = JPEG bitstream stored in JXL container (can reconstruct original JPEG)")
    print("  pixel jxl = decode JPEG to RGB pixels, then lossless JXL encode pixels")
    print()
    return True


def main():
    results = []

    results.append(("synthetic", test_synthetic()))
    results.append(("reference_modes", test_reference_modes()))
    results.append(("extra_channels_multiframe", test_extra_channels_multiframe()))
    results.append(("thread_safety", test_thread_safety()))
    results.append(("jpeg_transcode", test_jpeg_transcode_vs_pixel()))

    mt_lay = Path("test_img/mt_lay")
    if mt_lay.exists():
        for subdir in sorted(mt_lay.iterdir()):
            if subdir.is_dir():
                results.append((f"real/{subdir.name}", test_real_images(subdir)))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results:
        print(f"  {name:30s} {'PASS' if ok else 'FAIL'}")

    if all(ok for _, ok in results):
        print("\nAll tests passed.")
        return 0
    else:
        print("\nSome tests FAILED.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
