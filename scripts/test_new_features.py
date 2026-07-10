"""Quick test for reconstruct_jpeg and analyze_multiframe."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jxlpy


def test_jpeg_reconstruct():
    print("=== JPEG Reconstruction ===")
    jpegs = sorted(Path("test_img").glob("*.jpg"))
    for jp in jpegs[:5]:
        original = jp.read_bytes()
        jxl = jxlpy.encode(original)
        roundtrip = jxlpy.decode_to_jpeg(jxl)
        match = roundtrip == original
        print(f"  {jp.name:<28s} {len(original)/1024:>7.1f}K -> {len(jxl)/1024:>7.1f}K  RT={'OK' if match else 'FAIL'}")
    print()


def test_analyze():
    print("=== Analyze Multiframe ===")
    from PIL import Image, ImageOps

    for folder in sorted(Path("test_img/mt_lay").iterdir()):
        if not folder.is_dir():
            continue
        files = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        )
        if not files:
            continue

        images = []
        for f in files:
            with Image.open(f) as im:
                images.append(ImageOps.exif_transpose(im).copy())
        max_w = max(im.size[0] for im in images)
        max_h = max(im.size[1] for im in images)
        frames = []
        for im in images:
            canvas = Image.new("RGBA", (max_w, max_h), (0, 0, 0, 0))
            canvas.paste(im.convert("RGBA"), (0, 0))
            frames.append(np.array(canvas))

        result = jxlpy.analyze_multiframe(frames)
        print(f"  {folder.name}: {result['num_frames']} frames, "
              f"canvas={result['canvas_size']}")
        print(f"    avg_bbox={result['avg_bbox_pct']:.1f}%, "
              f"avg_changed={result['avg_changed_pct']:.1f}%")
        print(f"    recommendation: {result['recommendation']}")
        for s in result["frames"]:
            print(f"      frame {s['index']}: bbox={s['bbox_pct']:.1f}%, "
                  f"changed={s['changed_pct']:.1f}%")
        print()


def test_analyze_auto_alignment():
    print("=== Analyze Auto Alignment ===")

    frames = [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(4)]
    frames[1] = frames[0].copy()
    frames[1][2:4, 2:4, 0] = 80
    frames[2] = frames[1].copy()
    frames[2][5:7, 5:7, 1] = 120
    frames[3] = frames[0].copy()
    frames[3][12:14, 12:14, 2] = 200

    masks = [np.zeros((16, 16), dtype=np.uint8) for _ in range(4)]
    masks[2][8:10, 8:10] = 255

    extras = [("mask", "selection_mask", masks)]
    report = jxlpy.analyze_multiframe(frames, extra_channels=extras, reference="auto")
    jxl = jxlpy.encode_multiframe(
        frames,
        extra_channels=extras,
        reference="auto",
        effort=1,
    )

    expected_sources = ["none", "first", "previous", "first"]
    sources = [frame["source"] for frame in report["frames"]]
    if sources != expected_sources:
        raise AssertionError(f"unexpected auto sources: {sources}")

    for stat in report["frames"]:
        layer, meta = jxlpy.decode_layer(jxl, layer=stat["index"])
        if bool(meta["layer_have_crop"]) != bool(stat["use_crop"]):
            raise AssertionError(f"layer {stat['index']} crop mismatch")
        if stat["use_crop"]:
            if meta["crop_x0"] != stat["crop_x0"] or meta["crop_y0"] != stat["crop_y0"]:
                raise AssertionError(f"layer {stat['index']} crop offset mismatch")
            if layer.shape[1] != stat["crop_xsize"] or layer.shape[0] != stat["crop_ysize"]:
                raise AssertionError(f"layer {stat['index']} crop size mismatch")

    print("  sources:", ", ".join(sources))
    print("  RESULT: PASS")
    print()


def test_cjxl_modular_aliases():
    print("=== cjxl Modular Aliases ===")

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[2:14, 2:14] = (255, 255, 255)
    img[:, 7:9] = (0, 0, 0)

    alias = jxlpy.encode(
        img,
        distance=0,
        effort=1,
        modular_group_size=3,
        iterations=100,
        modular_predictor=0,
        modular_palette_colors=10000,
        patches=False,
        post_compact=0,
    )
    explicit = jxlpy.encode(
        img,
        distance=0,
        effort=1,
        modular_group_size=3,
        modular_ma_tree_learning_percent=100,
        modular_predictor=0,
        modular_palette_colors=10000,
        patches=False,
        modular_channel_colors_group_percent=0,
    )
    if alias != explicit:
        raise AssertionError("alias options did not match explicit options")
    if not np.array_equal(jxlpy.decode(alias), img):
        raise AssertionError("alias-encoded image did not roundtrip")

    via_dict = jxlpy.encode(
        img,
        encoder_options={
            "distance": 0,
            "effort": 1,
            "modular_group_size": 3,
            "iterations": 100,
            "modular_predictor": 0,
            "modular_palette_colors": 10000,
            "patches": False,
            "post_compact": 0,
        },
    )
    if via_dict != alias:
        raise AssertionError("encoder_options dict did not match keyword options")

    try:
        jxlpy.encode(img, iterations=50, modular_ma_tree_learning_percent=100)
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting alias options should fail")

    print(f"  encoded: {len(alias)} bytes")
    print("  RESULT: PASS")
    print()


def test_screenshot_heuristic():
    print("=== Screenshot Heuristic ===")

    screenshot = np.full((256, 384, 4), 255, dtype=np.uint8)
    screenshot[20:236:18, 32:352] = (24, 24, 24, 255)
    screenshot[28:232:18, 32:220] = (128, 128, 128, 255)
    analysis = jxlpy.analyze_lossless(screenshot, source_format="png")
    names = [candidate.name for candidate in analysis.recommendation.candidates]
    if analysis.recommendation.profile != "simple_screenshot":
        raise AssertionError("simple screenshot profile was not detected")
    if names != ["screenshot_modular_e9", "default_e9"]:
        raise AssertionError(f"unexpected screenshot candidates: {names}")

    preset = analysis.recommendation.candidates[0].kwargs
    expected = {
        "lossless": True,
        "distance": 0.0,
        "effort": 9,
        "modular": 1,
        "modular_group_size": 3,
        "modular_predictor": 0,
        "modular_palette_colors": 10_000,
        "iterations": 100,
        "patches": False,
        "post_compact": 0,
    }
    if preset != expected:
        raise AssertionError(f"unexpected screenshot preset: {preset}")

    balanced = jxlpy.analyze_lossless(
        screenshot, source_format="png", mode="balanced"
    )
    balanced_names = [
        candidate.name for candidate in balanced.recommendation.candidates
    ]
    if balanced_names != ["default_e8"]:
        raise AssertionError(f"unexpected balanced candidates: {balanced_names}")

    rng = np.random.default_rng(1234)
    photo_like = rng.integers(0, 256, (256, 384, 3), dtype=np.uint8)
    general = jxlpy.analyze_lossless(photo_like, source_format="png")
    if general.recommendation.profile != "general":
        raise AssertionError("high-entropy raster was misclassified as a screenshot")

    print("  candidates:", ", ".join(names))
    print("  RESULT: PASS")
    print()


def test_frame_settings_passthrough():
    print("=== Raw Frame Settings Passthrough ===")

    img = np.zeros((16, 16, 3), dtype=np.uint8)
    try:
        jxl = jxlpy.encode(
            img,
            distance=0,
            effort=1,
            frame_settings={"use_full_image_heuristics": 0},
        )
    except RuntimeError as exc:
        if "requires rebuilding" in str(exc):
            print("  SKIP: native shim has not been rebuilt for passthrough")
            print()
            return
        raise
    if not np.array_equal(jxlpy.decode(jxl), img):
        raise AssertionError("frame_settings-encoded image did not roundtrip")

    print(f"  encoded: {len(jxl)} bytes")
    print("  RESULT: PASS")
    print()


if __name__ == "__main__":
    test_jpeg_reconstruct()
    test_analyze()
    test_analyze_auto_alignment()
    test_cjxl_modular_aliases()
    test_screenshot_heuristic()
    test_frame_settings_passthrough()
    print("Done.")
