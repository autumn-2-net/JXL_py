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


if __name__ == "__main__":
    test_jpeg_reconstruct()
    test_analyze()
    print("Done.")
