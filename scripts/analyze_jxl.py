from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jxlpy


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".jxl", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze image statistics and recommend lossless JXL candidates."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--mode", choices=["balanced", "archive"], default="archive")
    parser.add_argument("--jpeg-pixels", action="store_true")
    parser.add_argument("--max-sample-pixels", type=int, default=1_000_000)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def iter_images(inputs: list[Path]):
    seen: set[Path] = set()
    for item in inputs:
        paths = [item] if item.is_file() else sorted(item.rglob("*"))
        for path in paths:
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield path


def main() -> None:
    args = parse_args()
    records = []
    for path in iter_images(args.inputs):
        analysis = jxlpy.analyze_lossless(
            path,
            mode=args.mode,
            exact_jpeg=not args.jpeg_pixels,
            max_sample_pixels=args.max_sample_pixels,
        )
        metrics = analysis.metrics
        recommendation = analysis.recommendation
        record = {
            "path": str(path),
            **analysis.to_dict(),
        }
        records.append(record)
        if not args.json:
            names = ", ".join(item.name for item in recommendation.candidates)
            print(
                f"{path} profile={recommendation.profile} "
                f"entropy={metrics.entropy_gray:.3f} "
                f"flat4={metrics.flat4_pct:.1f}% "
                f"white={metrics.near_white_pct:.1f}% "
                f"unique/mpx={metrics.unique_per_mpx:.0f}"
            )
            print(f"  candidates: {names}")
            for candidate in recommendation.candidates:
                print(f"    {candidate.name}: {candidate.kwargs}")
    if not records:
        raise SystemExit("no supported image files found")
    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
