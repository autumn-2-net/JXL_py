from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import jxlpy


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".jxl", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze exact multiframe diff/crop coverage.")
    parser.add_argument("folder", type=Path)
    parser.add_argument(
        "--reference", choices=["auto", "previous", "first", "none", "full"], default="auto"
    )
    parser.add_argument("--min-crop-ratio", type=float, default=0.98)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--compare-references", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _opaque_value(dtype: np.dtype):
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    return 1.0


def _convert_channels(arr: np.ndarray, channels: int) -> np.ndarray:
    if arr.shape[2] == channels:
        return arr
    h, w, source_channels = arr.shape
    if channels == 4:
        out = np.zeros((h, w, 4), dtype=arr.dtype)
        if source_channels == 1:
            out[:, :, :3] = np.repeat(arr, 3, axis=2)
        elif source_channels == 2:
            out[:, :, :3] = np.repeat(arr[:, :, :1], 3, axis=2)
            out[:, :, 3] = arr[:, :, 1]
            return out
        else:
            out[:, :, :3] = arr[:, :, :3]
        out[:, :, 3] = _opaque_value(arr.dtype)
        return out
    if channels == 3 and source_channels == 1:
        return np.repeat(arr, 3, axis=2)
    raise ValueError(f"cannot normalize {source_channels} channels to {channels}")


def load_frames(folder: Path, limit: int) -> tuple[list[Path], list[np.ndarray]]:
    files = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if limit > 0:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no supported images in {folder}")
    decoded = [jxlpy.decode(path, out="numpy") for path in files]
    dtype = decoded[0].dtype
    if any(frame.dtype != dtype for frame in decoded):
        raise ValueError("all frames must decode to the same dtype")
    source_channels = {frame.shape[2] for frame in decoded}
    if source_channels & {2, 4}:
        channels = 4
    elif 3 in source_channels:
        channels = 3
    else:
        channels = 1
    height = max(frame.shape[0] for frame in decoded)
    width = max(frame.shape[1] for frame in decoded)
    frames = []
    for frame in decoded:
        frame = _convert_channels(frame, channels)
        canvas = np.zeros((height, width, channels), dtype=dtype)
        canvas[: frame.shape[0], : frame.shape[1]] = frame
        frames.append(canvas)
    return files, frames


def main() -> None:
    args = parse_args()
    files, frames = load_frames(args.folder, args.limit)
    report = jxlpy.analyze_multiframe(
        frames,
        reference=args.reference,
        min_crop_ratio=args.min_crop_ratio,
    )
    comparison = None
    if args.compare_references:
        from jxlpy.multiframe import compare_reference_modes

        comparison = compare_reference_modes(
            frames, min_crop_ratio=args.min_crop_ratio
        )
    if args.json:
        payload = dict(report)
        payload["dtype"] = str(payload["dtype"])
        payload["files"] = [str(path) for path in files]
        if comparison is not None:
            payload["reference_comparison"] = comparison
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print(
        f"frames={report['num_frames']} canvas={report['canvas_size']} "
        f"reference={report['reference']} recommendation={report['recommendation']}"
    )
    print(
        f"avg changed={report['avg_changed_pct']:.2f}% "
        f"bbox={report['avg_bbox_pct']:.2f}% encoded={report['avg_encoded_pct']:.2f}%"
    )
    if comparison is not None:
        print(f"best reference: {comparison['best_reference']}")
        for mode, stats in comparison["modes"].items():
            print(
                f"  {mode:<8s} encoded={stats['avg_encoded_pct']:7.3f}% "
                f"bbox={stats['avg_bbox_pct']:7.3f}% "
                f"area={stats['total_encoded_area']}"
            )
    for path, stat in zip(files, report["frames"]):
        print(
            f"  {stat['index']:3d} {path.name:<32s} source={stat['source']:<8s} "
            f"changed={stat['changed_pct']:7.3f}% bbox={stat['bbox_pct']:7.3f}% "
            f"encoded={stat['encoded_pct']:7.3f}% crop={int(stat['use_crop'])}"
        )


if __name__ == "__main__":
    main()
