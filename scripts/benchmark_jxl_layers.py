from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageOps, PngImagePlugin


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
JPEG_SUFFIXES = {".jpg", ".jpeg"}


@dataclass
class MethodResult:
    group: str
    method: str
    bytes: int
    seconds: float
    output: str


def fmt_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}" if unit != "B" else f"{value} B"
        size /= 1024
    return f"{value} B"


def pct_delta(value: int, base: int) -> str:
    if base == 0:
        return "n/a"
    return f"{(value / base - 1.0) * 100.0:+.2f}%"


def run(cmd: list[str], timeout: int = 900) -> float:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed\n"
            + " ".join(cmd)
            + "\nstdout:\n"
            + proc.stdout
            + "\nstderr:\n"
            + proc.stderr
        )
    return elapsed


def find_groups(input_root: Path) -> dict[str, list[Path]]:
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    direct = sorted(
        p for p in input_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if direct:
        return {input_root.name: direct}
    groups: dict[str, list[Path]] = {}
    for child in sorted(p for p in input_root.iterdir() if p.is_dir()):
        files = sorted(
            p for p in child.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if files:
            groups[child.name] = files
    return groups


def load_frames(files: list[Path]) -> tuple[list[Image.Image], list[tuple[int, int]]]:
    opened = []
    for path in files:
        with Image.open(path) as im:
            opened.append(ImageOps.exif_transpose(im).copy())
    has_alpha = any(im.mode in ("RGBA", "LA") or "transparency" in im.info for im in opened)
    sizes = [im.size for im in opened]
    varying_size = len(set(sizes)) != 1
    mode = "RGBA" if has_alpha or varying_size else "RGB"
    converted = [im.convert(mode) for im in opened]
    if not varying_size:
        return converted, sizes

    max_w = max(w for w, _ in sizes)
    max_h = max(h for _, h in sizes)
    background = (0, 0, 0, 0) if mode == "RGBA" else (0, 0, 0)
    frames: list[Image.Image] = []
    for im in converted:
        canvas = Image.new(mode, (max_w, max_h), background)
        canvas.paste(im, (0, 0))
        frames.append(canvas)
    return frames, sizes


def diff_stats(frames: list[Image.Image]) -> dict[str, object]:
    if len(frames) < 2:
        return {
            "canvas_pixels": frames[0].size[0] * frames[0].size[1],
            "exact_bbox_area_pct": [],
            "exact_changed_pixel_pct": [],
        }
    canvas = frames[0].size[0] * frames[0].size[1]
    bbox_pcts: list[float] = []
    changed_pcts: list[float] = []
    for prev, cur in zip(frames, frames[1:]):
        diff = ImageChops.difference(prev, cur)
        try:
            bbox = diff.getbbox(alpha_only=False)
        except TypeError:
            bbox = diff.getbbox()
        bbox_area = 0 if bbox is None else (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        bbox_pcts.append(bbox_area * 100.0 / canvas)
        mask = diff.convert("L").point(lambda value: 255 if value else 0)
        changed = mask.histogram()[255]
        changed_pcts.append(changed * 100.0 / canvas)
    return {
        "canvas_pixels": canvas,
        "exact_bbox_area_pct": bbox_pcts,
        "exact_changed_pixel_pct": changed_pcts,
    }


def save_apng(frames: list[Image.Image], path: Path, mode: str, duration_ms: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first, rest = frames[0], frames[1:]
    if mode == "full":
        disposal = [PngImagePlugin.Disposal.OP_BACKGROUND] * len(frames)
        blend = [PngImagePlugin.Blend.OP_SOURCE] * len(frames)
    elif mode == "delta":
        disposal = [PngImagePlugin.Disposal.OP_NONE] * len(frames)
        blend = [PngImagePlugin.Blend.OP_SOURCE] * len(frames)
    else:
        raise ValueError(mode)
    first.save(
        path,
        save_all=True,
        append_images=rest,
        default_image=False,
        duration=[duration_ms] * len(frames),
        loop=0,
        disposal=disposal,
        blend=blend,
        optimize=False,
        compress_level=0,
    )


def encode_one(
    cjxl: Path,
    src: Path,
    dst: Path,
    extra: list[str],
    force: bool,
    distance: str = "0",
) -> float:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return 0.0
    cmd = [str(cjxl), str(src), str(dst), "-d", distance, "-e", "10", "--quiet", *extra]
    return run(cmd)


def encode_individual(
    cjxl: Path,
    files: list[Path],
    out_dir: Path,
    method: str,
    pixel_jpeg: bool,
    force: bool,
    distance: str = "0",
) -> MethodResult:
    total = 0
    elapsed = 0.0
    for src in files:
        dst = out_dir / method / (src.stem + ".jxl")
        extra: list[str] = []
        if src.suffix.lower() in JPEG_SUFFIXES:
            extra.append("--lossless_jpeg=0" if pixel_jpeg else "--lossless_jpeg=1")
        elapsed += encode_one(cjxl, src, dst, extra, force, distance=distance)
        total += dst.stat().st_size
    return MethodResult(out_dir.name, method, total, elapsed, str(out_dir / method))


def encode_apng_method(
    cjxl: Path,
    frames: list[Image.Image],
    out_dir: Path,
    apng_mode: str,
    patches: int,
    duration_ms: int,
    force: bool,
    distance: str = "0",
    prefix: str = "",
) -> MethodResult:
    method = f"{prefix}apng_{apng_mode}_patches{patches}"
    apng = out_dir / "_apng" / f"{method}.png"
    if prefix:
        # Lossy/lossless JXL variants can share the same APNG source.
        apng = out_dir / "_apng" / f"apng_{apng_mode}_patches{patches}.png"
    jxl = out_dir / f"{method}.jxl"
    if force or not apng.exists():
        save_apng(frames, apng, apng_mode, duration_ms)
    elapsed = encode_one(cjxl, apng, jxl, [f"--patches={patches}"], force, distance=distance)
    return MethodResult(out_dir.name, method, jxl.stat().st_size, elapsed, str(jxl))


def write_reports(
    rows: list[dict[str, object]],
    stats: dict[str, object],
    out_root: Path,
) -> None:
    csv_path = out_root / "results.csv"
    md_path = out_root / "results.md"
    json_path = out_root / "diff_stats.json"
    out_root.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "method",
                "bytes",
                "size",
                "vs_original",
                "vs_single_source_jxl",
                "vs_single_pixel_jxl",
                "seconds",
                "output",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# JXL Layer Benchmark", ""]
    for group in sorted({str(row["group"]) for row in rows}):
        lines.append(f"## {group}")
        lines.append("")
        lines.append(
            "| method | size | bytes | vs original | vs single source JXL | vs single pixel JXL | seconds |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in [r for r in rows if r["group"] == group]:
            lines.append(
                f"| {row['method']} | {row['size']} | {row['bytes']} | "
                f"{row['vs_original']} | {row['vs_single_source_jxl']} | "
                f"{row['vs_single_pixel_jxl']} | {float(row['seconds']):.2f} |"
            )
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test_img/mt_lay")
    parser.add_argument("--output", default="out/mt_lay_benchmark")
    parser.add_argument(
        "--cjxl",
        default="out/build/windows-clang-cl-cli/libjxl/tools/cjxl.exe",
    )
    parser.add_argument("--duration-ms", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_root = Path(args.input)
    out_root = Path(args.output)
    cjxl = Path(args.cjxl)
    if not cjxl.exists():
        raise FileNotFoundError(cjxl)

    groups = find_groups(input_root)
    rows: list[dict[str, object]] = []
    all_stats: dict[str, object] = {}

    for group, files in groups.items():
        group_out = out_root / group
        frames, original_sizes = load_frames(files)
        stats = diff_stats(frames)
        all_stats[group] = {
            "files": [str(p) for p in files],
            "original_dimensions": original_sizes,
            "canvas_dimensions": frames[0].size,
            "original_bytes": sum(p.stat().st_size for p in files),
            **stats,
        }

        original = sum(p.stat().st_size for p in files)
        results: list[MethodResult] = [
            MethodResult(group, "original_sources", original, 0.0, str(input_root / group)),
            encode_individual(
                cjxl, files, group_out, "single_source_jxl_sum", pixel_jpeg=False, force=args.force
            ),
            encode_individual(
                cjxl, files, group_out, "single_pixel_jxl_sum", pixel_jpeg=True, force=args.force
            ),
            encode_individual(
                cjxl,
                files,
                group_out,
                "single_lossy_d1_jxl_sum",
                pixel_jpeg=True,
                force=args.force,
                distance="1",
            ),
            encode_apng_method(cjxl, frames, group_out, "full", 0, args.duration_ms, args.force),
            encode_apng_method(cjxl, frames, group_out, "full", 1, args.duration_ms, args.force),
            encode_apng_method(cjxl, frames, group_out, "delta", 0, args.duration_ms, args.force),
            encode_apng_method(cjxl, frames, group_out, "delta", 1, args.duration_ms, args.force),
            encode_apng_method(
                cjxl,
                frames,
                group_out,
                "full",
                0,
                args.duration_ms,
                args.force,
                distance="1",
                prefix="lossy_d1_",
            ),
            encode_apng_method(
                cjxl,
                frames,
                group_out,
                "full",
                1,
                args.duration_ms,
                args.force,
                distance="1",
                prefix="lossy_d1_",
            ),
            encode_apng_method(
                cjxl,
                frames,
                group_out,
                "delta",
                0,
                args.duration_ms,
                args.force,
                distance="1",
                prefix="lossy_d1_",
            ),
            encode_apng_method(
                cjxl,
                frames,
                group_out,
                "delta",
                1,
                args.duration_ms,
                args.force,
                distance="1",
                prefix="lossy_d1_",
            ),
        ]

        source_base = next(r.bytes for r in results if r.method == "single_source_jxl_sum")
        pixel_base = next(r.bytes for r in results if r.method == "single_pixel_jxl_sum")
        for result in results:
            rows.append(
                {
                    "group": group,
                    "method": result.method,
                    "bytes": result.bytes,
                    "size": fmt_bytes(result.bytes),
                    "vs_original": pct_delta(result.bytes, original),
                    "vs_single_source_jxl": pct_delta(result.bytes, source_base),
                    "vs_single_pixel_jxl": pct_delta(result.bytes, pixel_base),
                    "seconds": f"{result.seconds:.3f}",
                    "output": result.output,
                }
            )

    write_reports(rows, all_stats, out_root)
    print((out_root / "results.md").resolve())
    print((out_root / "results.csv").resolve())
    print((out_root / "diff_stats.json").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
