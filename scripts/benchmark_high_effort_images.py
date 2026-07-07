from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path


PNG_SUFFIXES = {".png"}
JPEG_SUFFIXES = {".jpg", ".jpeg"}
IMAGE_SUFFIXES = PNG_SUFFIXES | JPEG_SUFFIXES


@dataclass(frozen=True)
class Method:
    name: str
    args: tuple[str, ...]
    applies_to: tuple[str, ...]


@dataclass
class Result:
    suite: str
    group: str
    source: str
    method: str
    original_bytes: int
    encoded_bytes: int
    ratio: float
    saved_pct: float
    seconds: float
    output: str


@dataclass
class Summary:
    suite: str
    group: str
    method: str
    count: int
    original_bytes: int
    encoded_bytes: int
    ratio: float
    saved_pct: float
    seconds: float


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_cjxl() -> Path:
    root = repo_root()
    exe = "cjxl.exe" if is_windows() else "cjxl"
    candidates = [
        root / "out" / "build" / "windows-clang-cl-cli" / "libjxl" / "tools" / exe,
        root / "out" / "build" / "linux-clang-python" / "libjxl" / "tools" / exe,
        root / "out" / "build" / "macos-clang-python" / "libjxl" / "tools" / exe,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(exe)


def is_windows() -> bool:
    import os

    return os.name == "nt"


def parse_patches(values: list[str]) -> list[str]:
    allowed = {"auto", "off", "on"}
    out = []
    for value in values:
        value = value.lower()
        if value not in allowed:
            raise ValueError(f"patch mode must be one of {sorted(allowed)}: {value}")
        out.append(value)
    return out


def parse_jpeg_modes(value: str) -> set[str]:
    if value == "none":
        return set()
    modes = {item.strip().lower() for item in value.split(",") if item.strip()}
    if "both" in modes:
        modes.remove("both")
        modes.update({"transcode", "pixel"})
    allowed = {"transcode", "pixel"}
    unknown = modes - allowed
    if unknown:
        raise ValueError(f"unknown jpeg mode(s): {', '.join(sorted(unknown))}")
    return modes


def patch_args(mode: str) -> tuple[str, ...]:
    if mode == "auto":
        return ()
    if mode == "off":
        return ("--patches=0",)
    if mode == "on":
        return ("--patches=1",)
    raise ValueError(mode)


def build_methods(
    efforts: list[int],
    patches: list[str],
    jpeg_modes: set[str],
) -> list[Method]:
    methods: list[Method] = []
    for effort in efforts:
        for patch_mode in patches:
            methods.append(
                Method(
                    name=f"png_lossless_e{effort}_patches_{patch_mode}",
                    args=("-d", "0", "-e", str(effort), *patch_args(patch_mode)),
                    applies_to=(".png",),
                )
            )
        if "transcode" in jpeg_modes:
            methods.append(
                Method(
                    name=f"jpeg_transcode_e{effort}",
                    args=("--lossless_jpeg=1", "-e", str(effort)),
                    applies_to=(".jpg", ".jpeg"),
                )
            )
        if "pixel" in jpeg_modes:
            for patch_mode in patches:
                methods.append(
                    Method(
                        name=f"jpeg_pixel_lossless_e{effort}_patches_{patch_mode}",
                        args=(
                            "--lossless_jpeg=0",
                            "-d",
                            "0",
                            "-e",
                            str(effort),
                            *patch_args(patch_mode),
                        ),
                        applies_to=(".jpg", ".jpeg"),
                    )
                )
    return methods


def rel_group(path: Path, root: Path) -> str:
    rel = path.parent.relative_to(root)
    return "." if str(rel) == "." else rel.as_posix()


def find_pngs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in PNG_SUFFIXES)


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def output_path(out_dir: Path, suite: str, root: Path, source: Path, method: str) -> Path:
    rel = source.relative_to(root)
    dst = out_dir / "encoded" / suite / method / rel
    return dst.with_suffix(".jxl")


def run_cjxl(cjxl: Path, source: Path, output: Path, args: tuple[str, ...], timeout: int) -> float:
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


def encode_one(
    *,
    cjxl: Path,
    suite: str,
    root: Path,
    source: Path,
    method: Method,
    out_dir: Path,
    timeout: int,
    reuse: bool,
) -> Result:
    output = output_path(out_dir, suite, root, source, method.name)
    if output.exists() and reuse:
        seconds = 0.0
    else:
        seconds = run_cjxl(cjxl, source, output, method.args, timeout)
    original = source.stat().st_size
    encoded = output.stat().st_size
    ratio = encoded / original if original else 0.0
    return Result(
        suite=suite,
        group=rel_group(source, root),
        source=source.relative_to(root).as_posix(),
        method=method.name,
        original_bytes=original,
        encoded_bytes=encoded,
        ratio=ratio,
        saved_pct=(1.0 - ratio) * 100.0 if original else 0.0,
        seconds=seconds,
        output=output.as_posix(),
    )


def summarize(results: list[Result]) -> list[Summary]:
    buckets: dict[tuple[str, str, str], list[Result]] = {}
    for result in results:
        buckets.setdefault((result.suite, result.group, result.method), []).append(result)
    rows: list[Summary] = []
    for (suite, group, method), items in sorted(buckets.items()):
        original = sum(item.original_bytes for item in items)
        encoded = sum(item.encoded_bytes for item in items)
        ratio = encoded / original if original else 0.0
        rows.append(
            Summary(
                suite=suite,
                group=group,
                method=method,
                count=len(items),
                original_bytes=original,
                encoded_bytes=encoded,
                ratio=ratio,
                saved_pct=(1.0 - ratio) * 100.0 if original else 0.0,
                seconds=sum(item.seconds for item in items),
            )
        )
    return rows


def write_csv(path: Path, rows: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(path: Path, results: list[Result], summaries: list[Summary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": [asdict(row) for row in results],
        "summaries": [asdict(row) for row in summaries],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_summary(rows: list[Summary]) -> None:
    if not rows:
        print("No results.")
        return
    headers = ["suite", "group", "method", "count", "encoded", "saved%", "seconds"]
    print("\t".join(headers))
    for row in rows:
        print(
            "\t".join(
                [
                    row.suite,
                    row.group,
                    row.method,
                    str(row.count),
                    str(row.encoded_bytes),
                    f"{row.saved_pct:.2f}",
                    f"{row.seconds:.3f}",
                ]
            )
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark high-effort lossless JXL encodes for test PNGs and mt_lay single images."
    )
    parser.add_argument("--root", type=Path, default=repo_root() / "test_img")
    parser.add_argument("--diff-root", type=Path, default=repo_root() / "test_img" / "mt_lay")
    parser.add_argument("--out", type=Path, default=repo_root() / "out" / "test-run" / "high_effort_images")
    parser.add_argument("--cjxl", type=Path, default=default_cjxl())
    parser.add_argument("--efforts", type=int, nargs="+", default=[8])
    parser.add_argument("--patches", nargs="+", default=["auto", "on"])
    parser.add_argument(
        "--jpeg-modes",
        default="transcode,pixel",
        help="For diff-root JPEGs: comma list of transcode,pixel,both, or none.",
    )
    parser.add_argument("--no-test-png", action="store_true")
    parser.add_argument("--no-diff-single", action="store_true")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing .jxl outputs and report seconds=0.")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    root = args.root.resolve()
    diff_root = args.diff_root.resolve()
    out_dir = args.out.resolve()
    cjxl = args.cjxl.resolve()
    patches = parse_patches(args.patches)
    jpeg_modes = parse_jpeg_modes(args.jpeg_modes)
    methods = build_methods(args.efforts, patches, jpeg_modes)

    if not cjxl.exists():
        raise FileNotFoundError(f"cjxl not found: {cjxl}")

    jobs: list[tuple[str, Path, Path, Method]] = []
    if not args.no_test_png:
        for source in find_pngs(root):
            for method in methods:
                if source.suffix.lower() in method.applies_to and method.name.startswith("png_"):
                    jobs.append(("test_png", root, source, method))

    if not args.no_diff_single and diff_root.exists():
        for source in find_images(diff_root):
            for method in methods:
                if source.suffix.lower() in method.applies_to:
                    jobs.append(("diff_single", diff_root, source, method))

    print(f"cjxl={cjxl}")
    print(f"out={out_dir}")
    print(f"jobs={len(jobs)}")

    results: list[Result] = []
    for index, (suite, suite_root, source, method) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {suite} {source.name} {method.name}", flush=True)
        results.append(
            encode_one(
                cjxl=cjxl,
                suite=suite,
                root=suite_root,
                source=source,
                method=method,
                out_dir=out_dir,
                timeout=args.timeout,
                reuse=args.reuse,
            )
        )

    summaries = summarize(results)
    write_csv(out_dir / "results.csv", results)
    write_csv(out_dir / "summary.csv", summaries)
    write_json(out_dir / "results.json", results, summaries)

    print()
    print_summary(summaries)
    print()
    print(f"wrote {out_dir / 'results.csv'}")
    print(f"wrote {out_dir / 'summary.csv'}")
    print(f"wrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
