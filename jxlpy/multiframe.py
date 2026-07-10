from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class DiffStats:
    bbox: tuple[int, int, int, int]
    changed_pixels: int
    changed_pct: float
    bbox_area: int
    bbox_pct: float


def changed_mask(
    current: np.ndarray,
    reference: np.ndarray,
    current_extras: Iterable[np.ndarray] = (),
    reference_extras: Iterable[np.ndarray] = (),
) -> np.ndarray:
    """Return an exact per-pixel change mask across main and extra channels."""
    if current.shape != reference.shape:
        raise ValueError("current and reference frames must have the same shape")
    if current.ndim != 3:
        raise ValueError("frames must be HWC arrays")
    cur_extras = list(current_extras)
    ref_extras = list(reference_extras)
    if len(cur_extras) != len(ref_extras):
        raise ValueError("current and reference extra-channel counts must match")

    changed = np.any(current != reference, axis=2)
    for cur_extra, ref_extra in zip(cur_extras, ref_extras):
        if cur_extra.shape != ref_extra.shape or cur_extra.shape != changed.shape:
            raise ValueError("extra-channel dimensions must match the frame canvas")
        changed |= cur_extra != ref_extra
    return changed


def diff_stats_from_mask(changed: np.ndarray) -> DiffStats:
    """Measure changed coverage and its enclosing bounding box."""
    changed = np.asarray(changed, dtype=bool)
    if changed.ndim != 2:
        raise ValueError("changed mask must be a 2D array")
    changed_count = int(np.count_nonzero(changed))
    full_area = int(changed.shape[0] * changed.shape[1])
    if changed_count == 0:
        bbox = (0, 0, 1, 1)
    else:
        rows = np.flatnonzero(np.any(changed, axis=1))
        cols = np.flatnonzero(np.any(changed, axis=0))
        bbox = (
            int(cols[0]),
            int(rows[0]),
            int(cols[-1]) + 1,
            int(rows[-1]) + 1,
        )
    x0, y0, x1, y1 = bbox
    bbox_area = int((x1 - x0) * (y1 - y0))
    return DiffStats(
        bbox=bbox,
        changed_pixels=changed_count,
        changed_pct=changed_count / full_area * 100.0 if full_area else 0.0,
        bbox_area=bbox_area,
        bbox_pct=bbox_area / full_area * 100.0 if full_area else 0.0,
    )


def diff_stats(
    current: np.ndarray,
    reference: np.ndarray,
    current_extras: Iterable[np.ndarray] = (),
    reference_extras: Iterable[np.ndarray] = (),
) -> DiffStats:
    return diff_stats_from_mask(
        changed_mask(current, reference, current_extras, reference_extras)
    )


def select_reference_bbox(
    *,
    index: int,
    current: np.ndarray,
    arrays: list[np.ndarray],
    current_extras: list[np.ndarray],
    extra_specs: list[dict[str, Any]],
    refs: dict[int, tuple[np.ndarray, list[np.ndarray]]],
    reference: str,
) -> tuple[int, DiffStats, str]:
    """Choose previous or first reference by encoded bbox area."""
    candidates: list[tuple[int, DiffStats, str]] = []
    if reference in ("previous", "auto") and 1 in refs:
        ref_main, ref_extras = refs[1]
        candidates.append(
            (1, diff_stats(current, ref_main, current_extras, ref_extras), "previous")
        )
    if reference in ("first", "auto") and 2 in refs:
        ref_main, ref_extras = refs[2]
        candidates.append(
            (2, diff_stats(current, ref_main, current_extras, ref_extras), "first")
        )
    if not candidates:
        previous_extras = [spec["arrays"][index - 1] for spec in extra_specs]
        candidates.append(
            (
                1,
                diff_stats(current, arrays[index - 1], current_extras, previous_extras),
                "previous",
            )
        )
    return min(candidates, key=lambda item: (item[1].bbox_area, item[1].changed_pixels))


def analyze_frame_arrays(
    arrays: Iterable[np.ndarray],
    *,
    extra_specs: list[dict[str, Any]] | None = None,
    reference: str = "auto",
    min_crop_ratio: float = 0.98,
) -> dict[str, Any]:
    """Analyze exact frame differences using the same crop policy as the encoder."""
    if reference not in ("auto", "first", "previous", "none", "full"):
        raise ValueError("reference must be 'auto', 'first', 'previous', 'none' or 'full'")
    frames = [np.asarray(frame) for frame in arrays]
    if not frames:
        raise ValueError("frames must not be empty")
    first_shape = frames[0].shape
    first_dtype = frames[0].dtype
    if len(first_shape) != 3:
        raise ValueError("frames must be HWC arrays")
    for frame in frames:
        if frame.shape != first_shape or frame.dtype != first_dtype:
            raise ValueError("all frames must have the same shape and dtype")

    specs = list(extra_specs or [])
    h, w, channels = first_shape
    full_area = w * h
    frame_stats: list[dict[str, Any]] = []
    refs: dict[int, tuple[np.ndarray, list[np.ndarray]]] = {}

    for i, frame in enumerate(frames):
        full_extras = [spec["arrays"][i] for spec in specs]
        source_ref = 0
        source_name = "none"
        save_ref = 0
        use_crop = False
        x0 = y0 = 0
        x1 = w
        y1 = h

        if i == 0:
            if reference in ("auto", "first") and len(frames) > 1:
                save_ref = 2
            elif reference == "previous" and len(frames) > 1:
                save_ref = 1
            frame_stats.append(
                _frame_record(
                    index=0,
                    source_ref=0,
                    source="none",
                    save_ref=save_ref,
                    use_crop=False,
                    bbox=(0, 0, w, h),
                    crop=(0, 0, w, h),
                    changed_pixels=full_area,
                    changed_pct=100.0,
                    bbox_area=full_area,
                    bbox_pct=100.0,
                    encoded_area=full_area,
                    full_area=full_area,
                )
            )
            if save_ref:
                refs[save_ref] = (frame, full_extras)
            continue

        if reference in ("none", "full"):
            diff = diff_stats(
                frame,
                frames[i - 1],
                full_extras,
                [spec["arrays"][i - 1] for spec in specs],
            )
            frame_stats.append(
                _frame_record(
                    index=i,
                    source_ref=0,
                    source="none",
                    save_ref=0,
                    use_crop=False,
                    bbox=(0, 0, w, h),
                    crop=(0, 0, w, h),
                    changed_pixels=diff.changed_pixels,
                    changed_pct=diff.changed_pct,
                    bbox_area=full_area,
                    bbox_pct=100.0,
                    encoded_area=full_area,
                    full_area=full_area,
                )
            )
            continue

        source_ref, diff, source_name = select_reference_bbox(
            index=i,
            current=frame,
            arrays=frames,
            current_extras=full_extras,
            extra_specs=specs,
            refs=refs,
            reference=reference,
        )
        bbox = diff.bbox
        x0, y0, x1, y1 = bbox
        if diff.bbox_area < full_area * float(min_crop_ratio):
            use_crop = True
        else:
            source_ref = 0
            source_name = "none"
            x0 = y0 = 0
            x1 = w
            y1 = h
        if reference in ("previous", "auto"):
            save_ref = 1

        frame_stats.append(
            _frame_record(
                index=i,
                source_ref=source_ref,
                source=source_name,
                save_ref=save_ref,
                use_crop=use_crop,
                bbox=bbox,
                crop=(x0, y0, x1, y1),
                changed_pixels=diff.changed_pixels,
                changed_pct=diff.changed_pct,
                bbox_area=diff.bbox_area,
                bbox_pct=diff.bbox_pct,
                encoded_area=diff.bbox_area if use_crop else full_area,
                full_area=full_area,
            )
        )
        if save_ref:
            refs[save_ref] = (frame, full_extras)

    tail = frame_stats[1:]
    avg_bbox_pct = float(np.mean([item["bbox_pct"] for item in tail])) if tail else 100.0
    avg_encoded_pct = (
        float(np.mean([item["encoded_pct"] for item in tail])) if tail else 100.0
    )
    avg_changed_pct = (
        float(np.mean([item["changed_pct"] for item in tail])) if tail else 100.0
    )
    if avg_encoded_pct < 30.0:
        recommendation = "highly_beneficial"
    elif avg_encoded_pct < 70.0:
        recommendation = "moderately_beneficial"
    else:
        recommendation = "minimal_benefit"

    return {
        "num_frames": len(frames),
        "canvas_size": (w, h),
        "channels": channels,
        "dtype": first_dtype,
        "reference": reference,
        "min_crop_ratio": float(min_crop_ratio),
        "avg_bbox_pct": avg_bbox_pct,
        "avg_encoded_pct": avg_encoded_pct,
        "avg_changed_pct": avg_changed_pct,
        "recommendation": recommendation,
        "frames": frame_stats,
    }


def compare_reference_modes(
    arrays: Iterable[np.ndarray],
    *,
    extra_specs: list[dict[str, Any]] | None = None,
    min_crop_ratio: float = 0.98,
    modes: Iterable[str] = ("auto", "previous", "first", "full"),
) -> dict[str, Any]:
    """Compare reference policies by estimated encoded pixel area."""
    frames = [np.asarray(frame) for frame in arrays]
    reports: dict[str, dict[str, Any]] = {}
    for mode in modes:
        report = analyze_frame_arrays(
            frames,
            extra_specs=extra_specs,
            reference=mode,
            min_crop_ratio=min_crop_ratio,
        )
        total_encoded_area = int(
            sum(frame["encoded_area"] for frame in report["frames"])
        )
        reports[mode] = {
            "total_encoded_area": total_encoded_area,
            "avg_encoded_pct": report["avg_encoded_pct"],
            "avg_bbox_pct": report["avg_bbox_pct"],
            "avg_changed_pct": report["avg_changed_pct"],
            "recommendation": report["recommendation"],
        }
    best_reference = min(
        reports,
        key=lambda mode: (
            reports[mode]["total_encoded_area"],
            reports[mode]["avg_changed_pct"],
        ),
    )
    return {"best_reference": best_reference, "modes": reports}


def _frame_record(
    *,
    index: int,
    source_ref: int,
    source: str,
    save_ref: int,
    use_crop: bool,
    bbox: tuple[int, int, int, int],
    crop: tuple[int, int, int, int],
    changed_pixels: int,
    changed_pct: float,
    bbox_area: int,
    bbox_pct: float,
    encoded_area: int,
    full_area: int,
) -> dict[str, Any]:
    bx0, by0, bx1, by1 = bbox
    cx0, cy0, cx1, cy1 = crop
    return {
        "index": index,
        "source_ref": source_ref,
        "source": source,
        "save_as_ref": save_ref,
        "use_crop": use_crop,
        "bbox_x0": bx0,
        "bbox_y0": by0,
        "bbox_x1": bx1,
        "bbox_y1": by1,
        "bbox_xsize": bx1 - bx0,
        "bbox_ysize": by1 - by0,
        "crop_x0": cx0,
        "crop_y0": cy0,
        "crop_x1": cx1,
        "crop_y1": cy1,
        "crop_xsize": cx1 - cx0,
        "crop_ysize": cy1 - cy0,
        "changed_pixels": changed_pixels,
        "changed_pct": changed_pct,
        "bbox_area": bbox_area,
        "bbox_pct": bbox_pct,
        "encoded_area": encoded_area,
        "encoded_pct": encoded_area / full_area * 100.0 if full_area else 0.0,
    }


__all__ = [
    "DiffStats",
    "analyze_frame_arrays",
    "changed_mask",
    "compare_reference_modes",
    "diff_stats",
    "diff_stats_from_mask",
    "select_reference_bbox",
]
