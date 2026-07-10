from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageMetrics:
    width: int
    height: int
    channels: int
    dtype: str
    sampled_pixels: int
    entropy_gray: float
    unique_colors: int
    unique_per_mpx: float
    flat4_pct: float
    near_white_pct: float
    near_black_pct: float
    edge_mean: float
    has_alpha: bool
    opaque_alpha: bool
    transparent_pct: float
    partial_alpha_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EncoderCandidate:
    name: str
    kwargs: dict[str, Any]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kwargs": dict(self.kwargs), "reason": self.reason}


@dataclass(frozen=True)
class CompressionRecommendation:
    profile: str
    candidates: tuple[EncoderCandidate, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class CompressionAnalysis:
    source_format: str
    metrics: ImageMetrics
    recommendation: CompressionRecommendation

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_format": self.source_format,
            "metrics": self.metrics.to_dict(),
            "recommendation": self.recommendation.to_dict(),
        }


def analyze_pixels(
    pixels: Any,
    *,
    layout: str = "auto",
    max_sample_pixels: int = 1_000_000,
) -> ImageMetrics:
    """Compute deterministic, image-only metrics for lossless JXL selection."""
    arr = _as_hwc(pixels, layout=layout)
    h, w, channels = arr.shape
    sample = _sample(arr, max_sample_pixels)
    sample8 = _to_uint8(sample)
    rgb = _rgb(sample8)
    gray = np.rint(
        rgb[..., 0].astype(np.float32) * 0.2126
        + rgb[..., 1].astype(np.float32) * 0.7152
        + rgb[..., 2].astype(np.float32) * 0.0722
    ).astype(np.uint8)

    sampled_pixels = int(sample8.shape[0] * sample8.shape[1])
    unique_colors = int(np.unique(sample8.reshape(-1, channels), axis=0).shape[0])
    unique_per_mpx = unique_colors / max(1, sampled_pixels) * 1_000_000.0

    hh = (sample8.shape[0] // 4) * 4
    ww = (sample8.shape[1] // 4) * 4
    if hh and ww:
        blocks = sample8[:hh, :ww].reshape(
            hh // 4, 4, ww // 4, 4, channels
        )
        flat = np.all(blocks == blocks[:, :1, :, :1, :], axis=(1, 3, 4))
        flat4_pct = float(np.mean(flat) * 100.0)
    else:
        flat4_pct = 0.0

    rgb_i = rgb.astype(np.int16)
    dx = np.abs(rgb_i[:, 1:] - rgb_i[:, :-1]).mean() if rgb.shape[1] > 1 else 0.0
    dy = np.abs(rgb_i[1:] - rgb_i[:-1]).mean() if rgb.shape[0] > 1 else 0.0

    has_alpha = channels in (2, 4)
    opaque_alpha = False
    transparent_pct = 0.0
    partial_alpha_pct = 0.0
    if has_alpha:
        alpha = sample8[..., -1]
        opaque_alpha = bool(np.all(alpha == 255))
        transparent_pct = float(np.mean(alpha == 0) * 100.0)
        partial_alpha_pct = float(np.mean((alpha > 0) & (alpha < 255)) * 100.0)

    return ImageMetrics(
        width=w,
        height=h,
        channels=channels,
        dtype=str(arr.dtype),
        sampled_pixels=sampled_pixels,
        entropy_gray=_entropy_u8(gray),
        unique_colors=unique_colors,
        unique_per_mpx=float(unique_per_mpx),
        flat4_pct=flat4_pct,
        near_white_pct=float(np.mean(np.all(rgb >= 245, axis=2)) * 100.0),
        near_black_pct=float(np.mean(np.all(rgb <= 16, axis=2)) * 100.0),
        edge_mean=float((dx + dy) / 2.0),
        has_alpha=has_alpha,
        opaque_alpha=opaque_alpha,
        transparent_pct=transparent_pct,
        partial_alpha_pct=partial_alpha_pct,
    )


def analyze_image(
    source: Any,
    *,
    layout: str = "auto",
    max_sample_pixels: int = 1_000_000,
) -> ImageMetrics:
    """Decode a path/bytes input with jxlpy and compute selection metrics."""
    if isinstance(source, np.ndarray):
        pixels = source
    else:
        from .api import decode

        pixels = decode(source, out="numpy")
    return analyze_pixels(
        pixels, layout=layout, max_sample_pixels=max_sample_pixels
    )


def analyze_lossless(
    source: Any,
    *,
    source_format: str | None = None,
    mode: str = "archive",
    exact_jpeg: bool = True,
    layout: str = "auto",
    max_sample_pixels: int = 1_000_000,
) -> CompressionAnalysis:
    """Analyze an input and return metrics plus a small lossless candidate plan."""
    if source_format is None:
        if isinstance(source, (str, Path)):
            source_format = Path(source).suffix
        else:
            source_format = ""
    metrics = analyze_image(
        source,
        layout=layout,
        max_sample_pixels=max_sample_pixels,
    )
    recommendation = recommend_lossless_candidates(
        metrics,
        source_format=source_format,
        mode=mode,
        exact_jpeg=exact_jpeg,
    )
    return CompressionAnalysis(
        source_format=source_format.lower().lstrip("."),
        metrics=metrics,
        recommendation=recommendation,
    )


def is_document_candidate(metrics: ImageMetrics) -> bool:
    return (
        metrics.near_white_pct > 50.0
        and metrics.entropy_gray < 4.5
        and (metrics.flat4_pct > 40.0 or metrics.unique_per_mpx < 10_000.0)
    )


def is_simple_screenshot_candidate(metrics: ImageMetrics) -> bool:
    """Detect simple UI, terminal, browser, and PDF/document screenshots."""
    return (
        is_document_candidate(metrics)
        or (
            metrics.flat4_pct > 70.0
            and metrics.entropy_gray < 4.5
        )
        or (
            metrics.flat4_pct > 35.0
            and metrics.entropy_gray < 2.5
            and metrics.unique_per_mpx < 10_000.0
        )
    )


def is_patch_candidate(metrics: ImageMetrics) -> bool:
    return (
        (
            metrics.entropy_gray < 2.0
            and metrics.unique_per_mpx < 5_000.0
        )
        or (
            metrics.flat4_pct > 70.0
            and metrics.unique_per_mpx < 2_000.0
        )
    )


def is_simple_jpeg_candidate(metrics: ImageMetrics) -> bool:
    return metrics.entropy_gray < 5.0 and (
        metrics.flat4_pct > 35.0
        or metrics.unique_per_mpx < 10_000.0
        or metrics.near_white_pct > 25.0
    )


def recommend_lossless_candidates(
    metrics: ImageMetrics,
    *,
    source_format: str = "",
    mode: str = "archive",
    exact_jpeg: bool = True,
) -> CompressionRecommendation:
    """Return a small candidate set; callers may encode all and keep the smallest."""
    if mode not in ("balanced", "archive"):
        raise ValueError("mode must be 'balanced' or 'archive'")
    effort = 9 if mode == "archive" else 8
    fmt = source_format.lower().lstrip(".")
    is_jpeg = fmt in ("jpg", "jpeg")
    is_screenshot = is_simple_screenshot_candidate(metrics)
    reasons: list[str] = []
    candidates: list[EncoderCandidate] = []

    if is_jpeg and exact_jpeg:
        reasons.append("JPEG byte-exact reconstruction requested")
        candidates.append(
            EncoderCandidate(
                name=f"jpeg_transcode_e{effort}",
                kwargs={"lossless_jpeg": True, "effort": effort},
                reason="preserve the original JPEG bitstream",
            )
        )
        return CompressionRecommendation("jpeg_transcode", tuple(candidates), tuple(reasons))

    default_candidate = EncoderCandidate(
        name=f"default_e{effort}",
        kwargs={
            "lossless": True,
            "distance": 0.0,
            "effort": effort,
            "patches": False,
        },
        reason="strong general lossless default",
    )
    candidates.append(default_candidate)

    if is_jpeg:
        candidates.insert(
            0,
            EncoderCandidate(
                name=f"jpeg_transcode_e{effort}",
                kwargs={"lossless_jpeg": True, "effort": effort},
                reason="cheap exact-JPEG candidate for size comparison",
            ),
        )
        if is_simple_jpeg_candidate(metrics):
            reasons.append("decoded JPEG pixels look simple enough to trial pixel-lossless")
        else:
            return CompressionRecommendation("jpeg_photo", tuple(candidates[:1]), tuple(reasons))

    if is_screenshot:
        if mode == "archive":
            reasons.append("simple screenshot statistics justify the archive modular preset")
            candidates.insert(
                0,
                EncoderCandidate(
                    name="screenshot_modular_e9",
                    kwargs={
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
                    },
                    reason="high-compression archive preset for simple screenshots",
                ),
            )
        else:
            reasons.append("simple screenshot detected; slow archive preset skipped")
    elif is_patch_candidate(metrics):
        reasons.append("flat/low-color statistics justify a patch trial")
        candidates.append(
            EncoderCandidate(
                name=f"patch_e{effort}",
                kwargs={
                    "lossless": True,
                    "distance": 0.0,
                    "effort": effort,
                    "patches": True,
                },
                reason="trial candidate for repeated text or flat structures",
            )
        )

    profile = "simple_screenshot" if is_screenshot else (
        "flat_or_palette" if is_patch_candidate(metrics) else "general"
    )
    return CompressionRecommendation(profile, tuple(candidates), tuple(reasons))


def _as_hwc(value: Any, *, layout: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim != 3:
        raise ValueError("expected a 2D image or 3D HWC/CHW array")
    if layout == "auto":
        if 1 <= arr.shape[-1] <= 4:
            pass
        elif 1 <= arr.shape[0] <= 4:
            arr = np.moveaxis(arr, 0, -1)
        else:
            raise ValueError("cannot infer channel axis")
    elif layout == "chw":
        arr = np.moveaxis(arr, 0, -1)
    elif layout != "hwc":
        raise ValueError("layout must be 'auto', 'hwc' or 'chw'")
    if not 1 <= arr.shape[2] <= 4:
        raise ValueError("image channels must be in 1..4")
    return np.ascontiguousarray(arr)


def _sample(arr: np.ndarray, max_pixels: int) -> np.ndarray:
    pixels = int(arr.shape[0] * arr.shape[1])
    if max_pixels <= 0 or pixels <= max_pixels:
        return arr
    step = int(np.ceil(np.sqrt(pixels / max_pixels)))
    return arr[::step, ::step]


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    if arr.dtype == np.uint16:
        return np.ascontiguousarray((arr >> 8).astype(np.uint8))
    values = arr.astype(np.float32)
    if values.max(initial=0.0) <= 1.0:
        values *= 255.0
    return np.clip(values, 0.0, 255.0).astype(np.uint8)


def _rgb(arr: np.ndarray) -> np.ndarray:
    if arr.shape[2] >= 3:
        return arr[:, :, :3]
    return np.repeat(arr[:, :, :1], 3, axis=2)


def _entropy_u8(values: np.ndarray) -> float:
    hist = np.bincount(values.reshape(-1), minlength=256).astype(np.float64)
    prob = hist[hist > 0]
    prob /= prob.sum()
    return float(-(prob * np.log2(prob)).sum())


__all__ = [
    "CompressionAnalysis",
    "CompressionRecommendation",
    "EncoderCandidate",
    "ImageMetrics",
    "analyze_image",
    "analyze_lossless",
    "analyze_pixels",
    "is_document_candidate",
    "is_patch_candidate",
    "is_simple_screenshot_candidate",
    "is_simple_jpeg_candidate",
    "recommend_lossless_candidates",
]
