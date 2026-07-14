from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


EXTRA_TYPE_TO_NATIVE = {
    "alpha": 0,
    "depth": 1,
    "spot_color": 2,
    "selection_mask": 3,
    "black": 4,
    "cfa": 5,
    "thermal": 6,
    "unknown": 15,
    "optional": 16,
}
NATIVE_EXTRA_TYPE = {value: key for key, value in EXTRA_TYPE_TO_NATIVE.items()}

COLOR_SPACE = {"rgb": 0, "gray": 1, "grey": 1, "xyb": 2, "unknown": 3}
WHITE_POINT = {"d65": 1, "custom": 2, "e": 10, "dci": 11}
PRIMARIES = {"srgb": 1, "custom": 2, "bt2100": 9, "2100": 9, "p3": 11}
TRANSFER_FUNCTION = {
    "bt709": 1,
    "709": 1,
    "unknown": 2,
    "linear": 8,
    "srgb": 13,
    "pq": 16,
    "dci": 17,
    "hlg": 18,
    "gamma": 65535,
}
RENDERING_INTENT = {
    "perceptual": 0,
    "relative": 1,
    "relative_colorimetric": 1,
    "saturation": 2,
    "absolute": 3,
    "absolute_colorimetric": 3,
}

_COLOR_SPACE_NAME = {0: "rgb", 1: "gray", 2: "xyb", 3: "unknown"}
_WHITE_POINT_NAME = {1: "d65", 2: "custom", 10: "e", 11: "dci"}
_PRIMARIES_NAME = {1: "srgb", 2: "custom", 9: "bt2100", 11: "p3"}
_TRANSFER_NAME = {
    1: "bt709",
    2: "unknown",
    8: "linear",
    13: "srgb",
    16: "pq",
    17: "dci",
    18: "hlg",
    65535: "gamma",
}
_INTENT_NAME = {0: "perceptual", 1: "relative", 2: "saturation", 3: "absolute"}

_SRGB_BASE = {
    "color_space": "rgb",
    "white_point": "d65",
    "white_point_xy": (0.3127, 0.3290),
    "primaries": "srgb",
    "primaries_red_xy": (0.639998686, 0.330010138),
    "primaries_green_xy": (0.300003784, 0.600003357),
    "primaries_blue_xy": (0.150002046, 0.059997204),
    "transfer_function": "srgb",
    "gamma": 0.0,
    "rendering_intent": "relative",
}

COLOR_PRESETS = {
    "srgb": {},
    "linear_srgb": {"transfer_function": "linear"},
    "gray_srgb": {"color_space": "gray"},
    "linear_gray": {"color_space": "gray", "transfer_function": "linear"},
    "display_p3": {"primaries": "p3"},
    "rec2100_pq": {"primaries": "bt2100", "transfer_function": "pq"},
    "rec2100_hlg": {"primaries": "bt2100", "transfer_function": "hlg"},
}


@dataclass(frozen=True)
class ParsedExtraChannel:
    name: str
    type_id: int
    bits_per_sample: int
    exponent_bits_per_sample: int
    data: Any
    dim_shift: int = 0
    alpha_premultiplied: bool = False
    spot_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    cfa_channel: int = 0


def extra_type_id(value: Any) -> int:
    if value is None:
        return EXTRA_TYPE_TO_NATIVE["optional"]
    if isinstance(value, str):
        key = value.lower().replace("-", "_")
        if key not in EXTRA_TYPE_TO_NATIVE:
            raise ValueError(f"unknown extra channel type: {value!r}")
        result = EXTRA_TYPE_TO_NATIVE[key]
    else:
        result = int(value)
    if result == EXTRA_TYPE_TO_NATIVE["unknown"]:
        raise ValueError(
            "extra channel type 'unknown' is decoder-only; use 'optional' "
            "for application-defined data"
        )
    if result not in {*range(7), EXTRA_TYPE_TO_NATIVE["optional"]}:
        raise ValueError(f"extra channel type is not encodable: {value!r}")
    return result


def parse_extra_channel(spec: Any) -> ParsedExtraChannel:
    if isinstance(spec, ParsedExtraChannel):
        return spec
    if isinstance(spec, Mapping):
        if "data" not in spec:
            raise ValueError("extra channel mappings require a 'data' value")
        spot_color = tuple(
            float(value) for value in spec.get("spot_color", (0, 0, 0, 0))
        )
        if len(spot_color) != 4:
            raise ValueError("spot_color must contain four linear RGBA values")
        dim_shift = int(spec.get("dim_shift", 0))
        if not 0 <= dim_shift <= 3:
            raise ValueError("dim_shift must be in 0..3")
        bits_per_sample = int(spec.get("bits_per_sample", 0))
        exponent_bits_per_sample = int(spec.get("exponent_bits_per_sample", 0))
        cfa_channel = int(spec.get("cfa_channel", 0))
        if bits_per_sample < 0 or exponent_bits_per_sample < 0:
            raise ValueError("extra channel bit depths must be non-negative")
        if not 0 <= cfa_channel <= 0xFFFFFFFF:
            raise ValueError("cfa_channel must be an unsigned 32-bit integer")
        return ParsedExtraChannel(
            name=str(spec.get("name", "")),
            type_id=extra_type_id(spec.get("type", "optional")),
            bits_per_sample=bits_per_sample,
            exponent_bits_per_sample=exponent_bits_per_sample,
            data=spec["data"],
            dim_shift=dim_shift,
            alpha_premultiplied=bool(
                spec.get("alpha_premultiplied", spec.get("alpha_associated", False))
            ),
            spot_color=spot_color,
            cfa_channel=cfa_channel,
        )
    if isinstance(spec, tuple):
        if len(spec) == 2:
            return ParsedExtraChannel(
                name=str(spec[0]),
                type_id=EXTRA_TYPE_TO_NATIVE["optional"],
                bits_per_sample=0,
                exponent_bits_per_sample=0,
                data=spec[1],
            )
        if len(spec) == 3:
            return ParsedExtraChannel(
                name=str(spec[0]),
                type_id=extra_type_id(spec[1]),
                bits_per_sample=0,
                exponent_bits_per_sample=0,
                data=spec[2],
            )
        raise ValueError("extra channel tuple must be (name, data) or (name, type, data)")
    return ParsedExtraChannel(
        name="",
        type_id=EXTRA_TYPE_TO_NATIVE["optional"],
        bits_per_sample=0,
        exponent_bits_per_sample=0,
        data=spec,
    )


def _enum_value(value: Any, choices: Mapping[str, int], field: str) -> int:
    if isinstance(value, str):
        key = value.lower().replace("-", "_")
        if key not in choices:
            raise ValueError(f"unknown {field}: {value!r}")
        return choices[key]
    return int(value)


def normalize_color_encoding(value: str | Mapping[str, Any]) -> dict[str, Any]:
    result = dict(_SRGB_BASE)
    if isinstance(value, str):
        key = value.lower().replace("-", "_")
        if key not in COLOR_PRESETS:
            raise ValueError(f"unknown color encoding preset: {value!r}")
        result.update(COLOR_PRESETS[key])
    elif isinstance(value, Mapping):
        preset = value.get("preset")
        if preset is not None:
            key = str(preset).lower().replace("-", "_")
            if key not in COLOR_PRESETS:
                raise ValueError(f"unknown color encoding preset: {preset!r}")
            result.update(COLOR_PRESETS[key])
        result.update({key: item for key, item in value.items() if key != "preset"})
    else:
        raise TypeError("color_encoding must be a preset name or mapping")
    return result


def fill_color_encoding(target: Any, value: str | Mapping[str, Any]) -> None:
    color = normalize_color_encoding(value)
    target.available = 1
    target.color_space = _enum_value(color["color_space"], COLOR_SPACE, "color space")
    target.white_point = _enum_value(color["white_point"], WHITE_POINT, "white point")
    target.primaries = _enum_value(color["primaries"], PRIMARIES, "primaries")
    target.transfer_function = _enum_value(
        color["transfer_function"], TRANSFER_FUNCTION, "transfer function"
    )
    target.rendering_intent = _enum_value(
        color["rendering_intent"], RENDERING_INTENT, "rendering intent"
    )
    if len(color["white_point_xy"]) != 2:
        raise ValueError("white_point_xy must contain two values")
    for index, item in enumerate(color["white_point_xy"]):
        target.white_point_xy[index] = float(item)
    for field in ("primaries_red_xy", "primaries_green_xy", "primaries_blue_xy"):
        if len(color[field]) != 2:
            raise ValueError(f"{field} must contain two values")
        for index, item in enumerate(color[field]):
            getattr(target, field)[index] = float(item)
    target.gamma = float(color.get("gamma", 0.0))


def color_encoding_to_dict(value: Any) -> dict[str, Any] | None:
    if not bool(value.available):
        return None
    color_space = int(value.color_space)
    white_point = int(value.white_point)
    primaries = int(value.primaries)
    transfer = int(value.transfer_function)
    intent = int(value.rendering_intent)
    return {
        "color_space": _COLOR_SPACE_NAME.get(color_space, color_space),
        "color_space_id": color_space,
        "white_point": _WHITE_POINT_NAME.get(white_point, white_point),
        "white_point_id": white_point,
        "white_point_xy": tuple(float(item) for item in value.white_point_xy),
        "primaries": _PRIMARIES_NAME.get(primaries, primaries),
        "primaries_id": primaries,
        "primaries_red_xy": tuple(float(item) for item in value.primaries_red_xy),
        "primaries_green_xy": tuple(float(item) for item in value.primaries_green_xy),
        "primaries_blue_xy": tuple(float(item) for item in value.primaries_blue_xy),
        "transfer_function": _TRANSFER_NAME.get(transfer, transfer),
        "transfer_function_id": transfer,
        "gamma": float(value.gamma),
        "rendering_intent": _INTENT_NAME.get(intent, intent),
        "rendering_intent_id": intent,
    }


def read_icc_profile(value: Any) -> bytes:
    if isinstance(value, (str, Path)):
        return Path(value).read_bytes()
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    raise TypeError("icc_profile must be a path or bytes-like object")


__all__ = [
    "COLOR_PRESETS",
    "EXTRA_TYPE_TO_NATIVE",
    "NATIVE_EXTRA_TYPE",
    "ParsedExtraChannel",
    "color_encoding_to_dict",
    "fill_color_encoding",
    "parse_extra_channel",
    "read_icc_profile",
]
