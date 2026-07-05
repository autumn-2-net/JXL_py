"""Focused test for extra channels: single/multiple, dtypes, types, single-pass decode."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jxlpy


def test_single_extra_channel():
    """One extra channel on a single-frame RGB image."""
    print("=" * 60)
    print("TEST: Single extra channel (RGB + selection_mask)")
    print("=" * 60)

    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[..., 0] = 100
    rgb[..., 1] = 50
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:16, 8:16] = 255
    mask[20:28, 20:28] = 128

    jxl = jxlpy.encode(rgb, extra_channels=[("mask", "selection_mask", mask)])
    print(f"  Encoded: {len(jxl)} bytes")

    # Method 1: decode_extra_channel
    plane, meta = jxlpy.decode_extra_channel(jxl, 0)
    assert plane.shape == (32, 32), f"shape mismatch: {plane.shape}"
    assert np.array_equal(plane, mask), "mask mismatch via decode_extra_channel"
    assert meta["extra_channel_type"] == "selection_mask", f"type: {meta['extra_channel_type']}"
    print(f"  decode_extra_channel: OK (shape={plane.shape}, type={meta['extra_channel_type']})")

    # Method 2: decode(return_extra_channels=True) — single-pass path
    img, meta2 = jxlpy.decode(jxl, return_info=True, return_extra_channels=True)
    assert np.array_equal(img, rgb), "color mismatch via return_extra_channels"
    assert len(meta2["extra_channels"]) == 1, f"expected 1 extra, got {len(meta2['extra_channels'])}"
    ec = meta2["extra_channels"][0]
    assert ec["name"] == "mask", f"name: {ec['name']}"
    assert ec["type"] == "selection_mask", f"type: {ec['type']}"
    assert np.array_equal(ec["data"], mask), "mask mismatch via return_extra_channels"
    print(f"  return_extra_channels: OK (name={ec['name']}, type={ec['type']})")

    print("  RESULT: PASS")
    print()
    return True


def test_rgba_alpha_extra():
    """RGBA image: alpha is extra channel 0, user extra starts at 1."""
    print("=" * 60)
    print("TEST: RGBA + user extra channel (alpha handling)")
    print("=" * 60)

    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = 255
    rgba[4:8, 4:8, 1] = 100

    depth = np.zeros((32, 32), dtype=np.uint16)
    depth[10:20, 10:20] = 1000
    depth[20:30, 20:30] = 4000

    jxl = jxlpy.encode(rgba, extra_channels=[("depth", "depth", depth)])
    print(f"  Encoded: {len(jxl)} bytes")

    # decode_extra_channel: index 0 = alpha, index 1 = depth
    alpha_plane, alpha_meta = jxlpy.decode_extra_channel(jxl, 0)
    assert alpha_meta["extra_channel_type"] == "alpha", f"expected alpha, got {alpha_meta['extra_channel_type']}"
    expected_alpha = rgba[..., 3]
    assert np.array_equal(alpha_plane, expected_alpha), "alpha mismatch"
    print(f"  alpha (index 0): OK (type={alpha_meta['extra_channel_type']})")

    depth_plane, depth_meta = jxlpy.decode_extra_channel(jxl, 1)
    assert depth_meta["extra_channel_type"] == "depth", f"expected depth, got {depth_meta['extra_channel_type']}"
    assert np.array_equal(depth_plane, depth), "depth mismatch"
    assert depth_plane.dtype == np.uint16, f"depth dtype: {depth_plane.dtype}"
    print(f"  depth (index 1): OK (type={depth_meta['extra_channel_type']}, dtype={depth_plane.dtype})")

    # return_extra_channels: default skips alpha
    img, meta = jxlpy.decode(jxl, return_info=True, return_extra_channels=True)
    assert np.array_equal(img, rgba), "color mismatch"
    assert len(meta["extra_channels"]) == 1, f"expected 1 (alpha skipped), got {len(meta['extra_channels'])}"
    assert meta["extra_channels"][0]["type"] == "depth"
    assert np.array_equal(meta["extra_channels"][0]["data"], depth)
    print(f"  return_extra_channels (skip alpha): OK ({len(meta['extra_channels'])} extras)")

    # include_alpha_extra=True
    img2, meta2 = jxlpy.decode(jxl, return_info=True, return_extra_channels=True, include_alpha_extra=True)
    assert len(meta2["extra_channels"]) == 2, f"expected 2 (alpha included), got {len(meta2['extra_channels'])}"
    assert meta2["extra_channels"][0]["type"] == "alpha"
    assert meta2["extra_channels"][1]["type"] == "depth"
    print(f"  return_extra_channels (include alpha): OK ({len(meta2['extra_channels'])} extras)")

    print("  RESULT: PASS")
    print()
    return True


def test_multiple_extra_channels():
    """Multiple extra channels with different types and dtypes."""
    print("=" * 60)
    print("TEST: Multiple extra channels (mask + depth + thermal)")
    print("=" * 60)

    rgb = np.zeros((48, 48, 3), dtype=np.uint8)
    rgb[..., 0] = 50
    rgb[..., 1] = 100
    rgb[..., 2] = 150

    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[10:20, 10:20] = 255

    depth = np.zeros((48, 48), dtype=np.uint16)
    depth[20:30, 20:30] = 5000

    thermal = np.zeros((48, 48), dtype=np.uint16)
    thermal[30:40, 30:40] = 300

    jxl = jxlpy.encode(rgb, extra_channels=[
        ("mask", "selection_mask", mask),
        ("depth", "depth", depth),
        {"name": "thermal", "type": "thermal", "data": thermal},
    ])
    print(f"  Encoded: {len(jxl)} bytes")

    # Verify each via decode_extra_channel
    for i, (name, expected, expected_type) in enumerate([
        ("mask", mask, "selection_mask"),
        ("depth", depth, "depth"),
        ("thermal", thermal, "thermal"),
    ]):
        plane, meta = jxlpy.decode_extra_channel(jxl, i)
        assert meta["extra_channel_type"] == expected_type, f"{name}: type {meta['extra_channel_type']}"
        assert np.array_equal(plane, expected), f"{name}: value mismatch"
        print(f"  decode_extra_channel({i}): {name} OK (type={expected_type}, dtype={plane.dtype})")

    # Verify via return_extra_channels (single-pass)
    img, meta = jxlpy.decode(jxl, return_info=True, return_extra_channels=True)
    assert np.array_equal(img, rgb), "color mismatch"
    assert len(meta["extra_channels"]) == 3, f"expected 3 extras, got {len(meta['extra_channels'])}"
    for i, (name, expected, expected_type) in enumerate([
        ("mask", mask, "selection_mask"),
        ("depth", depth, "depth"),
        ("thermal", thermal, "thermal"),
    ]):
        ec = meta["extra_channels"][i]
        assert ec["name"] == name, f"{name}: name {ec['name']}"
        assert ec["type"] == expected_type, f"{name}: type {ec['type']}"
        assert np.array_equal(ec["data"], expected), f"{name}: value mismatch"
        print(f"  return_extra_channels[{i}]: {name} OK (type={ec['type']}, dtype={ec['data'].dtype})")

    print("  RESULT: PASS")
    print()
    return True


def test_extra_channels_multiframe():
    """Multi-frame with extra channels, verify single-pass decode per frame."""
    print("=" * 60)
    print("TEST: Multi-frame + extra channels (single-pass decode)")
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

    jxl = jxlpy.encode_multiframe(
        frames, reference="auto", effort=3,
        extra_channels=[("mask", "selection_mask", masks)],
    )
    print(f"  Encoded 3 frames + extra -> {len(jxl)} bytes")

    all_ok = True
    for i in range(3):
        # Single-pass decode with extra channels
        img, meta = jxlpy.decode(jxl, frame=i, return_info=True, return_extra_channels=True)
        if not np.array_equal(img, frames[i]):
            print(f"  Frame {i} color: MISMATCH")
            all_ok = False
        else:
            print(f"  Frame {i} color: OK")

        if len(meta["extra_channels"]) != 1:
            print(f"  Frame {i} extra: expected 1, got {len(meta['extra_channels'])}")
            all_ok = False
            continue

        ec = meta["extra_channels"][0]
        if not np.array_equal(ec["data"], masks[i]):
            print(f"  Frame {i} mask: MISMATCH")
            all_ok = False
        else:
            print(f"  Frame {i} mask: OK (type={ec['type']})")

        # Cross-check with decode_extra_channel
        plane, _ = jxlpy.decode_extra_channel(jxl, 1, frame=i)
        if not np.array_equal(plane, masks[i]):
            print(f"  Frame {i} mask (decode_extra_channel): MISMATCH")
            all_ok = False

    print(f"  RESULT: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


def test_extra_channel_float32():
    """Extra channel with float32 dtype."""
    print("=" * 60)
    print("TEST: Extra channel with float32 dtype")
    print("=" * 60)

    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[..., 0] = 200

    heatmap = np.zeros((16, 16), dtype=np.float32)
    heatmap[4:8, 4:8] = 1.5
    heatmap[8:12, 8:12] = 3.7

    jxl = jxlpy.encode(rgb, extra_channels=[("heat", "thermal", heatmap)])
    print(f"  Encoded: {len(jxl)} bytes")

    # decode_extra_channel
    plane, meta = jxlpy.decode_extra_channel(jxl, 0)
    assert plane.dtype == np.float32, f"dtype: {plane.dtype}"
    assert np.array_equal(plane, heatmap), "float32 heatmap mismatch"
    print(f"  decode_extra_channel: OK (dtype={plane.dtype})")

    # return_extra_channels
    img, meta2 = jxlpy.decode(jxl, return_info=True, return_extra_channels=True)
    assert np.array_equal(img, rgb), "color mismatch"
    ec = meta2["extra_channels"][0]
    assert ec["data"].dtype == np.float32, f"ec dtype: {ec['data'].dtype}"
    assert np.array_equal(ec["data"], heatmap), "float32 heatmap mismatch via return_extra_channels"
    print(f"  return_extra_channels: OK (dtype={ec['data'].dtype})")

    print("  RESULT: PASS")
    print()
    return True


def test_anonymous_extra_channel():
    """Extra channel with 'optional' type (no name)."""
    print("=" * 60)
    print("TEST: Anonymous extra channel (optional type)")
    print("=" * 60)

    rgb = np.zeros((16, 16, 3), dtype=np.uint8)
    rgb[..., 1] = 100

    plane_data = np.zeros((16, 16), dtype=np.uint8)
    plane_data[0:4, 0:4] = 200

    # Use 'optional' type which libjxl supports for generic extra channels
    jxl = jxlpy.encode(rgb, extra_channels=[("", "optional", plane_data)])
    print(f"  Encoded: {len(jxl)} bytes")

    img, meta = jxlpy.decode(jxl, return_info=True, return_extra_channels=True)
    assert np.array_equal(img, rgb), "color mismatch"
    assert len(meta["extra_channels"]) == 1
    ec = meta["extra_channels"][0]
    assert np.array_equal(ec["data"], plane_data), "optional extra mismatch"
    assert ec["type"] == "optional", f"expected optional, got {ec['type']}"
    print(f"  return_extra_channels: OK (type={ec['type']}, name='{ec['name']}')")

    # Cross-check with decode_extra_channel
    plane, ec_meta = jxlpy.decode_extra_channel(jxl, 0)
    assert np.array_equal(plane, plane_data), "optional extra mismatch via decode_extra_channel"
    assert ec_meta["extra_channel_type"] == "optional"
    print(f"  decode_extra_channel: OK (type={ec_meta['extra_channel_type']})")

    print("  RESULT: PASS")
    print()
    return True


def test_consistency_old_vs_new():
    """Verify that decode_extra_channel and return_extra_channels give same results."""
    print("=" * 60)
    print("TEST: Consistency — decode_extra_channel vs return_extra_channels")
    print("=" * 60)

    rgba = np.zeros((32, 32, 4), dtype=np.uint8)
    rgba[..., 0] = 50
    rgba[..., 1] = 100
    rgba[..., 3] = 255
    rgba[5:10, 5:10, 2] = 200

    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    mask[20:30, 20:30] = 128

    depth = np.zeros((32, 32), dtype=np.uint16)
    depth[0:16, 0:16] = 800

    jxl = jxlpy.encode(rgba, extra_channels=[
        ("mask", "selection_mask", mask),
        ("depth", "depth", depth),
    ])
    print(f"  Encoded: {len(jxl)} bytes")

    # Old path: decode_extra_channel per channel
    old_results = []
    for i in range(3):  # alpha(0), mask(1), depth(2)
        plane, meta = jxlpy.decode_extra_channel(jxl, i, return_info=True)
        old_results.append((plane.copy(), meta))

    # New path: single-pass
    img, meta = jxlpy.decode(jxl, return_info=True, return_extra_channels=True, include_alpha_extra=True)
    assert np.array_equal(img, rgba), "color mismatch"

    all_ok = True
    for i, ec in enumerate(meta["extra_channels"]):
        old_plane, old_meta = old_results[i]
        if not np.array_equal(ec["data"], old_plane):
            print(f"  Channel {i} ({ec['name']}): DATA MISMATCH")
            all_ok = False
        elif ec["type"] != old_meta["extra_channel_type"]:
            print(f"  Channel {i} ({ec['name']}): TYPE MISMATCH ({ec['type']} vs {old_meta['extra_channel_type']})")
            all_ok = False
        else:
            print(f"  Channel {i} ({ec['name']}): consistent (type={ec['type']}, dtype={ec['data'].dtype})")

    print(f"  RESULT: {'PASS' if all_ok else 'FAIL'}")
    print()
    return all_ok


def main():
    results = []
    results.append(("single_extra", test_single_extra_channel()))
    results.append(("rgba_alpha", test_rgba_alpha_extra()))
    results.append(("multiple_extras", test_multiple_extra_channels()))
    results.append(("multiframe_extras", test_extra_channels_multiframe()))
    results.append(("float32_extra", test_extra_channel_float32()))
    results.append(("anonymous_extra", test_anonymous_extra_channel()))
    results.append(("consistency", test_consistency_old_vs_new()))

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, ok in results:
        print(f"  {name:30s} {'PASS' if ok else 'FAIL'}")

    if all(ok for _, ok in results):
        print("\nAll extra channel tests passed.")
        return 0
    else:
        print("\nSome tests FAILED.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
