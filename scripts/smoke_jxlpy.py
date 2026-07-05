import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jxlpy


def main() -> None:
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = 255
    encoded = jxlpy.encode(rgba)
    decoded, meta = jxlpy.decode(encoded, return_info=True)
    print("single_size", len(encoded))
    print("single_shape", decoded.shape, decoded.dtype, bool(np.array_equal(decoded, rgba)))
    print("single_meta", meta["xsize"], meta["ysize"], meta["num_frames"])

    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 255
    with_extra = jxlpy.encode(
        rgba,
        extra_channels=[("mask", "selection_mask", mask)],
    )
    extra_plane, extra_meta = jxlpy.decode_extra_channel(with_extra, 1)
    decoded_extra, decoded_extra_meta = jxlpy.decode(
        with_extra,
        return_info=True,
        return_extra_channels=True,
    )
    print("extra_size", len(with_extra))
    print("extra_direct", extra_plane.shape, extra_meta["extra_channel_type"], bool(np.array_equal(extra_plane, mask)))
    print("extra_meta", len(decoded_extra_meta["extra_channels"]), decoded_extra_meta["extra_channels"][0]["name"])
    print("extra_main", bool(np.array_equal(decoded_extra, rgba)))

    frames = [rgba.copy() for _ in range(3)]
    frames[1][4:8, 4:8, 1] = 128
    frames[2][8:12, 8:12, 2] = 200
    masks = [mask.copy() for _ in frames]
    masks[2][10:12, 10:12] = 64
    multi = jxlpy.encode_multiframe(
        frames,
        reference="auto",
        extra_channels=[("mask", "selection_mask", masks)],
    )
    roundtrip1 = jxlpy.decode(multi, frame=1)
    roundtrip, meta2 = jxlpy.decode(multi, frame=2, return_info=True)
    layer, layer_meta = jxlpy.decode_layer(multi, layer=1)
    mask2, mask2_meta = jxlpy.decode_extra_channel(multi, 1, frame=2)
    print("multi_size", len(multi))
    print("multi_frame1", roundtrip1.shape, roundtrip1.dtype, bool(np.array_equal(roundtrip1, frames[1])))
    print("multi_frame2", roundtrip.shape, roundtrip.dtype, bool(np.array_equal(roundtrip, frames[2])))
    print("layer1", layer.shape, layer_meta["layer_have_crop"], layer_meta["crop_x0"], layer_meta["crop_y0"])
    print("multi_extra2", mask2.shape, mask2_meta["extra_channel_type"], bool(np.array_equal(mask2, masks[2])))


if __name__ == "__main__":
    main()
