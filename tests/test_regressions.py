from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import jxlpy
from jxlpy._ffi import ffi, lib


class NativeAbiTests(unittest.TestCase):
    def test_native_abi_matches_cffi_layout(self) -> None:
        self.assertEqual(lib.jxlpy_abi_version(), 2)
        layouts = {
            1: "jxlpy_encode_options",
            2: "jxlpy_extra_channel",
            3: "jxlpy_frame",
            4: "jxlpy_result",
            5: "jxlpy_extra_channel_result",
            6: "jxlpy_decode_all_result",
            7: "jxlpy_color_encoding",
            8: "jxlpy_encoder_setting",
        }
        for struct_id, name in layouts.items():
            self.assertEqual(lib.jxlpy_abi_struct_size(struct_id), ffi.sizeof(name))


class LayerAndDecodeTests(unittest.TestCase):
    @staticmethod
    def frames(count: int = 5) -> list[np.ndarray]:
        frames = []
        for index in range(count):
            image = np.zeros((32, 40, 4), dtype=np.uint8)
            image[..., 3] = 255
            image[4 + index : 8 + index, 6:12, :3] = index * 31
            frames.append(image)
        return frames

    def test_zero_duration_frames_are_layers(self) -> None:
        frames = self.frames(3)
        masks = [frame[..., 0].copy() for frame in frames]
        encoded = jxlpy.encode_multiframe(
            frames,
            durations=None,
            reference="previous",
            effort=3,
            keep_invisible=True,
            extra_channels=[{"type": "selection_mask", "data": masks}],
        )
        metadata = jxlpy.info(encoded)
        self.assertFalse(metadata["have_animation"])
        self.assertEqual(metadata["num_frames"], 3)
        canvas = np.zeros_like(frames[0])
        for index, expected in enumerate(frames):
            decoded, layer_info = jxlpy.decode_layer(encoded, layer=index)
            self.assertEqual(layer_info["duration"], 0)
            x0 = layer_info["crop_x0"]
            y0 = layer_info["crop_y0"]
            canvas[y0 : y0 + decoded.shape[0], x0 : x0 + decoded.shape[1]] = decoded
            self.assertTrue(np.array_equal(canvas, expected))
            mask, _ = jxlpy.decode_extra_channel(
                encoded, 0, frame=index, coalesced=False
            )
            self.assertEqual(mask.shape, decoded.shape[:2])

    def test_target_frame_stops_early_unless_full_scan_requested(self) -> None:
        encoded = jxlpy.encode_multiframe(
            self.frames(), durations=1, reference="none", effort=3
        )
        _, fast = jxlpy.decode(encoded, frame=2, return_info=True)
        self.assertFalse(fast["num_frames_known"])
        self.assertEqual(fast["num_frames"], 3)

        _, exact = jxlpy.decode(
            encoded, frame=2, return_info=True, scan_all_frames=True
        )
        self.assertTrue(exact["num_frames_known"])
        self.assertEqual(exact["num_frames"], 5)

    def test_decode_limits_and_thread_argument(self) -> None:
        encoded = jxlpy.encode(self.frames(1)[0], effort=3)
        decoded = jxlpy.decode(encoded, threads=1, max_pixels=32 * 40)
        self.assertEqual(decoded.shape, (32, 40, 4))
        with self.assertRaisesRegex(RuntimeError, "max_pixels"):
            jxlpy.decode(encoded, max_pixels=100)


class MetadataTests(unittest.TestCase):
    def test_icc_profile_roundtrip(self) -> None:
        try:
            from PIL import ImageCms
        except ImportError:
            self.skipTest("Pillow is not installed")
        profile = ImageCms.ImageCmsProfile(
            ImageCms.createProfile("sRGB")
        ).tobytes()
        image = np.zeros((12, 13, 3), dtype=np.uint8)
        encoded = jxlpy.encode(image, effort=3, icc_profile=profile)
        metadata = jxlpy.info(encoded)
        self.assertTrue(metadata["color_profile_is_icc"])
        self.assertEqual(metadata["icc_profile"], profile)

    def test_unspecified_extra_channel_is_optional(self) -> None:
        image = np.zeros((8, 9, 3), dtype=np.uint8)
        plane = np.arange(8 * 9, dtype=np.uint8).reshape(8, 9)
        encoded = jxlpy.encode(image, extra_channels=[plane], effort=3)
        metadata = jxlpy.info(encoded)
        self.assertEqual(metadata["extra_channels"][0]["type"], "optional")
        decoded, channel = jxlpy.decode_extra_channel(encoded, 0)
        self.assertEqual(channel["extra_channel_type"], "optional")
        self.assertTrue(np.array_equal(decoded, plane))

        with self.assertRaisesRegex(ValueError, "decoder-only"):
            jxlpy.encode(
                image,
                extra_channels=[{"type": "unknown", "data": plane}],
                effort=3,
            )

    def test_structured_color_roundtrip(self) -> None:
        image = np.arange(24 * 20 * 3, dtype=np.uint8).reshape(24, 20, 3)
        encoded = jxlpy.encode(image, effort=3, color_encoding="linear_srgb")
        metadata = jxlpy.info(encoded)
        self.assertFalse(metadata["color_profile_is_icc"])
        self.assertEqual(metadata["color_encoding"]["transfer_function"], "linear")

    def test_extra_channel_metadata_roundtrip(self) -> None:
        image = np.zeros((24, 20, 3), dtype=np.uint8)
        plane = np.repeat(
            np.repeat(np.arange(12 * 10, dtype=np.uint16).reshape(12, 10), 2, axis=0),
            2,
            axis=1,
        )
        encoded = jxlpy.encode(
            image,
            effort=3,
            extra_channels=[
                {
                    "name": "depth",
                    "type": "depth",
                    "data": plane,
                    "bits_per_sample": 16,
                    "dim_shift": 1,
                },
                {
                    "name": "ink",
                    "type": "spot_color",
                    "data": plane,
                    "spot_color": (0.1, 0.2, 0.3, 0.8),
                },
                {
                    "name": "mosaic",
                    "type": "cfa",
                    "data": plane,
                    "cfa_channel": 2,
                },
                {
                    "name": "matte",
                    "type": "alpha",
                    "data": plane,
                    "alpha_premultiplied": True,
                },
            ],
        )
        header_metadata = jxlpy.info(encoded)
        header_channels = {
            channel["name"]: channel
            for channel in header_metadata["extra_channels"]
        }
        self.assertEqual(header_channels["depth"]["dim_shift"], 1)
        self.assertEqual(header_channels["mosaic"]["cfa_channel"], 2)
        self.assertTrue(header_channels["matte"]["alpha_premultiplied"])
        _, metadata = jxlpy.decode(
            encoded,
            return_info=True,
            return_extra_channels=True,
            include_alpha_extra=True,
        )
        channels = {channel["name"]: channel for channel in metadata["extra_channels"]}
        self.assertEqual(channels["depth"]["dim_shift"], 1)
        for actual, expected in zip(
            channels["ink"]["spot_color"], (0.1, 0.2, 0.3, 0.8)
        ):
            self.assertAlmostEqual(actual, expected, places=3)
        self.assertEqual(channels["mosaic"]["cfa_channel"], 2)
        self.assertTrue(channels["matte"]["alpha_premultiplied"])
        self.assertEqual(channels["depth"]["data"].shape, plane.shape)
        self.assertEqual(channels["depth"]["data"].dtype, plane.dtype)


class AnalysisTests(unittest.TestCase):
    def test_sampling_uses_real_neighbors(self) -> None:
        checker = np.indices((512, 512)).sum(axis=0).astype(np.uint8) & 1
        image = np.repeat((checker * 255)[:, :, None], 3, axis=2)
        metrics = jxlpy.analyze_pixels(image, max_sample_pixels=65_536)
        self.assertLess(metrics.flat4_pct, 1.0)
        self.assertGreater(metrics.edge_mean, 250.0)

    def test_jpeg_path_and_bytes_get_same_source_format(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[8:24, 8:24] = 200
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jpg"
            Image.fromarray(image).save(path, format="JPEG")
            from_path = jxlpy.analyze_lossless(path)
            from_bytes = jxlpy.analyze_lossless(path.read_bytes())
        self.assertEqual(from_path.source_format, "jpg")
        self.assertEqual(from_bytes.source_format, "jpeg")
        self.assertEqual(from_path.recommendation.profile, "jpeg_transcode")
        self.assertEqual(from_bytes.recommendation.profile, "jpeg_transcode")

    def test_torch_input_when_available(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        tensor = torch.zeros((3, 32, 32), dtype=torch.uint8)
        metrics = jxlpy.analyze_pixels(tensor, layout="chw")
        analysis = jxlpy.analyze_lossless(tensor, layout="chw")
        self.assertEqual(metrics.channels, 3)
        self.assertEqual(analysis.source_format, "pixels")


if __name__ == "__main__":
    unittest.main()
