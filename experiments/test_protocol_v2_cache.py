"""CPU-only tests for the protocol-v2 two-layer cache."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

from protocol_v2_cache import ManifestCache, PixelCache, decoded_rgb_key, sha256_file


class CacheTests(unittest.TestCase):
    @staticmethod
    def temporary_directory() -> tempfile.TemporaryDirectory:
        # Keep tests inside the selected workspace. Some managed Windows
        # environments expose the system temp directory as read-only.
        return tempfile.TemporaryDirectory(dir=Path.cwd())

    def test_pixel_key_ignores_png_metadata(self) -> None:
        with self.temporary_directory() as directory:
            root = Path(directory)
            first, second = root / "first.png", root / "second.png"
            image = Image.new("RGB", (8, 6), (10, 20, 30))
            image.save(first)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("manifest", "different-container-metadata")
            image.save(second, pnginfo=metadata)
            first_key, first_info = decoded_rgb_key(first)
            second_key, second_info = decoded_rgb_key(second)
            self.assertEqual(first_key, second_key)
            self.assertEqual(first_info, second_info)
            self.assertNotEqual(sha256_file(first), sha256_file(second))

    def test_pixel_cache_preserves_bits_and_float16_logits(self) -> None:
        with self.temporary_directory() as directory:
            cache = PixelCache(Path(directory) / "pixels")
            logits = np.linspace(0, 1, 35, dtype=np.float32).reshape(5, 7)
            cache.put(
                key="a" * 64,
                decoded_bits="01" * 16,
                mask_logits=logits,
                image={"width": 7, "height": 5, "mode": "RGB"},
                model_sha256="b" * 64,
                checkpoint_sha256="c" * 64,
                preprocessing_version="wam-default-transform-v1",
            )
            loaded = cache.get("a" * 64)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["decoded_bits"], "01" * 16)
            self.assertEqual(loaded["mask_logits"].dtype, np.float16)
            np.testing.assert_array_equal(loaded["mask_logits"], logits.astype(np.float16))

    def test_manifest_cache_is_independent(self) -> None:
        with self.temporary_directory() as directory:
            cache = ManifestCache(Path(directory) / "manifests")
            key = "d" * 64
            cache.put(manifest_sha256=key, payload={"lineage": "ancestor", "evidence": "region-removed"})
            loaded = cache.get(key)
            self.assertEqual(loaded["lineage"], "ancestor")
            self.assertEqual(loaded["evidence"], "region-removed")


if __name__ == "__main__":
    unittest.main()
