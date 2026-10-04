from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from protocol_v2_cache import PixelCache
from protocol_v2_cache_warm_v1 import PREPROCESSING_VERSION, validate_entry


class CacheWarmTests(unittest.TestCase):
    def test_round_trip_validation_preserves_bits_and_logits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = PixelCache(Path(temporary))
            image = {"width": 16, "height": 8, "mode": "RGB"}
            logits = np.arange(12, dtype=np.float32).reshape(3, 4)
            cache.put(
                key="ab" * 32,
                decoded_bits="1010",
                mask_logits=logits,
                image=image,
                model_sha256="model",
                checkpoint_sha256="checkpoint",
                preprocessing_version=PREPROCESSING_VERSION,
            )
            entry = cache.get("ab" * 32)
            self.assertIsNotNone(entry)
            assert entry is not None
            validate_entry(
                entry,
                key="ab" * 32,
                image=image,
                model_sha256="model",
                checkpoint_sha256="checkpoint",
            )
            self.assertEqual(entry["decoded_bits"], "1010")
            np.testing.assert_array_equal(entry["mask_logits"], logits.astype(np.float16))


if __name__ == "__main__":
    unittest.main()
