from __future__ import annotations

import math
import unittest

import numpy as np

from granularity_control_common_v1 import (
    centered_crop_box,
    energy_match_fullframe,
    mse,
    quantize_rgb01,
    random_compact_mask_same_area,
    stable_seed,
    translate_mask_same_area,
)


class GranularityControlTests(unittest.TestCase):
    def test_center_crop_matches_c1_definition(self):
        self.assertEqual(centered_crop_box(100, 100, 0.0), (0, 0, 100, 100))
        for rho in (0.1, 0.4, 0.8):
            box = centered_crop_box(1024, 1024, rho)
            side = round(1024 * math.sqrt(1 - rho))
            self.assertEqual(box[2] - box[0], side)
            self.assertEqual(box[3] - box[1], side)

    def test_translation_is_deterministic_and_area_preserving(self):
        mask = np.zeros((24, 32), dtype=bool)
        mask[3:9, 5:12] = True
        first, meta1 = translate_mask_same_area(mask, stable_seed(7, "x.png", "mask"))
        second, meta2 = translate_mask_same_area(mask, stable_seed(7, "x.png", "mask"))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(meta1, meta2)
        self.assertEqual(int(first.sum()), int(mask.sum()))
        self.assertFalse(meta1["translation_degenerate"])

    def test_full_canvas_bbox_uses_recorded_cyclic_translation(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[0, 0] = True
        mask[-1, -1] = True
        mask[5:10, 5:10] = True
        shifted, meta = translate_mask_same_area(mask, 12345)
        self.assertEqual(int(shifted.sum()), int(mask.sum()))
        self.assertEqual(meta["operation"], "cyclic_translation")
        self.assertTrue(meta["wrapped"])
        self.assertFalse(meta["translation_degenerate"])

    def test_compact_random_mask_is_exact_contiguous_and_deterministic(self):
        first, meta1 = random_compact_mask_same_area((64, 80), 2345, 9876)
        second, meta2 = random_compact_mask_same_area((64, 80), 2345, 9876)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(meta1, meta2)
        self.assertEqual(int(first.sum()), 2345)
        self.assertFalse(meta1["wrapped"])
        # Every occupied row is contiguous, and occupied rows are consecutive.
        occupied_rows = np.where(first.any(axis=1))[0]
        np.testing.assert_array_equal(
            occupied_rows, np.arange(occupied_rows[0], occupied_rows[-1] + 1))
        for row in first[occupied_rows]:
            xs = np.where(row)[0]
            np.testing.assert_array_equal(xs, np.arange(xs[0], xs[-1] + 1))

    def test_energy_match_uses_post_quantization_mse(self):
        original = np.full((32, 32, 3), 0.4, dtype=np.float64)
        candidate = np.full((32, 32, 3), 0.6, dtype=np.float64)
        subject = original.copy()
        subject[:16] = candidate[:16]
        target = mse(quantize_rgb01(subject), quantize_rgb01(original))
        matched, alpha, achieved = energy_match_fullframe(original, candidate, target)
        self.assertGreater(alpha, 0.0)
        self.assertLess(alpha, 1.0)
        self.assertLess(abs(achieved - target) / target, 0.01)
        self.assertAlmostEqual(mse(matched, quantize_rgb01(original)), achieved)

    def test_integer_threshold_boundary(self):
        threshold = 24
        self.assertFalse(23 >= threshold)
        self.assertTrue(24 >= threshold)


if __name__ == "__main__":
    unittest.main()
