import unittest

import numpy as np

from benign_controls_common_v1 import (
    classify_benign,
    crop_grid,
    select_max_retention_crop,
)


class BenignControlsCommonTests(unittest.TestCase):
    def test_crop_selection_uses_geometry_only(self):
        grid = np.zeros((64, 64), dtype=np.float32)
        grid[10:30, 0:20] = 1
        side, retention = select_max_retention_crop(grid, 0.05)
        self.assertEqual(side, "right")
        self.assertEqual(retention, 1.0)

    def test_crop_grid_removes_requested_side(self):
        grid = np.ones((64, 64), dtype=np.float32)
        self.assertEqual(crop_grid(grid, "left", 0.05).shape, (64, 61))
        self.assertEqual(crop_grid(grid, "bottom", 0.05).shape, (61, 64))

    def test_honest_watermarked_asset_is_not_flagged_when_decoder_succeeds(self):
        verdict, flagged = classify_benign(
            expected_region_assertion=True,
            manifest_valid=True,
            assertion_present=True,
            payload_hash_ok=True,
            bit_accuracy=0.9,
            region_retention=1.0,
            logit_in_region=0.9,
        )
        self.assertEqual(verdict, "CONSISTENT_SYNTHETIC")
        self.assertFalse(flagged)

    def test_benign_signal_loss_is_counted_as_a_false_flag(self):
        verdict, flagged = classify_benign(
            expected_region_assertion=True,
            manifest_valid=True,
            assertion_present=True,
            payload_hash_ok=True,
            bit_accuracy=0.5,
            region_retention=1.0,
            logit_in_region=0.2,
        )
        self.assertEqual(verdict, "WATERMARK_SUPPRESSED")
        self.assertTrue(flagged)

    def test_clean_negative_without_region_claim_is_not_an_assertion_failure(self):
        verdict, flagged = classify_benign(
            expected_region_assertion=False,
            manifest_valid=True,
            assertion_present=False,
            payload_hash_ok=None,
            bit_accuracy=0.8,
            region_retention=None,
            logit_in_region=0.2,
        )
        self.assertEqual(verdict, "NO_REGION_CLAIM")
        self.assertFalse(flagged)


if __name__ == "__main__":
    unittest.main()
