"""Regression tests for per-side grouping and signed geometry validation."""
import unittest

from protocol_v2_main_curve_cache_v2 import inventory
from protocol_v2_main_curve_full_v2 import signed_transform_matches
from protocol_v2_pilot_v1 import crop_transform


class StubCache:
    def contains(self, key):
        return key == "hit"


class MainCurveV2Tests(unittest.TestCase):
    def test_half_steps_remain_separate(self):
        rows = [
            {"crop_frac": c, "pixel_key": key}
            for c, key in ((0.05, "hit"), (0.10, "miss"), (0.15, "other"))
        ]
        result = inventory(rows, StubCache())
        self.assertEqual(set(result["per_crop"]), {"0.05", "0.10", "0.15"})
        self.assertEqual(result["cache_hits"], 1)
        self.assertEqual(result["cache_misses"], 2)

    def test_signed_geometry_must_match_written_parameters(self):
        case = {"crop_transform": crop_transform([1024, 1024], [205, 205, 819, 819], [1024, 1024])}
        inspection = {"geometry": {"status": "valid", "transform": {
            "source_size_px": [1024, 1024], "crop_box_px": [205, 205, 819, 819],
            "output_size_px": [1024, 1024], "resize_filter": "LANCZOS",
        }}}
        self.assertTrue(signed_transform_matches(case, inspection))
        inspection["geometry"]["transform"]["crop_box_px"][0] = 204
        self.assertFalse(signed_transform_matches(case, inspection))

    def test_identity_has_no_crop_action(self):
        self.assertTrue(signed_transform_matches({"crop_transform": None}, {"geometry": {"status": "missing"}}))
        self.assertFalse(signed_transform_matches({"crop_transform": None}, {"geometry": {"status": "valid"}}))


if __name__ == "__main__":
    unittest.main()
