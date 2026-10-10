"""CPU unit tests for integer matching, strict thresholds and bbox geometry."""
import unittest

from protocol_v2_sdxl_derived_v1 import (
    ablations, bbox_retention, classify, summarize, validate_cohort, wilson_interval,
)
from protocol_v2_pixart_inventory_v1 import frozen_reference


class DerivedTests(unittest.TestCase):
    def row(self, matches=23, retention=.05, logit=.5, crop=.1):
        return dict(match_count=matches, region_retention=retention,
                    logit_in_region=logit, crop_frac=crop, bbox_retention=retention)

    def test_integer_threshold_precedes_region(self):
        self.assertEqual(classify(self.row(24, 0, 0), .05, .5), "message-detected")
        self.assertEqual(classify(self.row(23, 0, 0), .05, .5), "region-removed")

    def test_strict_boundaries(self):
        self.assertEqual(classify(self.row(), .05, .5), "silent-other")
        self.assertEqual(classify(self.row(logit=.499999), .05, .5), "signal-suppressed")
        self.assertEqual(classify(self.row(retention=.049999), .05, .5), "region-removed")

    def test_bbox_signed_source_pixel_geometry(self):
        geometry = dict(status="valid", transform=dict(source_size_px=[100, 200],
                        crop_box_px=[25, 50, 75, 150]))
        self.assertEqual(bbox_retention(dict(bbox_csv="0,0,1,1"), geometry), .25)
        self.assertEqual(bbox_retention(dict(bbox_csv="0,.25,.25,.75"), geometry), 0)
        self.assertEqual(bbox_retention(dict(bbox_csv=".25,.25,.75,.75"), geometry), 1)

    def test_invalid_bbox_and_geometry(self):
        for bbox in ("0,0,0,1", "0,0,nan,1", "-1,0,1,1", "0,1"):
            with self.assertRaises(ValueError):
                bbox_retention(dict(bbox_csv=bbox), dict(status="missing"))
        with self.assertRaises(ValueError):
            bbox_retention(dict(bbox_csv="0,0,1,1"), dict(status="invalid"))

    def test_no_edit_not_false_positive(self):
        rows = [self.row(crop=0), self.row(logit=.1, crop=0), self.row(24, crop=0)]
        result = summarize(rows, .05, .5)
        self.assertEqual((result["escapes"], result["reviewed_escapes"]), (2, 1))
        self.assertEqual((result["no_edit_assets"], result["no_edit_reviews"]), (3, 1))

    def test_geometry_and_signal_gains_separated(self):
        row = self.row(retention=.01)
        row["bbox_retention"] = .1
        results = ablations([row, self.row(logit=.1), self.row(24)])
        self.assertEqual([r["reviewed_escapes"] for r in results], [0, 1, 2])

    def test_wilson_zero_is_not_zero_uncertainty(self):
        lo, hi = wilson_interval(0, 200)
        self.assertAlmostEqual(lo, 0)
        self.assertGreater(hi, 1.8)
        self.assertEqual(wilson_interval(0, 0), [None, None])

    def test_cohort_rejects_missing_and_duplicate_rows(self):
        rows = [dict(case_id=f"{i}_{j}", crop_frac=round(i*.05, 2))
                for i in range(9) for j in range(200)]
        validate_cohort(rows)
        with self.assertRaises(RuntimeError):
            validate_cohort(rows[:-1])
        rows[-1] = rows[-2]
        with self.assertRaises(RuntimeError):
            validate_cohort(rows)

    def test_pixart_count_boundary_and_reference(self):
        reference, matches = frozen_reference(dict(bit_accuracy=.75), dict(message_bits=[1]*32))
        self.assertEqual((reference, matches), ("1"*32, 24))
        with self.assertRaises(RuntimeError):
            frozen_reference(dict(bit_accuracy=.751), dict(message_bits=[1]*32))
        with self.assertRaises(RuntimeError):
            frozen_reference(dict(bit_accuracy=.75), dict(message_bits=[1]*31))


if __name__ == "__main__":
    unittest.main()
