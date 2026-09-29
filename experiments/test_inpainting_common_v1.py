import unittest

from inpainting_common_v1 import (
    classify_inpainting,
    select_stratified_indices,
    wilson_interval,
)


class InpaintingCommonTests(unittest.TestCase):
    def test_stratified_selection_includes_extremes(self):
        coverages = [0.5, 0.1, 0.9, 0.3, 0.7]
        selected = select_stratified_indices(coverages, 3)
        self.assertEqual([coverages[index] for index in selected], [0.1, 0.5, 0.9])

    def test_signal_check_isolated_when_retention_is_one(self):
        baseline, region = classify_inpainting(0.5, True, True, 1.0, 0.2)
        self.assertEqual(baseline, "SILENT_ESCAPE")
        self.assertEqual(region, "WATERMARK_SUPPRESSED")

    def test_residual_escape_when_signal_check_does_not_fire(self):
        baseline, region = classify_inpainting(0.5, True, True, 1.0, 0.8)
        self.assertEqual(baseline, "SILENT_ESCAPE")
        self.assertEqual(region, "SILENT_OTHER")

    def test_wilson_interval_contains_observed_rate(self):
        low, high = wilson_interval(5, 10)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)


if __name__ == "__main__":
    unittest.main()
