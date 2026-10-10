import unittest
from copy import deepcopy

from protocol_v2_pixart_full_v1 import validate_cohort, validate_config
from protocol_v2_pixart_full_audit_v1 import summary_counts


class PixArtFullTests(unittest.TestCase):
    def setUp(self):
        self.cases = [{"case_id": f"pixart_{i}_{j}", "pixel_key": f"pixel_{i}_{j}", "image_index": i,
                       "crop_frac": round(j*.05, 2), "legacy_match_count": 23 if i*9+j < 940 else 24}
                      for i in range(200) for j in range(9)]

    def test_complete_cohort(self):
        validate_cohort(self.cases)

    def test_duplicate_pixels_rejected(self):
        self.cases[1]["pixel_key"] = self.cases[0]["pixel_key"]
        with self.assertRaises(RuntimeError):
            validate_cohort(self.cases)

    def test_wrong_crop_semantics_rejected(self):
        config = dict(expected_cases=1800, expected_images=200, expected_legacy_escapes=940,
                      required_matches=24, retention_floor=.05, suppression_threshold=.5,
                      crop_parameter="removed-area rho")
        with self.assertRaises(RuntimeError):
            validate_config(config)

    def test_wrong_image_crop_pair_rejected(self):
        self.cases[1]["image_index"], self.cases[9]["image_index"] = 1, 0
        with self.assertRaises(RuntimeError):
            validate_cohort(self.cases)

    def test_summary_keeps_residual_escape(self):
        rows = [{"case_id": c["case_id"], "manifest_valid": True, "state_checks": {"valid": True},
                 "match_count": c["legacy_match_count"],
                 "evidence": "silent-other" if c["legacy_match_count"] < 24 else "message-detected",
                 "conditions": {t: {"policy_verdict": "silent-other"} for t in
                                ("trusted-editor", "untrusted-editor")}} for c in self.cases]
        result = summary_counts(rows)
        self.assertEqual(result["trusted_editor_silent"], 940)
        self.assertEqual(result["trusted_editor_reviews"], 0)
        invalid = deepcopy(rows)
        invalid[0]["state_checks"]["valid"] = False
        with self.assertRaises(RuntimeError):
            summary_counts(invalid)


if __name__ == "__main__":
    unittest.main()
