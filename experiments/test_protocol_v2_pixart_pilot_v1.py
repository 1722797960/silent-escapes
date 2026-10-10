"""Boundary selection and audit-summary tests, without model weights or GPU."""
import unittest

from protocol_v2_pixart_pilot_v1 import choose_boundaries
from protocol_v2_pixart_audit_v1 import summarize


class PixArtPilotTests(unittest.TestCase):
    def fixtures(self):
        rows = []
        metadata = {f"im{i}": dict(mask_coverage=(i+1)/10) for i in range(8)}
        for i in range(8):
            for level in range(9):
                count = 32 if level == 0 else 24+(i % 3)-1
                if i == 0 and level == 0:
                    count = 13
                rows.append(dict(attack_id=f"pixart_c{level}_{i}", image=f"im{i}",
                                 image_index=i, crop_frac=round(level*.05, 2), legacy_match_count=count))
        return rows, metadata

    def test_twenty_deterministic_cases_cover_boundaries_and_controls(self):
        rows, metadata = self.fixtures()
        selected = choose_boundaries(rows, metadata)
        self.assertEqual(len(selected), 20)
        self.assertEqual(selected, choose_boundaries(list(reversed(rows)), metadata))
        self.assertEqual(len({r["crop_frac"] for r in selected}), 9)
        self.assertTrue({23, 24, 25} <= {r["legacy_match_count"] for r in selected})
        controls = [r for r in selected if r["crop_frac"] == 0]
        self.assertTrue(any(r["legacy_match_count"] < 24 for r in controls))
        self.assertTrue(any(r["legacy_match_count"] >= 24 for r in controls))

    def test_missing_boundary_stops_selection(self):
        rows, metadata = self.fixtures()
        with self.assertRaises(ValueError):
            choose_boundaries([r for r in rows if r["legacy_match_count"] != 24], metadata)

    def test_summary_requires_twenty_unique_valid_rows(self):
        def row(i):
            return dict(case_id=f"pixart{i}", match_count=23, manifest_valid=True,
                        evidence="signal-suppressed", crop_frac=.1,
                        conditions={"trusted-editor": dict(policy_verdict="review-signal-suppression"),
                                    "untrusted-editor": dict(policy_verdict="review-untrusted-geometry")},
                        state_checks=dict(valid=True))
        rows = [row(i) for i in range(20)]
        result = summarize(rows, 1)
        self.assertEqual((result["escapes"], result["trusted_editor_reviews"]), (20, 20))
        self.assertFalse(result["population_estimate"])
        with self.assertRaises(RuntimeError):
            summarize(rows[:-1], 1)
        rows[-1] = rows[-2]
        with self.assertRaises(RuntimeError):
            summarize(rows, 1)


if __name__ == "__main__":
    unittest.main()
