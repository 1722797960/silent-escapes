from __future__ import annotations

import unittest

from protocol_v2_main_curve_full_v1 import legacy_evidence, trust_condition


class MainCurveFullTest(unittest.TestCase):
    def test_legacy_evidence_order(self) -> None:
        base = {"detected": False, "region_retention": 1.0, "logit_in_region": 1.0}
        self.assertEqual(legacy_evidence({**base, "detected": True}), "message-detected")
        self.assertEqual(legacy_evidence({**base, "region_retention": 0.01}), "region-removed")
        self.assertEqual(legacy_evidence({**base, "logit_in_region": 0.1}), "signal-suppressed")
        self.assertEqual(legacy_evidence(base), "silent-other")

    def test_untrusted_crop_fails_closed(self) -> None:
        raw = {
            "lineage": "ancestor",
            "semantics": "no-ai-disclosure-in-chain",
            "evidence": "silent-other",
            "policy_verdict": "silent-other",
        }
        result = trust_condition(
            raw,
            gate="accepted",
            cropped_action_count=1,
            editor_trust="untrusted-editor",
        )
        self.assertFalse(result["pixel_evidence_consumed"])
        self.assertEqual(result["policy_verdict"], "review-untrusted-geometry")


if __name__ == "__main__":
    unittest.main()
