"""Unit tests for the editor-trust policy-only ablation."""
from __future__ import annotations

import unittest

from protocol_v2_geometry_trust_ablation_v1 import evaluate_row


def raw_row(*, gate: str = "accepted", evidence: str = "silent-other",
            policy: str = "silent-other") -> dict:
    return {
        "run_id": "case__adaptive-high-logit",
        "base_case_id": "case",
        "dataset": "centered_crop_20pct_per_side",
        "dataset_index": 0,
        "condition": "adaptive-high-logit",
        "manifest_sha256": "a" * 64,
        "manifest_valid": True,
        "geometry_gate": gate,
        "lineage": "ancestor",
        "semantics": "no-ai-disclosure-in-chain",
        "evidence": evidence,
        "policy_verdict": policy,
        "baseline_evidence": "signal-suppressed",
        "baseline_policy_verdict": "review-signal-suppression",
    }


class TrustAblationTests(unittest.TestCase):
    def test_trusted_editor_preserves_raw_bypass(self) -> None:
        result = evaluate_row(raw_row(), editor_trust="trusted-editor")
        self.assertEqual(result["policy_verdict"], "silent-other")
        self.assertTrue(result["pixel_evidence_consumed"])
        self.assertTrue(result["trust_gated_review_evasion"])

    def test_untrusted_editor_is_fail_closed_without_pixel_evidence(self) -> None:
        result = evaluate_row(raw_row(), editor_trust="untrusted-editor")
        self.assertEqual(result["evidence"], "not-evaluated-untrusted-geometry")
        self.assertEqual(result["policy_verdict"], "review-untrusted-geometry")
        self.assertFalse(result["pixel_evidence_consumed"])
        self.assertFalse(result["trust_gated_review_evasion"])

    def test_invalid_geometry_precedes_trust_policy(self) -> None:
        result = evaluate_row(
            raw_row(
                gate="reject-invalid",
                evidence="not-evaluated-invalid-geometry",
                policy="geometry-rejected",
            ),
            editor_trust="untrusted-editor",
        )
        self.assertEqual(result["geometry_trust_gate"], "schema-rejected")
        self.assertEqual(result["policy_verdict"], "geometry-rejected")
        self.assertFalse(result["pixel_evidence_consumed"])


if __name__ == "__main__":
    unittest.main()
