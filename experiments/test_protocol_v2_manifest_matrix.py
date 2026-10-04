from __future__ import annotations

import unittest

from protocol_v2_manifest_matrix_v1 import CHAIN_RETAINED, expected_policy, summarize


class ManifestMatrixTests(unittest.TestCase):
    def test_policy_is_conditioned_only_after_evidence(self) -> None:
        self.assertEqual(
            expected_policy(
                lineage="ancestor",
                evidence="message-detected",
                semantics="ai-disclosed-in-chain",
            ),
            "consistent-synthetic",
        )
        self.assertEqual(
            expected_policy(
                lineage="ancestor",
                evidence="message-detected",
                semantics="no-ai-disclosure-in-chain",
            ),
            "integrity-clash",
        )
        self.assertEqual(
            expected_policy(
                lineage="ancestor",
                evidence="region-removed",
                semantics="ai-disclosed-in-chain",
            ),
            "review-evidence-removal",
        )
        self.assertEqual(
            expected_policy(
                lineage="discontinuity",
                evidence="no-reference",
                semantics="no-ai-disclosure-in-chain",
            ),
            "discontinuity",
        )

    def test_summary_requires_evidence_invariance_across_retained_chain(self) -> None:
        rows = []
        for condition in sorted(CHAIN_RETAINED):
            rows.append(
                {
                    "base_case_id": "case-1",
                    "condition": condition,
                    "manifest_valid": True,
                    "pixel_cache_hit": True,
                    "expected_state_match": True,
                    "expected_policy_match": True,
                    "evidence": "message-detected",
                    "policy_verdict": "consistent-synthetic",
                }
            )
        rows.append(
            {
                "base_case_id": "case-1",
                "condition": "cut-chain-fresh-root",
                "manifest_valid": True,
                "pixel_cache_hit": True,
                "expected_state_match": True,
                "expected_policy_match": True,
                "evidence": "no-reference",
                "policy_verdict": "discontinuity",
            }
        )
        summary = summarize(rows, expected_rows=5)
        self.assertEqual(summary["evidence_invariant_cases"], 1)
        self.assertEqual(summary["gpu_inference_calls"], 0)


if __name__ == "__main__":
    unittest.main()
