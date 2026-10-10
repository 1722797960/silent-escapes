"""Unit tests for full declared-geometry aggregation and acceptance helpers."""
from __future__ import annotations

import unittest

from protocol_v2_geometry_lie_full_v1 import (
    condition_summary,
    correct_matches_baseline,
    transition_counts,
    wilson_interval,
)


def row(
    *,
    baseline_evidence: str,
    evidence: str,
    baseline_policy: str,
    policy: str,
    gate: str = "accepted",
    baseline_logit: float = 0.4,
    logit: float = 0.6,
) -> dict:
    return {
        "baseline_evidence": baseline_evidence,
        "evidence": evidence,
        "baseline_policy_verdict": baseline_policy,
        "policy_verdict": policy,
        "geometry_gate": gate,
        "manifest_valid": True,
        "pixel_cache_hit": True,
        "baseline_logit_in_region": baseline_logit,
        "logit_in_region": logit,
    }


class WilsonTests(unittest.TestCase):
    def test_zero_successes_has_nonzero_upper_bound(self) -> None:
        interval = wilson_interval(0, 200)
        self.assertEqual(interval["rate"], 0.0)
        self.assertEqual(interval["low"], 0.0)
        self.assertGreater(interval["high"], 0.0)

    def test_invalid_counts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            wilson_interval(2, 1)


class TransitionTests(unittest.TestCase):
    def test_transition_matrix_preserves_direction(self) -> None:
        rows = [
            row(
                baseline_evidence="region-removed",
                evidence="silent-other",
                baseline_policy="review-evidence-removal",
                policy="silent-other",
            ),
            row(
                baseline_evidence="region-removed",
                evidence="message-detected",
                baseline_policy="review-evidence-removal",
                policy="integrity-clash",
            ),
        ]
        transitions = transition_counts(
            rows, baseline_field="baseline_evidence", observed_field="evidence"
        )
        self.assertEqual(
            transitions,
            {"region-removed": {"silent-other": 1, "message-detected": 1}},
        )

    def test_condition_summary_counts_review_evasion(self) -> None:
        rows = [
            row(
                baseline_evidence="region-removed",
                evidence="silent-other",
                baseline_policy="review-evidence-removal",
                policy="silent-other",
            ),
            row(
                baseline_evidence="signal-suppressed",
                evidence="signal-suppressed",
                baseline_policy="review-signal-suppression",
                policy="review-signal-suppression",
            ),
        ]
        summary = condition_summary(rows)
        self.assertEqual(summary["policy_changes"]["successes"], 1)
        self.assertEqual(summary["new_silent_other"]["successes"], 1)
        self.assertEqual(summary["review_to_non_review"]["successes"], 1)
        self.assertEqual(summary["review_to_silent_other"]["successes"], 1)
        self.assertAlmostEqual(summary["logit_gain_mean"], 0.2)


class BaselineTests(unittest.TestCase):
    def test_correct_geometry_requires_metric_and_verdict_agreement(self) -> None:
        audit = {
            "lineage": "ancestor",
            "reference": "signed-ancestor",
            "semantics": "no-ai-disclosure-in-chain",
            "evidence": "message-detected",
            "policy_verdict": "integrity-clash",
        }
        baseline = {
            **audit,
            "region_retention": 0.8,
            "logit_in_region": 0.7,
        }
        self.assertTrue(
            correct_matches_baseline(
                audit, retention=0.8, regional_logit=0.7, baseline=baseline
            )
        )
        self.assertFalse(
            correct_matches_baseline(
                audit, retention=0.8, regional_logit=0.6, baseline=baseline
            )
        )


if __name__ == "__main__":
    unittest.main()
