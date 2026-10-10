"""CPU synthetic tests for full C1 cohort, resumability, budget and statistics."""
import json
import subprocess
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from protocol_v2_c1_full_v1 import (
    MAX_INFERENCES, inference_starts, indexed_rows, obtain_raw, pipeline, validate_cohort,
    validate_config, validate_raw,
    require_new_cache_slot,
)
from protocol_v2_c1_full_audit_v1 import group_counts
from protocol_v2_c1_full_derived_v1 import (
    cell_statistics, clustered_review_interval, directional_transitions, mask_aware_rows, paired_escape_difference,
)
from protocol_v2_c1_pilot_v1 import METHODS, RHO_LEVELS


def fixture_cases():
    cases = []
    for image in range(200):
        for rho in sorted(RHO_LEVELS):
            for method in sorted(METHODS):
                identity = f"{image:04d}|{method}|{rho:.3f}"
                cases.append({"case_id": identity, "attack_id": identity, "image_index": image,
                              "method": method, "rho_requested": rho,
                              "pixel_key": f"{image}|zero" if rho == 0 else identity,
                              "legacy_match_count": 23 if len(cases) < 2774 else 24})
    return cases


def evidence_row(image, *, escaped=True, reviewed=True):
    evidence = "signal-suppressed" if reviewed else "silent-other"
    if not escaped:
        evidence = "message-detected"
    return {"image_index": image, "match_count": 23 if escaped else 24, "evidence": evidence,
            "conditions": {t: {"policy_verdict": evidence} for t in ("trusted-editor", "untrusted-editor")}}


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.cases = fixture_cases()

    def test_exact_cartesian_cohort_and_unique_pixels(self):
        validate_cohort(self.cases)
        self.assertEqual(len({c["pixel_key"] for c in self.cases}), 9800)

    def test_wrong_rho_or_method_is_rejected(self):
        for field, value in (("rho_requested", .05), ("method", "mask-aware"), ("image_index", 201)):
            cases = deepcopy(self.cases)
            cases[0][field] = value
            with self.assertRaises(RuntimeError):
                validate_cohort(cases)

    def test_missing_or_duplicate_record_is_rejected(self):
        with self.assertRaises(RuntimeError):
            validate_cohort(self.cases[:-1])
        self.cases[-1] = self.cases[-2]
        with self.assertRaises(RuntimeError):
            validate_cohort(self.cases)

    def test_fixed_config_budget_and_thresholds(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs/protocol_v2_c1_full_v1.json").read_text())
        validate_config(config)
        for field, value in (("max_new_inference_calls", 9800), ("required_matches", 23),
                             ("crop_parameter", "per-side c")):
            invalid = {**config, field: value}
            with self.assertRaises(RuntimeError):
                validate_config(invalid)


class ResumeBudgetTests(unittest.TestCase):
    def test_explicit_resume_and_exact_row_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"case_id":"one"}\n', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                indexed_rows(path, {"one"}, False)
            self.assertEqual(set(indexed_rows(path, {"one"}, True)), {"one"})
            with self.assertRaises(RuntimeError):
                indexed_rows(path, {"two"}, True)
            path.write_text('{"case_id":"one"}\n{"case_id":"one"}\n', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                indexed_rows(path, {"one"}, True)

    def test_inference_attempt_budget_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "starts.jsonl"
            keys = {str(i) for i in range(MAX_INFERENCES + 1)}
            path.write_text(''.join(json.dumps({"pixel_key": str(i)}) + '\n'
                                   for i in range(MAX_INFERENCES)), encoding="utf-8")
            self.assertEqual(len(inference_starts(path, keys)), MAX_INFERENCES)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"pixel_key": str(MAX_INFERENCES)}) + '\n')
            with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
                inference_starts(path, keys)
            path.write_text('{"pixel_key":"one"}\n{"pixel_key":"one"}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "journal"):
                inference_starts(path, {"one"})

    def test_orphan_receipt_reused_only_on_explicit_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "inference_receipts").mkdir()
            receipt = {"pixel_key": "key", "case_id": "case", "decoded_bits": "0" * 32}
            (output / "inference_receipts/key.json").write_text(json.dumps(receipt), encoding="utf-8")
            case = {"pixel_key": "key", "case_id": "case", "pilot_raw": None}
            with patch("protocol_v2_c1_full_v1.OUTPUT", output), patch("protocol_v2_c1_full_v1.infer_unique") as infer:
                with self.assertRaises(RuntimeError):
                    obtain_raw({"resume": False}, case, {"key"})
                self.assertEqual(obtain_raw({"resume": True}, case, {"key"}), receipt)
                infer.assert_not_called()

    def test_pilot_raw_reuse_never_invokes_detector(self):
        case = {"pixel_key": "key", "case_id": "case", "pilot_raw": {"decoded_bits": "0" * 32}}
        with patch("protocol_v2_c1_full_v1.infer_unique") as infer:
            row = obtain_raw({}, case, set())
        infer.assert_not_called()
        self.assertEqual(row["status"], "pilot-raw-reuse")

    def test_pipeline_failure_prevents_later_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            executable = output / "python"
            executable.write_text("fixture", encoding="utf-8")
            args = SimpleNamespace(wam_python=executable, resume=False, device="cuda:0")
            with (
                patch("protocol_v2_c1_full_v1.OUTPUT", output),
                patch("protocol_v2_c1_full_v1.WAM_ENV_ROOT", output),
                patch("protocol_v2_c1_full_v1.checked_inputs"),
                patch("protocol_v2_c1_full_v1.subprocess.run",
                      side_effect=subprocess.CalledProcessError(1, ["fixture"])) as run,
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    pipeline(args)
                self.assertEqual(run.call_count, 1)
            self.assertFalse((output / "audit.log").exists())

    def test_partial_cache_is_preserved_and_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = SimpleNamespace(root=Path(directory))
            require_new_cache_slot(cache, "abcdef")
            path = cache.root / "ab/abcdef.npz"
            path.parent.mkdir()
            path.write_bytes(b"preserve partial array")
            with self.assertRaisesRegex(RuntimeError, "partial/unindexed"):
                require_new_cache_slot(cache, "abcdef")
            self.assertEqual(path.read_bytes(), b"preserve partial array")

    def test_only_genuine_finite_float32_raw_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.npz"
            for value in (np.ones((2, 2), dtype=np.float16), np.full((2, 2), np.nan, dtype=np.float32),
                          np.full((2, 2), 1.1, dtype=np.float32)):
                np.savez_compressed(path, mask_logits=value)
                with self.assertRaises(RuntimeError):
                    validate_raw(path)
            np.savez_compressed(path, mask_logits=np.ones((2, 2), dtype=np.float32))
            self.assertEqual(validate_raw(path).dtype, np.float32)


class StatisticsTests(unittest.TestCase):
    def test_cell_wilson_and_residual_silent(self):
        rows = [evidence_row(i, reviewed=(i < 120)) for i in range(200)]
        cell = cell_statistics(rows)
        self.assertEqual((cell["reviews"], cell["silent_other"]), (120, 80))
        self.assertEqual(cell["review_given_escape_wilson95"]["total"], 200)
        rows[-1]["image_index"] = 0
        with self.assertRaises(RuntimeError):
            cell_statistics(rows)

    def test_cluster_interval_does_not_treat_repeated_rows_as_new_images(self):
        rows = [evidence_row(i, reviewed=i < 100) for i in range(200)]
        first = clustered_review_interval(rows, 123, 100)
        second = clustered_review_interval(rows + rows, 123, 100)
        self.assertEqual(first["clusters"], 200)
        self.assertEqual(first["interval95"], second["interval95"])

    def test_paired_difference_preserves_source_matching(self):
        center = [evidence_row(i, escaped=False) for i in range(200)]
        aware = [evidence_row(i, escaped=i < 100) for i in range(200)]
        result = paired_escape_difference(center, list(reversed(aware)), 123, 100)
        self.assertEqual(result["difference"], .5)
        self.assertEqual(result, paired_escape_difference(center, aware, 123, 100))

    def test_transition_direction_is_not_net_change(self):
        rows = [{"attack_id": "a", "legacy_evidence": "TAMPERED_REGION", "evidence": "silent-other"},
                {"attack_id": "b", "legacy_evidence": "SILENT_OTHER", "evidence": "signal-suppressed"}]
        result = directional_transitions(rows)
        self.assertEqual(result["changed_records"], 2)
        self.assertEqual(result["matrix"]["region-removed -> silent-other"], 1)

    def test_mask_aware_is_existing_edge_cohort(self):
        rows, selection = [], []
        for image in range(200):
            for rho in sorted(RHO_LEVELS):
                identity = f"{image}|left|{rho}"
                rows.append({"attack_id": identity, "image_index": image, "rho_requested": rho,
                             "pixel_key": identity, "method": "left"})
                selection.append({"selected_attack_id": identity, "image_index": image,
                                  "rho_requested": rho, "pixel_key": identity})
        self.assertEqual(len(mask_aware_rows(rows, selection)), 1800)
        rows[0]["method"] = "center"
        with self.assertRaises(RuntimeError):
            mask_aware_rows(rows, selection)


if __name__ == "__main__":
    unittest.main()
