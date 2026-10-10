"""CPU fixtures for selection, exact bits, geometry, trust and precision scopes."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from protocol_v2_c1_pilot_v1 import (
    METHODS, RHO_LEVELS, cache_hashes, check_decoded, checked_cases, choose_boundaries, validate_coverage, warm_case,
)
from protocol_v2_c1_pilot_audit_v1 import state_checks
from protocol_v2_c1_pilot_verify_v1 import check_precision_acceptance, precision_diagnostics
from protocol_v2_cache import PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_cache_warm_v1 import PREPROCESSING_VERSION
from protocol_v2_common import classify_evidence
from protocol_v2_main_curve_full_v1 import trust_condition
from protocol_v2_main_curve_full_v2 import signed_transform_matches
from protocol_v2_pilot_v1 import crop_transform
from PIL import Image


def fixture_rows():
    metadata, rows = {}, []
    for i in (0, 1, 2, 3, 46, 117):
        image = f"{i:04d}.png"
        metadata[image] = {"mask_coverage": (i + 1) / 200}
        for rho in sorted(RHO_LEVELS):
            for method in sorted(METHODS):
                matches = 17 if i == 117 and rho == 0 else 23 if i == 46 and rho == 0 else 24 + i % 3
                rows.append({"attack_id": f"{i:04d}|{method}|{rho:.3f}", "image_index": i,
                             "input_image": image, "rho_requested": rho, "method": method,
                             "legacy_match_count": matches, "cache_hit": method == "center" or rho == 0,
                             "pixel_key": f"{i}|zero" if rho == 0 else f"{i}|{method}|{rho}",
                             "region_retention_unrounded": max(0, .85 - rho),
                             "legacy_logit_in_region": .5 + (i - 2) / 1000})
    rows[-1]["region_retention_unrounded"] = 0
    return rows, metadata


class SelectionTests(unittest.TestCase):
    def test_deterministic_complete_unique_coverage(self):
        rows, metadata = fixture_rows()
        chosen = choose_boundaries(rows, metadata)
        self.assertEqual(chosen, choose_boundaries(list(reversed(rows)), metadata))
        self.assertEqual(len(chosen), 20)
        validate_coverage(chosen)
        reasons = [reason for r in chosen for reason in r["selection_reasons"]]
        for expected in ("low source-mask coverage", "high source-mask coverage", "empty retained region"):
            self.assertIn(expected, reasons)

    def test_no_missing_cache_candidate_rejected(self):
        rows, metadata = fixture_rows()
        for row in rows:
            row["cache_hit"] = True
        with self.assertRaises(RuntimeError):
            choose_boundaries(rows, metadata)

    def test_duplicate_pixel_rejected(self):
        rows, metadata = fixture_rows()
        chosen = choose_boundaries(rows, metadata)
        chosen[-1]["pixel_key"] = chosen[0]["pixel_key"]
        with self.assertRaises(RuntimeError):
            validate_coverage(chosen)

    def test_mutated_selection_hash_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "selection_summary.json").write_text(
                '{"config_sha256":"good", "selected_cases_sha256":"old"}', encoding="utf-8")
            (root / "selected_cases.json").write_text("[]", encoding="utf-8")
            args = SimpleNamespace(config=root / "config.json", output_dir=root)
            with patch("protocol_v2_c1_pilot_v1.checked_config"), patch(
                    "protocol_v2_c1_pilot_v1.sha256_file", side_effect=["good", "mutated"]):
                with self.assertRaisesRegex(RuntimeError, "selected cases changed"):
                    checked_cases(args)


class GeometryBitTests(unittest.TestCase):
    def test_rho_not_per_side_c_and_asymmetric_roundtrip(self):
        transform = crop_transform([1024, 1024], [205, 0, 1024, 1024], [1024, 1024])
        observed = {"status": "valid", "transform": {"source_size_px": [1024, 1024],
                    "crop_box_px": [205, 0, 1024, 1024], "output_size_px": [1024, 1024],
                    "resize_filter": "LANCZOS"}}
        self.assertTrue(signed_transform_matches({"crop_transform": transform}, {"geometry": observed}))
        self.assertNotEqual(1 - 819 / 1024, 1 - (1 - 2 * .2) ** 2)
        self.assertFalse(signed_transform_matches({"crop_transform": None}, {"geometry": observed}))

    def test_integer_boundary_and_exact_bits(self):
        case = {"region_assertion": {"reference_bits": "0" * 32}, "legacy_match_count": 24, "attack_id": "a"}
        bits = "0" * 24 + "1" * 8
        self.assertEqual(check_decoded(case, bits, {"decoded_bits": bits}), 24)
        with self.assertRaisesRegex(RuntimeError, "cached decoded bits changed"):
            check_decoded(case, bits, {"decoded_bits": "1" * 8 + "0" * 24})
        for count, evidence in ((23, "signal-suppressed"), (24, "message-detected")):
            self.assertEqual(classify_evidence(reference_state="signed-ancestor", match_count=count,
                             required_matches=24, region_retention=1, logit_in_region=.4), evidence)

    def test_empty_region_priority_and_untrusted_no_evidence(self):
        evidence = classify_evidence(reference_state="signed-ancestor", match_count=17, required_matches=24,
                                     region_retention=0, logit_in_region=0)
        self.assertEqual(evidence, "region-removed")
        raw = {"lineage": "ancestor", "evidence": evidence, "semantics": "no-ai-disclosure-in-chain",
               "policy_verdict": "review-evidence-removal"}
        untrusted = trust_condition(raw, gate="accepted", cropped_action_count=1, editor_trust="untrusted-editor")
        self.assertFalse(untrusted["pixel_evidence_consumed"])
        self.assertEqual(untrusted["policy_verdict"], "review-untrusted-geometry")


class PrecisionTests(unittest.TestCase):
    def test_rerun_and_quantization_are_separate(self):
        original = np.full((2, 2), .6001, dtype=np.float32)
        old = np.full((2, 2), .59, dtype=np.float16)
        with patch("protocol_v2_c1_pilot_verify_v1.logit_mean", side_effect=lambda x, _: float(x.mean())):
            metrics = precision_diagnostics(original, old, np.ones((2, 2)))
        self.assertGreater(metrics["rerun_vs_cache_max_abs"], metrics["quantization_max_abs"])

    def test_cast_float16_not_accepted_as_raw(self):
        with self.assertRaisesRegex(RuntimeError, "genuine raw"):
            precision_diagnostics(np.ones((2, 2), dtype=np.float16), np.ones((2, 2)), np.ones((2, 2)))

    def test_classification_change_stops(self):
        outputs = {"raw": "suppressed", "rerun_quantized": "silent", "cached": "silent"}
        with self.assertRaisesRegex(RuntimeError, "quantization changed"):
            check_precision_acceptance(outputs, True, {"rerun_vs_cache_max_abs": 0})
        outputs = {"raw": "silent", "rerun_quantized": "silent", "cached": "suppressed"}
        with self.assertRaisesRegex(RuntimeError, "preserved cache changed"):
            check_precision_acceptance(outputs, True, {"rerun_vs_cache_max_abs": .01})

    def test_new_cache_requires_exact_quantization(self):
        outputs = dict.fromkeys(("raw", "rerun_quantized", "cached"), "same")
        with self.assertRaisesRegex(RuntimeError, "exact raw-to-float16"):
            check_precision_acceptance(outputs, False, {"rerun_vs_cache_max_abs": .001})


class WarmingTests(unittest.TestCase):
    def test_hit_is_inferred_but_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            Image.new("RGB", (8, 8)).save(source)
            key, image = decoded_rgb_key(source)
            cache = PixelCache(root / "cache")
            bits = "0" * 32
            cache.put(key=key, decoded_bits=bits, mask_logits=np.full((2, 2), .6), image=image,
                      model_sha256="model", checkpoint_sha256="checkpoint", preprocessing_version=PREPROCESSING_VERSION)
            hashes = cache_hashes(cache, key)
            (root / "raw_logits").mkdir()
            case = {"child_source_relpath": "source.png", "source_file_sha256": sha256_file(source),
                    "case_id": "case", "attack_id": "attack", "pixel_key": key, "image_size": image,
                    "cache_hit": True, "legacy_match_count": 32, "preserved_cache_hashes": hashes,
                    "region_assertion": {"reference_bits": bits}}
            detector = SimpleNamespace(run=lambda _: (bits, np.full((2, 2), .6001, dtype=np.float32)))
            context = {"cache": cache, "output": root, "inference_calls": 0, "detector": detector,
                       "model_hash": "model", "checkpoint_hash": "checkpoint"}
            with patch("protocol_v2_c1_pilot_v1.ROOT", root):
                row = warm_case(context, case)
            self.assertEqual(context["inference_calls"], 1)
            self.assertEqual(row["comparison_scope"], "rerun-and-quantization")
            self.assertEqual(cache_hashes(cache, key), hashes)
            self.assertTrue((root / "raw_logits/case.npz").is_file())


if __name__ == "__main__":
    unittest.main()
