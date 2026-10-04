#!/usr/bin/env python3
"""Validate the compact release and recompute headline paper values."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str):
    with (ROOT / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


def close(actual: float, expected: float, tolerance: float = 1e-3) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"expected {expected}, observed {actual}")


def curve_value(rows: list[dict], crop: float, field: str) -> float:
    row = next(item for item in rows if item["crop_frac"] == crop)
    return float(row[field])


def c1_value(rows: list[dict], method: str, rho: float, field: str) -> float:
    row = next(
        item for item in rows if item["method"] == method and item["rho"] == rho
    )
    return float(row[field])


def granularity_value(rows: list[dict], condition: str, rho: float, field: str) -> float:
    row = next(
        item
        for item in rows
        if item["condition"] == condition and float(item["rho"]) == rho
    )
    return float(row[field])


def main() -> int:
    required = [
        "README.md",
        "DESIGN.md",
        "LICENSE",
        "REPRODUCIBILITY.md",
        "experiments/inpainting_common_v1.py",
        "experiments/inpainting_lama_v1.py",
        "experiments/inpainting_resign_v1.py",
        "experiments/inpainting_audit_v1.py",
        "experiments/summarize_inpainting_v1.py",
        "experiments/test_inpainting_common_v1.py",
        "experiments/benign_controls_common_v1.py",
        "experiments/benign_controls_generate_v1.py",
        "experiments/benign_controls_resign_v1.py",
        "experiments/benign_controls_audit_v1.py",
        "experiments/summarize_benign_controls_v1.py",
        "experiments/test_benign_controls_common_v1.py",
        "experiments/benign_centered_crops_v1.py",
        "experiments/lineage_ablation_v1.py",
        "experiments/lineage_common_v1.py",
        "experiments/resign_with_lineage_v1.py",
        "experiments/read_region_lineage_v1.py",
        "experiments/test_lineage_common_v1.py",
        "experiments/protocol_v2_cache_finalize_v1.py",
        "experiments/protocol_v2_cache_inventory_v1.py",
        "experiments/protocol_v2_cache_warm_v1.py",
        "experiments/protocol_v2_manifest_matrix_full_v1.py",
        "experiments/protocol_v2_manifest_matrix_v1.py",
        "experiments/test_protocol_v2_cache_warm.py",
        "experiments/test_protocol_v2_manifest_matrix.py",
        "experiments/test_protocol_v2_manifest_matrix_full.py",
        "experiments/granularity_control_common_v1.py",
        "experiments/granularity_control_embed_v1.py",
        "experiments/granularity_control_detect_v1.py",
        "experiments/granularity_control_summarize_v1.py",
        "experiments/verify_granularity_control_v1.py",
        "experiments/test_granularity_control_v1.py",
        "configs/granularity_control_v2.json",
        "configs/protocol_v2_manifest_matrix.json",
        "manifests/manifest_ai_edited.json",
        "results/summary_500.json",
        "results/regional_v2/summary.json",
        "results/regional_v2_pixart/summary.json",
        "results/c1_v1/full/attack_manifest.jsonl",
        "results/c1_v1/full/audit_rows.jsonl",
        "results/c1_v1/full/summary.json",
        "results/granularity_control_v2/full_v1/config_effective.json",
        "results/granularity_control_v2/full_v1/embed_rows.jsonl",
        "results/granularity_control_v2/full_v1/detection_subject_region.jsonl",
        "results/granularity_control_v2/full_v1/detection_random_compact_equal_area.jsonl",
        "results/granularity_control_v2/full_v1/detection_fullframe_native.jsonl",
        "results/granularity_control_v2/full_v1/detection_fullframe_energy_matched.jsonl",
        "results/granularity_control_v2/full_v1/detection_clean_reference.jsonl",
        "results/granularity_control_v2/full_v1/summary.json",
        "results/granularity_control_v2/full_v1/summary.csv",
        "results/granularity_control_v2/full_v1/verification.json",
        "results/granularity_control_v2/full_v1/fig_granularity_control.pdf",
        "results/granularity_control_v2/full_v1/fig_granularity_control.png",
        "results/threshold_sensitivity_v1/threshold_sensitivity.json",
        "results/inpainting_e2e_v1/audit.csv",
        "results/inpainting_e2e_v1/summary.json",
        "results/benign_controls_v1/audit.csv",
        "results/benign_controls_v1/audit.json",
        "results/benign_controls_v1/summary.json",
        "results/benign_controls_v1/contact_sheet_sample.png",
        "results/benign_centered_crops_v1/audit_honest.csv",
        "results/benign_centered_crops_v1/audit_honest.json",
        "results/benign_centered_crops_v1/summary.json",
        "results/benign_centered_crops_v1/environment.json",
        "results/lineage_ablation_v1/rows.jsonl",
        "results/lineage_ablation_v1/summary.json",
        "results/lineage_ablation_v1/summary.md",
        "results/protocol_v2/manifest_matrix_pilot_v2/rows.jsonl",
        "results/protocol_v2/manifest_matrix_pilot_v2/summary.json",
        "results/protocol_v2/manifest_matrix_full_v2/rows.jsonl",
        "results/protocol_v2/manifest_matrix_full_v2/summary.json",
        "results/protocol_v2/manifest_matrix_full_v2/selected_cases.json",
        "results/protocol_v2/manifest_matrix_full_v2/parent_index.json",
        "results/protocol_v2/full_pixel_cache_v1/warm_full_v1/rows.jsonl",
        "results/protocol_v2/full_pixel_cache_v1/warm_full_v1/summary.json",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    if missing:
        raise AssertionError(f"missing release files: {missing}")

    baseline = load("results/summary_500.json")
    close(float(baseline["wam_audit"]["metrics"]["TPR"]), 1.0)
    close(float(baseline["wam_audit"]["metrics"]["FPR"]), 0.0)

    sdxl = load("results/regional_v2/summary.json")
    pixart = load("results/regional_v2_pixart/summary.json")
    close(curve_value(sdxl, 0.0, "detection_rate_pct"), 99.0)
    close(curve_value(sdxl, 0.2, "detection_rate_pct"), 62.5)
    close(curve_value(sdxl, 0.4, "detection_rate_pct"), 0.5)
    close(curve_value(pixart, 0.0, "detection_rate_pct"), 98.0)
    close(curve_value(pixart, 0.2, "detection_rate_pct"), 51.0)
    close(curve_value(pixart, 0.4, "detection_rate_pct"), 0.5)

    c1 = load("results/c1_v1/full/summary.json")["summary"]
    close(c1_value(c1, "center", 0.4, "escape_rate_pct"), 17.0)
    close(c1_value(c1, "mask_aware", 0.4, "escape_rate_pct"), 55.0)

    granularity = load("results/granularity_control_v2/full_v1/summary.json")
    granularity_rows = granularity["summary"]
    close(granularity_value(granularity_rows, "subject_region", 0.8, "escape_rate_pct"), 55.0)
    close(granularity_value(granularity_rows, "random_compact_equal_area", 0.8, "escape_rate_pct"), 18.5)
    close(granularity_value(granularity_rows, "fullframe_native", 0.8, "escape_rate_pct"), 0.5)
    close(granularity_value(granularity_rows, "fullframe_energy_matched", 0.8, "escape_rate_pct"), 27.0)
    close(float(granularity["quality"]["subject_region"]["mean_psnr_db"]), 42.4615, 1e-4)
    close(float(granularity["quality"]["fullframe_energy_matched"]["mean_psnr_db"]), 42.4615, 1e-4)
    if int(granularity["embedding_validation"]["subject_rgb_hash_matches"]) != 200:
        raise AssertionError("granularity control must match all 200 subject hashes")
    if int(granularity["embedding_validation"]["random_area_matches"]) != 200:
        raise AssertionError("granularity control must match all 200 random-region areas")
    if int(granularity["embedding_validation"]["random_wrap_count"]) != 0:
        raise AssertionError("granularity control random regions must not wrap")

    granularity_verification = load(
        "results/granularity_control_v2/full_v1/verification.json"
    )
    failed_checks = [
        key for key, value in granularity_verification["checks"].items() if value is not True
    ]
    if failed_checks:
        raise AssertionError(f"granularity verification failed: {failed_checks}")

    expected_jsonl_rows = {
        "results/c1_v1/pilot/attack_manifest.jsonl": 360,
        "results/c1_v1/pilot/audit_rows.jsonl": 360,
        "results/c1_v1/full/attack_manifest.jsonl": 10_800,
        "results/c1_v1/full/audit_rows.jsonl": 10_800,
        "results/granularity_control_v2/full_v1/embed_rows.jsonl": 200,
        "results/granularity_control_v2/full_v1/detection_subject_region.jsonl": 1_800,
        "results/granularity_control_v2/full_v1/detection_random_compact_equal_area.jsonl": 1_800,
        "results/granularity_control_v2/full_v1/detection_fullframe_native.jsonl": 1_800,
        "results/granularity_control_v2/full_v1/detection_fullframe_energy_matched.jsonl": 1_800,
        "results/granularity_control_v2/full_v1/detection_clean_reference.jsonl": 200,
        "results/protocol_v2/manifest_matrix_pilot_v2/rows.jsonl": 100,
        "results/protocol_v2/manifest_matrix_full_v2/rows.jsonl": 8_000,
        "results/protocol_v2/full_pixel_cache_v1/warm_full_v1/rows.jsonl": 1_600,
    }
    for relative, expected in expected_jsonl_rows.items():
        with (ROOT / relative).open(encoding="utf-8") as handle:
            observed = sum(1 for line in handle if line.strip())
        if observed != expected:
            raise AssertionError(f"{relative}: expected {expected} rows, observed {observed}")

    sensitivity = load(
        "results/threshold_sensitivity_v1/threshold_sensitivity.json"
    )["reference_point"]
    reference = {row["dataset"]: row for row in sensitivity}
    close(reference["SDXL centered"]["escape_flag_rate_pct"], 91.223)
    close(reference["PixArt-alpha centered"]["escape_flag_rate_pct"], 92.553)
    close(reference["SDXL mask-aware"]["escape_flag_rate_pct"], 93.435)

    inpainting = load("results/inpainting_e2e_v1/summary.json")
    if int(inpainting["n"]) != 200:
        raise AssertionError("inpainting summary must contain 200 assets")
    for invariant, observed in inpainting["invariants"].items():
        if int(observed) != 200:
            raise AssertionError(
                f"inpainting invariant {invariant}: expected 200, observed {observed}"
            )
    if int(inpainting["baseline_silent_escape"]["count"]) != 199:
        raise AssertionError("expected 199 baseline inpainting escapes")
    if int(inpainting["region_aware_recovery_among_escapes"]["count"]) != 199:
        raise AssertionError("expected all 199 inpainting escapes to be recovered")
    close(float(inpainting["median_bit_accuracy_before"]), 1.0)
    close(float(inpainting["median_bit_accuracy_after"]), 0.5)
    close(float(inpainting["median_logit_before"]), 0.96075)
    close(float(inpainting["median_logit_after"]), 0.0006)

    with (ROOT / "results/inpainting_e2e_v1/audit.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        inpainting_rows = list(csv.DictReader(handle))
    if len(inpainting_rows) != 200:
        raise AssertionError(
            f"inpainting audit: expected 200 rows, observed {len(inpainting_rows)}"
        )
    escaped = [row for row in inpainting_rows if row["baseline_audit"] == "SILENT_ESCAPE"]
    recovered = [
        row for row in escaped if row["region_aware_audit"] == "WATERMARK_SUPPRESSED"
    ]
    if len(escaped) != 199 or len(recovered) != 199:
        raise AssertionError(
            f"inpainting audit: expected 199/199 recovered escapes, observed "
            f"{len(recovered)}/{len(escaped)}"
        )

    benign = load("results/benign_controls_v1/summary.json")
    if benign.get("version") != "benign_controls_v1":
        raise AssertionError("unexpected benign-control result version")
    benign_by_condition = {
        row["condition"]: row for row in benign["conditions"]
    }
    expected_benign_flags = {
        "identity_resave": 1,
        "jpeg_q90": 1,
        "resize_90": 1,
        "safe_crop_1pct": 2,
    }
    for condition, expected_flags in expected_benign_flags.items():
        row = benign_by_condition[condition]
        if int(row["n"]) != 200:
            raise AssertionError(f"{condition}: expected 200 benign assets")
        if int(row["manifest_valid"]) != 200:
            raise AssertionError(f"{condition}: all manifests must validate")
        if int(row["region_assertion_present"]) != 200:
            raise AssertionError(f"{condition}: all region assertions must be present")
        if int(row["payload_hash_ok"]) != 200:
            raise AssertionError(f"{condition}: all payload hashes must match")
        observed_flags = int(row["benign_region_false_flag"]["count"])
        if observed_flags != expected_flags:
            raise AssertionError(
                f"{condition}: expected {expected_flags} strict false flags, "
                f"observed {observed_flags}"
            )
    clean = benign_by_condition["clean_negative"]
    if int(clean["n"]) != 200:
        raise AssertionError("clean-negative summary must contain 200 assets")
    if int(clean["clean_detector_false_positive"]["count"]) != 1:
        raise AssertionError("expected one clean detector false positive")
    if clean["verdict_counts"] != {"NO_REGION_CLAIM": 200}:
        raise AssertionError("all clean negatives must end as NO_REGION_CLAIM")

    with (ROOT / "results/benign_controls_v1/audit.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        benign_rows = list(csv.DictReader(handle))
    if len(benign_rows) != 1_000:
        raise AssertionError(
            f"benign audit: expected 1,000 rows, observed {len(benign_rows)}"
        )

    centered = load("results/benign_centered_crops_v1/summary.json")
    if centered.get("version") != "benign_centered_crops_v1":
        raise AssertionError("unexpected centered-crop control result version")
    centered_by_crop = {
        float(row["crop_frac_per_side"]): row for row in centered["conditions"]
    }
    expected_centered = {
        0.1: (15, 14),
        0.2: (68, 67),
    }
    for crop, (strict_flags, qualified_flags) in expected_centered.items():
        row = centered_by_crop[crop]
        if int(row["n"]) != 200:
            raise AssertionError(f"centered {crop}: expected 200 assets")
        for invariant in ("manifest_valid", "region_assertion_present", "payload_hash_ok"):
            if int(row[invariant]) != 200:
                raise AssertionError(f"centered {crop}: invariant {invariant} failed")
        if int(row["strict_false_flag"]["count"]) != strict_flags:
            raise AssertionError(f"centered {crop}: unexpected strict flag count")
        if int(row["baseline_qualified_false_flag"]["count"]) != qualified_flags:
            raise AssertionError(f"centered {crop}: unexpected qualified flag count")

    with (ROOT / "results/benign_centered_crops_v1/audit_honest.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        centered_rows = list(csv.DictReader(handle))
    if len(centered_rows) != 400:
        raise AssertionError(
            f"centered-crop audit: expected 400 rows, observed {len(centered_rows)}"
        )

    environment = load("results/benign_centered_crops_v1/environment.json")
    if environment["software"]["pytorch"] != "2.6.0+cu124":
        raise AssertionError("unexpected reference PyTorch version")
    if int(environment["host"]["gpu_host_total"]) != 8:
        raise AssertionError("unexpected shared-host GPU count")
    if int(environment["host"]["gpu_allocated"]) != 3:
        raise AssertionError("unexpected instance GPU allocation")
    close(float(environment["execution"]["prepare_and_honest_resign_seconds"]), 452.532)
    close(float(environment["execution"]["region_aware_audit_seconds"]), 108.497)

    lineage = load("results/lineage_ablation_v1/summary.json")["conditions"]
    expected_lineage = {
        "propagated": (200, 200, "CURRENT_ASSERTION"),
        "ancestor-only": (200, 200, "ANCESTOR_ASSERTION"),
        "cut-chain": (200, 0, "PROVENANCE_DISCONTINUITY"),
    }
    for condition, (valid, parent_validated, status) in expected_lineage.items():
        row = lineage[condition]
        if int(row["n"]) != 200 or int(row["manifest_valid"]) != valid:
            raise AssertionError(f"{condition}: expected 200 valid manifests")
        if int(row["pixel_identical"]) != 200:
            raise AssertionError(f"{condition}: decoded pixels must remain identical")
        if int(row["parent_manifest_validated"]) != parent_validated:
            raise AssertionError(f"{condition}: unexpected parent-validation count")
        if row["lineage_status"] != {status: 200}:
            raise AssertionError(f"{condition}: unexpected lineage status")
    if lineage["propagated"]["audit_v3"] != lineage["ancestor-only"]["audit_v3"]:
        raise AssertionError("propagated and ancestor-only verdicts must match")

    lineage_rows_path = ROOT / "results/lineage_ablation_v1/rows.jsonl"
    with lineage_rows_path.open(encoding="utf-8") as handle:
        lineage_rows = [json.loads(line) for line in handle if line.strip()]
    if len(lineage_rows) != 600:
        raise AssertionError(
            f"lineage ablation: expected 600 rows, observed {len(lineage_rows)}"
        )

    protocol_pilot = load(
        "results/protocol_v2/manifest_matrix_pilot_v2/summary.json"
    )
    if int(protocol_pilot["row_count"]) != 100:
        raise AssertionError("protocol-v2 semantics pilot must contain 100 rows")
    for field in (
        "manifest_valid", "pixel_cache_hits", "expected_state_matches",
        "expected_policy_matches",
    ):
        if int(protocol_pilot[field]) != 100:
            raise AssertionError(f"protocol-v2 pilot acceptance failed: {field}")
    if int(protocol_pilot["evidence_invariant_cases"]) != 20:
        raise AssertionError("protocol-v2 pilot evidence must be invariant in 20 cases")

    protocol_full = load(
        "results/protocol_v2/manifest_matrix_full_v2/summary.json"
    )
    if int(protocol_full["case_count"]) != 1_600 or int(protocol_full["row_count"]) != 8_000:
        raise AssertionError("protocol-v2 full matrix must contain 1,600 cases and 8,000 rows")
    for field in (
        "manifest_valid", "pixel_cache_hits", "expected_state_matches",
        "expected_policy_matches",
    ):
        if int(protocol_full[field]) != 8_000:
            raise AssertionError(f"protocol-v2 full acceptance failed: {field}")
    if int(protocol_full["gpu_inference_calls"]) != 0:
        raise AssertionError("protocol-v2 manifest matrix must not invoke the detector")
    if int(protocol_full["evidence_invariant_cases"]) != 1_600:
        raise AssertionError("protocol-v2 evidence must be invariant in all 1,600 cases")
    for condition, counts in protocol_full["evidence_states_by_condition"].items():
        if condition == "cut-chain-fresh-root":
            if counts != {"no-reference": 1_600}:
                raise AssertionError("all cut-chain cases must lose the signed reference")
        elif counts != {
            "message-detected": 1_167,
            "region-removed": 159,
            "signal-suppressed": 230,
            "silent-other": 44,
        }:
            raise AssertionError(f"unexpected protocol-v2 evidence split: {condition}")
    if len(protocol_full["datasets"]) != 8:
        raise AssertionError("protocol-v2 full matrix must contain eight datasets")
    for dataset, counts in protocol_full["datasets"].items():
        if counts != {"cases": 200, "rows": 1_000}:
            raise AssertionError(f"unexpected protocol-v2 dataset coverage: {dataset}")
    if not all(protocol_full["acceptance"].values()):
        raise AssertionError("protocol-v2 full matrix acceptance flags must all pass")

    cache_summary = load(
        "results/protocol_v2/full_pixel_cache_v1/warm_full_v1/summary.json"
    )
    if int(cache_summary["expected_records"]) != 1_600:
        raise AssertionError("protocol-v2 exact cache must expect 1,600 records")
    if int(cache_summary["verified_records"]) != 1_600:
        raise AssertionError("protocol-v2 exact cache must verify 1,600 records")
    if not all(cache_summary["acceptance"].values()):
        raise AssertionError("protocol-v2 exact-cache acceptance flags must all pass")

    blocked_suffixes = {".pth", ".pt", ".ckpt", ".safetensors"}
    blocked = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in blocked_suffixes
    ]
    if blocked:
        raise AssertionError(f"model weights must not be committed: {blocked}")

    oversized = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path.stat().st_size > 50 * 1024 * 1024
    ]
    if oversized:
        raise AssertionError(f"files larger than 50 MiB found: {oversized}")

    print("Release verification passed.")
    print("  SDXL regional detection: 99.0% -> 62.5% -> 0.5%")
    print("  PixArt-alpha detection:   98.0% -> 51.0% -> 0.5%")
    print("  rho=0.4 escape rate:      center 17.0%, mask-aware 55.0%")
    print("  rho=0.8 support control:  subject 55.0%, compact 18.5%, full 0.5%, matched 27.0%")
    print("  defense flag rates:       91.223%, 92.553%, 93.435%")
    print("  exact-mask inpainting:    199/200 escapes, 199/199 recovered")
    print("  benign strict flags:      1/200, 1/200, 1/200, 2/200")
    print("  clean detector/verdict:   1/200 FP, 200/200 NO_REGION_CLAIM")
    print("  honest centered crops:    15/200 (10%), 68/200 (20%) flags")
    print("  lineage continuity:       200/200 ancestor recovery, 200/200 cut-chain flags")
    print("  protocol-v2 semantics:    8,000/8,000 rows accepted, 1,600/1,600 invariant")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, StopIteration, json.JSONDecodeError) as error:
        print(f"Release verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
