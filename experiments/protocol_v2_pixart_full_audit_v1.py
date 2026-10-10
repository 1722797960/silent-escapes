"""Resumable, detector-free signing and verification of the PixArt full curve."""
from __future__ import annotations

import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np

from protocol_v2_cache import decoded_rgb_key, sha256_file
from protocol_v2_common import audit_protocol_v2
from protocol_v2_geometry_lie_v1 import geometry_gate
from protocol_v2_main_curve_full_v1 import append_jsonl, package_version, parent_path, prepare_parents
from protocol_v2_main_curve_full_v1 import sign_child, trust_condition
from protocol_v2_pilot_v1 import atomic_json, load_json, load_jsonl, logit_mean, mapped_mask_and_retention
from protocol_v2_pilot_v1 import sanitize_pixels
from protocol_v2_pixart_audit_v1 import state_checks
from protocol_v2_pixart_full_v1 import OUTPUT, ROOT, checked_cases, indexed_rows, model_context
from protocol_v2_pixart_pilot_v1 import validate_cached
from read_actions_v2 import inspect_asset


def ready_cases():
    cases = checked_cases()
    warm_summary = load_json(OUTPUT / "warm_summary.json")
    if not warm_summary["acceptance_passed"] or warm_summary["cases"] != 1800:
        raise RuntimeError("complete warming acceptance required before signing")
    if sha256_file(OUTPUT / "warm_rows.jsonl") != warm_summary["warm_rows_sha256"]:
        raise RuntimeError("warm provenance changed")
    for case in cases:
        case["parent_source"] = str(ROOT / case["parent_source_relpath"])
    return cases


def ensure_signed(case, context, resume):
    source = ROOT / case["child_source_relpath"]
    if sha256_file(source) != case["source_file_sha256"]:
        raise RuntimeError("frozen child input changed")
    signed = OUTPUT / "signed" / f"{case['case_id']}.png"
    if signed.exists():
        if not resume:
            raise FileExistsError("signed orphan requires --resume and fresh validation")
        return signed
    pixels = OUTPUT / "pixels" / f"{case['case_id']}.png"
    sanitize_pixels(source, pixels)
    if decoded_rgb_key(pixels)[0] != case["pixel_key"]:
        raise RuntimeError("sanitizing changed pixels")
    sign_child(case, pixels=pixels, signed=signed, parent=parent_path(OUTPUT, case["image_index"]),
               signer=context["signer"])
    return signed


def fresh_audit(case, context):
    signed = OUTPUT / "signed" / f"{case['case_id']}.png"
    inspection = inspect_asset(signed)
    gate = geometry_gate(inspection)
    if not inspection["manifest_valid"] or gate != "accepted":
        raise RuntimeError("signed asset or geometry invalid")
    entry = context["cache"].get(case["pixel_key"])
    if entry is None:
        raise RuntimeError("cache miss; GPU forbidden in manifest evaluation")
    matches = validate_cached(context, case, entry)
    mapped, retention = mapped_mask_and_retention(inspection["region"], inspection["geometry"])
    logit = logit_mean(entry["mask_logits"], mapped)
    raw = audit_protocol_v2(
        lineage=inspection["lineage_state"], reference_state=inspection["reference_source"],
        semantics=inspection["chain_semantics"], reference_bits=inspection["region_assertion"]["reference_bits"],
        decoded_bits=entry["decoded_bits"], required_matches=24,
        region_retention=retention, logit_in_region=logit)
    metrics = dict(retention=retention, logit=logit, gate=gate, pixel_key=decoded_rgb_key(signed)[0])
    checks = state_checks(case, inspection, raw, metrics)
    if matches != raw["match_count"] or not all(checks.values()):
        raise RuntimeError(f"signed-chain acceptance failed: {case['case_id']}: {checks}")
    conditions = {trust: trust_condition(raw, gate=gate, cropped_action_count=inspection["cropped_action_count"],
                                        editor_trust=trust) for trust in ("trusted-editor", "untrusted-editor")}
    return {"case_id": case["case_id"], "crop_frac": case["crop_frac"], "image_index": case["image_index"],
            "manifest_valid": True, "manifest_sha256": sha256_file(signed), "pixel_key": case["pixel_key"],
            "pixel_cache_hit": True, "gpu_inference_calls": 0, "match_count": raw["match_count"],
            "legacy_match_count": case["legacy_match_count"], "region_retention": retention,
            "logit_in_region": logit, "evidence": raw["evidence"], "conditions": conditions,
            "state_checks": checks, **{k: raw[k] for k in ("lineage", "reference", "semantics")}}


def validate_audit_rows(rows):
    if len(rows) != 1800 or len({r["case_id"] for r in rows}) != 1800:
        raise RuntimeError("1800 unique audit rows required")
    if not all(all(r["state_checks"].values()) and r["manifest_valid"] for r in rows):
        raise RuntimeError("audit row acceptance failed")


def policy_counts(rows, trust):
    return dict(Counter(r["conditions"][trust]["policy_verdict"] for r in rows))


def summary_counts(rows):
    validate_audit_rows(rows)
    escapes = [r for r in rows if r["match_count"] < 24]
    if len(escapes) != 940:
        raise RuntimeError("new integer escape count differs from the frozen cohort")
    reviews = sum(r["evidence"] in {"region-removed", "signal-suppressed"} for r in escapes)
    if reviews + sum(r["evidence"] == "silent-other" for r in escapes) != len(escapes):
        raise RuntimeError("escape partition incomplete")
    return {"cases": 1800, "valid_manifests": 1800, "escapes": len(escapes),
            "trusted_editor_reviews": reviews, "trusted_editor_silent": 940-reviews,
            "trusted_editor_review_rate": reviews/940,
            "evidence": dict(Counter(r["evidence"] for r in rows)),
            "trusted_policy": policy_counts(rows, "trusted-editor"),
            "untrusted_policy": policy_counts(rows, "untrusted-editor"),
            "gpu_inference_calls": 0, "integer_count_matches": 1800, "acceptance_passed": True}


def summarize(rows):
    summary = summary_counts(rows)
    per_crop = {}
    for crop in sorted({r["crop_frac"] for r in rows}):
        subset = [r for r in rows if r["crop_frac"] == crop]
        escaped = [r for r in subset if r["match_count"] < 24]
        if len(subset) != 200:
            raise RuntimeError("per-crop cohort incomplete")
        per_crop[f"{crop:.2f}"] = {"cases": 200, "escapes": len(escaped),
            "reviews": sum(r["evidence"] in {"region-removed", "signal-suppressed"} for r in escaped),
            "evidence": dict(Counter(r["evidence"] for r in subset)),
            "untrusted_policy": dict(Counter(r["conditions"]["untrusted-editor"]["policy_verdict"] for r in subset))}
    summary.update(schema="protocol-v2-pixart-full-audit-1", per_crop=per_crop,
                   crop_parameter="fraction removed from each side",
                   audit_rows_sha256=sha256_file(OUTPUT / "audit_rows.jsonl"),
                   selected_cases_sha256=sha256_file(OUTPUT / "selected_cases.json"),
                   script_sha256=sha256_file(Path(__file__)))
    atomic_json(OUTPUT / "audit_summary.json", summary)


def checked_parents(cases, context, resume):
    for case in cases:
        if sha256_file(case["parent_source"]) != case["parent_file_sha256"]:
            raise RuntimeError("frozen parent input changed")
    prepare_parents(cases, output=OUTPUT, signer=context["signer"], resume=resume)
    for case in {c["image_index"]: c for c in cases}.values():
        if decoded_rgb_key(parent_path(OUTPUT, case["image_index"]))[0] != case["parent_pixel_key"]:
            raise RuntimeError("parent signing changed pixels")


def run_audit(args):
    cases = ready_cases()
    context = model_context()
    path = OUTPUT / "audit_rows.jsonl"
    rows = indexed_rows(path, cases, args.resume)
    from protocol_v2_sign_pilot import create_signer
    context["signer"] = create_signer(ROOT / "certs/ec_chain.pem", ROOT / "certs/ec_key.pem")
    started = time.perf_counter()
    checked_parents(cases, context, args.resume)
    for case in cases:
        if case["case_id"] in rows:
            saved = rows[case["case_id"]]
            if sha256_file(OUTPUT / "signed" / f"{case['case_id']}.png") != saved["manifest_sha256"]:
                raise RuntimeError("completed signed asset changed during resume")
            continue
        ensure_signed(case, context, args.resume)
        row = fresh_audit(case, context)
        append_jsonl(path, row)
        rows[case["case_id"]] = row
        if len(rows) % 25 == 0 or len(rows) == 1800:
            atomic_json(OUTPUT / "audit_progress.json", {"completed": len(rows), "total": 1800,
                        "session_seconds": time.perf_counter()-started})
            print(f"PixArt full signed/audited {len(rows)}/1800", flush=True)
    summarize(list(rows.values()))
    atomic_json(OUTPUT / "audit_environment.json", {
        "wall_seconds_current_session": time.perf_counter()-started,
        **{name: package_version(name) for name in ("torch", "numpy", "Pillow", "c2pa-python")},
        "gpu_inference_calls": 0})


def verify_raw(case, saved, warm, context):
    if fresh_audit(case, context) != saved:
        raise RuntimeError("independent signed-chain recomputation differs")
    entry = context["cache"].get(case["pixel_key"])
    if entry["decoded_bits"] != warm["decoded_bits"]:
        raise RuntimeError("cached bits differ from warm provenance")
    path = ROOT / warm["raw_logits_relpath"]
    if sha256_file(path) != warm["raw_logits_sha256"]:
        raise RuntimeError("raw float32 provenance changed")
    inspection = inspect_asset(OUTPUT / "signed" / f"{case['case_id']}.png")
    mapped, retention = mapped_mask_and_retention(inspection["region"], inspection["geometry"])
    with np.load(path, allow_pickle=False) as arrays:
        logits = arrays["mask_logits"]
        if logits.dtype != np.float32 or not np.isfinite(logits).all() or logits.shape != entry["mask_logits"].shape:
            raise RuntimeError("invalid raw float32 logits")
        error = float(np.max(np.abs(logits-entry["mask_logits"].astype(np.float32))))
        mean = logit_mean(logits, mapped)
    raw = audit_protocol_v2(
        lineage=saved["lineage"], reference_state=saved["reference"], semantics=saved["semantics"],
        reference_bits=inspection["region_assertion"]["reference_bits"], decoded_bits=entry["decoded_bits"],
        required_matches=24, region_retention=retention, logit_in_region=mean)
    if raw["evidence"] != saved["evidence"]:
        raise RuntimeError(f"quantization changed evidence: {case['case_id']}")
    return {"case_id": case["case_id"], "max_logit_quantization_error": error,
            "region_mean_difference": abs(mean-saved["logit_in_region"]), "passed": True}


def verify(args):
    cases = ready_cases()
    context = model_context()
    audit = indexed_rows(OUTPUT / "audit_rows.jsonl", cases, True)
    warm = indexed_rows(OUTPUT / "warm_rows.jsonl", cases, True)
    summary_counts(list(audit.values()))
    path = OUTPUT / "verification_rows.jsonl"
    rows = indexed_rows(path, cases, args.resume)
    for case in cases:
        if case["case_id"] not in rows:
            row = verify_raw(case, audit[case["case_id"]], warm[case["case_id"]], context)
            append_jsonl(path, row)
            rows[case["case_id"]] = row
        if (case["ordinal"]+1) % 200 == 0:
            print(f"PixArt full independent verification {case['ordinal']+1}/1800", flush=True)
    atomic_json(OUTPUT / "verification_summary.json", {
        "cases": len(rows), "signed_assets_reopened": len(rows), "gpu_inference_calls": 0,
        "evidence_changes_due_to_quantization": 0,
        "max_logit_quantization_error": max(r["max_logit_quantization_error"] for r in rows.values()),
        "max_region_mean_difference": max(r["region_mean_difference"] for r in rows.values()),
        "audit_rows_sha256": sha256_file(OUTPUT / "audit_rows.jsonl"),
        "verification_rows_sha256": sha256_file(path), "script_sha256": sha256_file(Path(__file__)),
        "acceptance_passed": len(rows) == 1800})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("audit", "verify"), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run_audit(args) if args.phase == "audit" else verify(args)
    except Exception as error:
        atomic_json(OUTPUT / f"failure-{time.time_ns()}.json", {"phase": args.phase, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
