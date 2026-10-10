"""Detector-free signing/auditing for the frozen twenty-case PixArt pilot."""
from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path

from protocol_v2_cache import decoded_rgb_key, sha256_file
from protocol_v2_common import audit_protocol_v2
from protocol_v2_geometry_lie_v1 import geometry_gate
from protocol_v2_main_curve_full_v1 import (
    append_jsonl, parent_path, prepare_parents, sign_child, trust_condition,
)
from protocol_v2_main_curve_full_v2 import signed_transform_matches
from protocol_v2_pilot_v1 import (
    atomic_json, load_json, load_jsonl, logit_mean, mapped_mask_and_retention, sanitize_pixels,
)
from protocol_v2_pixart_pilot_v1 import ROOT, checked_cases, validate_cached
from protocol_v2_cache import PixelCache
from read_actions_v2 import inspect_asset


def signing_cases(args) -> list[dict]:
    cases = checked_cases(args)
    if not load_json(args.output_dir / "warm_summary.json")["all_integer_counts_match"]:
        raise RuntimeError("warm acceptance required before signing")
    for case in cases:
        case["parent_source"] = str(ROOT / case["parent_source_relpath"])
        if sha256_file(case["parent_source"]) != case["parent_file_sha256"]:
            raise RuntimeError("parent source changed after selection")
    return cases


def signed_child(context: dict, case: dict) -> Path:
    output = context["output"]
    source = ROOT / case["child_source_relpath"]
    if sha256_file(source) != case["source_file_sha256"]:
        raise RuntimeError("child source changed after selection")
    pixels = output / "pixels" / f"{case['case_id']}.png"
    signed = output / "signed" / f"{case['case_id']}.png"
    if signed.exists():
        raise FileExistsError("pilot refuses to overwrite signed assets")
    sanitize_pixels(source, pixels)
    if decoded_rgb_key(pixels)[0] != case["pixel_key"]:
        raise RuntimeError("sanitizing changed pixels")
    sign_child(case, pixels=pixels, signed=signed,
               parent=parent_path(output, case["image_index"]), signer=context["signer"])
    return signed


def audit_case(context: dict, case: dict) -> dict:
    signed = signed_child(context, case)
    inspection = inspect_asset(signed)
    gate = geometry_gate(inspection)
    entry = context["cache"].get(case["pixel_key"])
    if entry is None:
        raise RuntimeError("cache miss; detector calls forbidden during auditing")
    matches = validate_cached(context, case, entry)
    mapped, retention = mapped_mask_and_retention(inspection["region"], inspection["geometry"])
    logit = logit_mean(entry["mask_logits"], mapped)
    raw = audit_protocol_v2(lineage=inspection["lineage_state"], reference_state=inspection["reference_source"],
                            semantics=inspection["chain_semantics"],
                            reference_bits=inspection["region_assertion"]["reference_bits"],
                            decoded_bits=entry["decoded_bits"], required_matches=24,
                            region_retention=retention, logit_in_region=logit)
    metrics = dict(retention=retention, logit=logit, gate=gate, pixel_key=decoded_rgb_key(signed)[0])
    checks = state_checks(case, inspection, raw, metrics)
    if matches != raw["match_count"] or not all(checks.values()):
        atomic_json(context["output"] / "audit_failure.json", {"case_id": case["case_id"], "checks": checks})
        raise RuntimeError(f"signed-chain acceptance failed: {case['case_id']}")
    conditions = {trust: trust_condition(raw, gate=gate, cropped_action_count=inspection["cropped_action_count"],
                                         editor_trust=trust) for trust in ("trusted-editor", "untrusted-editor")}
    return {"case_id": case["case_id"], "crop_frac": case["crop_frac"], "image_index": case["image_index"],
            "manifest_valid": inspection["manifest_valid"], "manifest_sha256": sha256_file(signed),
            "pixel_key": case["pixel_key"], "pixel_cache_hit": True, "gpu_inference_calls": 0,
            "lineage": raw["lineage"], "reference": raw["reference"], "semantics": raw["semantics"],
            "match_count": raw["match_count"], "legacy_match_count": case["legacy_match_count"],
            "region_retention": retention, "logit_in_region": logit, "evidence": raw["evidence"],
            "conditions": conditions, "state_checks": checks}


def state_checks(case: dict, inspection: dict, raw: dict, metrics: dict) -> dict:
    retention, logit = metrics["retention"], metrics["logit"]
    return {
        "manifest_valid": inspection["manifest_valid"], "geometry_accepted": metrics["gate"] == "accepted",
        "crop_action_count": inspection["cropped_action_count"] == (0 if case["crop_frac"] == 0 else 1),
        "ancestor_reference": raw["lineage"] == "ancestor" and raw["reference"] == "signed-ancestor",
        "capture_semantics": raw["semantics"] == "no-ai-disclosure-in-chain",
        "integer_matches": raw["match_count"] == case["legacy_match_count"],
        "reference_roundtrip": inspection["region_assertion"]["reference_bits"]
            == case["region_assertion"]["reference_bits"],
        "geometry_roundtrip": signed_transform_matches(case, inspection),
        "region_roundtrip": inspection["region"] == case["region_assertion"],
        "pixels_unchanged": metrics["pixel_key"] == case["pixel_key"],
        "finite_retention": math.isfinite(retention) and 0 <= retention <= 1,
        "finite_logit": math.isfinite(logit) and 0 <= logit <= 1,
    }


def validate_audit_rows(rows: list[dict]) -> None:
    if len(rows) != 20 or len({r["case_id"] for r in rows}) != 20:
        raise RuntimeError("twenty unique audit rows required")
    if not all(all(r["state_checks"].values()) for r in rows):
        raise RuntimeError("pilot row acceptance failed")


def summarize(rows: list[dict], elapsed: float) -> dict:
    validate_audit_rows(rows)
    escaped = [r for r in rows if r["match_count"] < 24]
    reviewed = sum(r["evidence"] in {"region-removed", "signal-suppressed"} for r in escaped)
    return {"schema": "protocol-v2-pixart-pilot-audit-1", "cases": len(rows),
            "valid_manifests": sum(r["manifest_valid"] for r in rows), "pixel_cache_hits": len(rows),
            "gpu_inference_calls": 0, "integer_count_matches": len(rows), "escapes": len(escaped),
            "trusted_editor_reviews": reviewed, "trusted_editor_silent": len(escaped)-reviewed,
            "evidence": dict(Counter(r["evidence"] for r in rows)),
            "trusted_policy": dict(Counter(r["conditions"]["trusted-editor"]["policy_verdict"] for r in rows)),
            "untrusted_policy": dict(Counter(r["conditions"]["untrusted-editor"]["policy_verdict"] for r in rows)),
            "per_crop": dict(Counter(f"{r['crop_frac']:.2f}" for r in rows)),
            "wall_seconds": elapsed, "population_estimate": False, "acceptance_passed": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/protocol_v2/pixart_pilot_v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/protocol_v2_pixart_pilot_v1.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results/protocol_v2/full_pixel_cache_v1/cache")
    parser.add_argument("--cert", type=Path, default=ROOT / "certs/ec_chain.pem")
    parser.add_argument("--key", type=Path, default=ROOT / "certs/ec_key.pem")
    args = parser.parse_args()
    cases = signing_cases(args)
    rows_path = args.output_dir / "audit_rows.jsonl"
    if rows_path.exists():
        raise FileExistsError("pilot audit is append-only; repeat runs require a new output directory")
    from protocol_v2_sign_pilot import create_signer
    context = dict(output=args.output_dir, cache=PixelCache(args.cache_dir),
                   signer=create_signer(args.cert.resolve(strict=True), args.key.resolve(strict=True)),
                   model_hash=sha256_file(ROOT / "ckpts/params.json"),
                   checkpoint_hash=sha256_file(ROOT / "ckpts/wam_mit.pth"))
    started = time.perf_counter()
    prepare_parents(cases, output=args.output_dir, signer=context["signer"], resume=False)
    for index, case in enumerate(cases, 1):
        append_jsonl(rows_path, audit_case(context, case))
        print(f"PixArt pilot signed/audited {index}/20", flush=True)
    summary = summarize(load_jsonl(rows_path), time.perf_counter()-started)
    summary.update(audit_script_sha256=sha256_file(Path(__file__)),
                   selected_cases_sha256=sha256_file(args.output_dir / "selected_cases.json"))
    atomic_json(args.output_dir / "audit_summary.json", summary)
    import json
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
