"""Detector-free C1 pilot signing; references and geometry are read from C2PA."""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

from protocol_v2_cache import PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_c1_inventory_v1 import contained_file
from protocol_v2_c1_pilot_v1 import ROOT, OUTPUT, checked_cases, validate_cached
from protocol_v2_common import audit_protocol_v2
from protocol_v2_geometry_lie_v1 import geometry_gate
from protocol_v2_main_curve_full_v1 import append_jsonl, prepare_parents, trust_condition, package_version
from protocol_v2_main_curve_full_v2 import signed_transform_matches
from protocol_v2_pilot_v1 import atomic_json, load_json, load_jsonl, logit_mean, mapped_mask_and_retention
from protocol_v2_pixart_audit_v1 import signed_child
from read_actions_v2 import inspect_asset


def state_checks(case, inspection, raw, metrics):
    return {
        "manifest_valid": inspection["manifest_valid"], "geometry_accepted": metrics["gate"] == "accepted",
        "crop_action_count": inspection["cropped_action_count"] == int(case["rho_requested"] > 0),
        "ancestor_reference": raw["lineage"] == "ancestor" and raw["reference"] == "signed-ancestor",
        "capture_semantics": raw["semantics"] == "no-ai-disclosure-in-chain",
        "integer_matches": raw["match_count"] == case["legacy_match_count"],
        "reference_roundtrip": inspection["region_assertion"]["reference_bits"]
            == case["region_assertion"]["reference_bits"],
        "geometry_roundtrip": signed_transform_matches(case, inspection),
        "region_roundtrip": inspection["region"] == case["region_assertion"],
        "pixels_unchanged": metrics["pixel_key"] == case["pixel_key"],
        "finite_retention": math.isfinite(metrics["retention"]) and 0 <= metrics["retention"] <= 1,
        "finite_logit": math.isfinite(metrics["logit"]) and 0 <= metrics["logit"] <= 1,
    }


def evidence_from_chain(inspection, bits, retention, mean):
    return audit_protocol_v2(
        lineage=inspection["lineage_state"], reference_state=inspection["reference_source"],
        semantics=inspection["chain_semantics"],
        reference_bits=inspection["region_assertion"]["reference_bits"],
        decoded_bits=bits, required_matches=24, region_retention=retention, logit_in_region=mean)


def conditions_from_chain(raw, inspection):
    return {trust: trust_condition(raw, gate="accepted", cropped_action_count=inspection["cropped_action_count"],
                                    editor_trust=trust) for trust in ("trusted-editor", "untrusted-editor")}


def audit_case(context, case):
    signed = signed_child(context, case)
    inspection = inspect_asset(signed)
    gate = geometry_gate(inspection)
    if gate != "accepted":
        raise RuntimeError("signed geometry rejected before evidence evaluation")
    entry = context["cache"].get(case["pixel_key"])
    if entry is None:
        raise RuntimeError("cache miss; GPU inference forbidden here")
    matches = validate_cached(context, case, entry)
    mapped, retention = mapped_mask_and_retention(inspection["region"], inspection["geometry"])
    mean = logit_mean(entry["mask_logits"], mapped)
    raw = evidence_from_chain(inspection, entry["decoded_bits"], retention, mean)
    metrics = {"gate": gate, "retention": retention, "logit": mean, "pixel_key": decoded_rgb_key(signed)[0]}
    checks = state_checks(case, inspection, raw, metrics)
    if matches != raw["match_count"] or not all(checks.values()):
        atomic_json(context["output"] / "audit_failure.json", {"case_id": case["case_id"], "checks": checks})
        raise RuntimeError(f"STOP: signed-chain acceptance failed: {case['attack_id']}")
    return {"case_id": case["case_id"], "attack_id": case["attack_id"], "rho_requested": case["rho_requested"],
            "rho_actual": case["rho_actual"], "method": case["method"], "image_index": case["image_index"],
            "manifest_valid": inspection["manifest_valid"], "manifest_sha256": sha256_file(signed),
            "pixel_key": case["pixel_key"], "gpu_inference_calls": 0,
            "match_count": raw["match_count"], "legacy_match_count": case["legacy_match_count"],
            "region_retention": retention, "logit_in_region": mean, "evidence": raw["evidence"],
            "legacy_evidence": case["legacy_evidence"],
            "lineage": raw["lineage"], "reference": raw["reference"], "semantics": raw["semantics"],
            "conditions": conditions_from_chain(raw, inspection), "state_checks": checks}


def summarize(rows, elapsed):
    if len(rows) != 20 or len({r["case_id"] for r in rows}) != 20:
        raise RuntimeError("twenty unique audit rows required")
    if not all(all(r["state_checks"].values()) for r in rows):
        raise RuntimeError("audit row acceptance failed")
    return {"schema": "protocol-v2-c1-pilot-audit-1", "cases": 20,
            "valid_manifests": sum(r["manifest_valid"] for r in rows), "integer_count_matches": 20,
            "gpu_inference_calls": 0, "escapes": sum(r["match_count"] < 24 for r in rows),
            "evidence": histogram(rows, "evidence"),
            "legacy_to_v2": dict(Counter(f"{r['legacy_evidence']} -> {r['evidence']}" for r in rows)),
            "trusted_policy": policy_histogram(rows, "trusted-editor"),
            "untrusted_policy": policy_histogram(rows, "untrusted-editor"),
            "wall_seconds": elapsed, "population_estimate": False, "acceptance_passed": True}


def histogram(rows, field):
    return dict(Counter(row[field] for row in rows))


def policy_histogram(rows, trust):
    return dict(Counter(row["conditions"][trust]["policy_verdict"] for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/protocol_v2_c1_pilot_v1.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results/protocol_v2/full_pixel_cache_v1/cache")
    parser.add_argument("--cert", type=Path, default=ROOT / "certs/ec_chain.pem")
    parser.add_argument("--key", type=Path, default=ROOT / "certs/ec_key.pem")
    args = parser.parse_args()
    cases = checked_cases(args)
    warm = load_json(args.output_dir / "warm_summary.json")
    if not warm["all_integer_counts_match"] or warm["inference_calls"] != 20 or warm["raw_float32_cases"] != 20:
        raise RuntimeError("twenty-case warming acceptance required")
    for case in cases:
        case["parent_source"] = str(contained_file(ROOT, case["parent_source_relpath"]))
        if sha256_file(case["parent_source"]) != case["parent_file_sha256"]:
            raise RuntimeError("parent source changed")
    rows_path = args.output_dir / "audit_rows.jsonl"
    if rows_path.exists() or (args.output_dir / "parent_index.json").exists():
        raise FileExistsError("preserve first signing outputs")
    from protocol_v2_sign_pilot import create_signer
    context = {"output": args.output_dir, "cache": PixelCache(args.cache_dir),
               "signer": create_signer(args.cert.resolve(strict=True), args.key.resolve(strict=True)),
               "model_hash": sha256_file(ROOT / "ckpts/params.json"),
               "checkpoint_hash": sha256_file(ROOT / "ckpts/wam_mit.pth")}
    started = time.perf_counter()
    prepare_parents(cases, output=args.output_dir, signer=context["signer"], resume=False)
    for index, case in enumerate(cases, 1):
        append_jsonl(rows_path, audit_case(context, case))
        print(f"C1 pilot signed/audited {index}/20", flush=True)
    summary = summarize(load_jsonl(rows_path), time.perf_counter() - started)
    summary.update(audit_script_sha256=sha256_file(Path(__file__)),
                   selected_cases_sha256=sha256_file(args.output_dir / "selected_cases.json"),
                   parent_signatures=len({r["image_index"] for r in cases}), child_signatures=20,
                   environment={n: package_version(n) for n in ("torch", "numpy", "Pillow", "c2pa-python")})
    atomic_json(args.output_dir / "audit_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
