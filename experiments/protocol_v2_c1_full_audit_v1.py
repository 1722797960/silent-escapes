"""Resumable detector-free C1 signing, with separate trusted/untrusted policies."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from protocol_v2_cache import decoded_rgb_key, sha256_file
from protocol_v2_c1_full_v1 import (
    ROOT, OUTPUT, COUNT, checked_inputs, checked_entry, indexed_rows, model_context, record_failure,
)
from protocol_v2_c1_inventory_v1 import contained_file
from protocol_v2_c1_pilot_audit_v1 import state_checks, evidence_from_chain, conditions_from_chain, policy_histogram
from protocol_v2_c1_pilot_verify_v1 import require
from protocol_v2_geometry_lie_v1 import geometry_gate
from protocol_v2_main_curve_full_v1 import append_jsonl, package_version, parent_path, prepare_parents, sign_child
from protocol_v2_pilot_v1 import (
    atomic_json, load_json, load_jsonl, logit_mean, mapped_mask_and_retention, sanitize_pixels,
)
from read_actions_v2 import inspect_asset

REVIEW_EVIDENCE = {"region-removed", "signal-suppressed"}


def ready_cases():
    cases, pixels = checked_inputs()
    warm = load_json(OUTPUT / "warm_summary.json")
    require(warm["acceptance_passed"] and warm["unique_pixels"] == 9800 and warm["new_inference_calls"] == 9780,
            "full warming acceptance required")
    require(sha256_file(OUTPUT / "warm_rows.jsonl") == warm["warm_rows_sha256"], "warm rows changed")
    require(sha256_file(OUTPUT / "inference_starts.jsonl") == warm["inference_starts_sha256"], "budget journal changed")
    by_key = {p["pixel_key"]: p for p in pixels}
    for case in cases:
        unique = by_key[case["pixel_key"]]
        case.update(cache_hit=unique["cache_hit"], preserved_cache_hashes=unique["preserved_cache_hashes"])
        case["parent_source"] = str(contained_file(ROOT, case["parent_source_relpath"]))
    return cases, pixels


def ensure_signed(case, context, resume):
    source = contained_file(ROOT / "results/c1_v1/full", case["output_relpath"])
    require(sha256_file(source) == case["source_file_sha256"], "child source drift")
    signed = OUTPUT / "signed" / f"{case['case_id']}.png"
    if signed.exists():
        require(resume, "signed orphan requires explicit resume and validation")
        return signed
    pixels = OUTPUT / "pixels" / f"{case['case_id']}.png"
    sanitize_pixels(source, pixels)
    require(decoded_rgb_key(pixels)[0] == case["pixel_key"], "sanitizing changed pixels")
    sign_child(case, pixels=pixels, signed=signed, parent=parent_path(OUTPUT, case["image_index"]),
               signer=context["signer"])
    return signed


def fresh_audit(case, context):
    signed = OUTPUT / "signed" / f"{case['case_id']}.png"
    inspection = inspect_asset(signed)
    gate = geometry_gate(inspection)
    require(inspection["manifest_valid"] and gate == "accepted", "invalid signed asset/geometry")
    entry = checked_entry(context, case, complete=True)
    mapped, retention = mapped_mask_and_retention(inspection["region"], inspection["geometry"])
    mean = logit_mean(entry["mask_logits"], mapped)
    raw = evidence_from_chain(inspection, entry["decoded_bits"], retention, mean)
    checks = state_checks(case, inspection, raw, {"gate": gate, "retention": retention,
                          "logit": mean, "pixel_key": decoded_rgb_key(signed)[0]})
    require(all(checks.values()), f"signed-chain row checks failed: {case['case_id']}: {checks}")
    return {"case_id": case["case_id"], "attack_id": case["attack_id"], "image_index": case["image_index"],
            "rho_requested": case["rho_requested"], "rho_actual": case["rho_actual"], "method": case["method"],
            "manifest_valid": True, "manifest_sha256": sha256_file(signed), "pixel_key": case["pixel_key"],
            "gpu_inference_calls": 0, "match_count": raw["match_count"],
            "legacy_match_count": case["legacy_match_count"],
            "region_retention": retention, "logit_in_region": mean, "evidence": raw["evidence"],
            "legacy_evidence": case["legacy_evidence"], "conditions": conditions_from_chain(raw, inspection),
            "lineage": raw["lineage"], "reference": raw["reference"], "semantics": raw["semantics"],
            "state_checks": checks}


def validate_audit_rows(rows):
    require(len(rows) == COUNT and len({r["case_id"] for r in rows}) == COUNT, "full audit incomplete")
    require({r["attack_id"] for r in rows} == {c["attack_id"] for c in load_json(OUTPUT / "selected_cases.json")},
            "audit identities differ from selection")
    require(all(all(r["state_checks"].values()) and r["manifest_valid"] for r in rows), "invalid full audit row")
    require(sum(r["match_count"] < 24 for r in rows) == 2774, "historical integer escapes changed")
    require(sum(r["gpu_inference_calls"] for r in rows) == 0, "detector-free audit boundary violated")


def group_counts(rows):
    escaped = [r for r in rows if r["match_count"] < 24]
    reviewed = sum(r["evidence"] in REVIEW_EVIDENCE for r in escaped)
    return {"records": len(rows), "escapes": len(escaped), "reviews": reviewed,
            "silent_other": sum(r["evidence"] == "silent-other" for r in escaped),
            "evidence": dict(Counter(r["evidence"] for r in rows)),
            "trusted_policy": policy_histogram(rows, "trusted-editor"),
            "untrusted_policy": policy_histogram(rows, "untrusted-editor")}


def summarize(rows):
    validate_audit_rows(rows)
    cells = {}
    for method in sorted({r["method"] for r in rows}):
        for rho in sorted({r["rho_requested"] for r in rows}):
            subset = [r for r in rows if r["method"] == method and r["rho_requested"] == rho]
            require(len(subset) == 200, "method/rho cell incomplete")
            cells[f"{method}|{rho:.1f}"] = group_counts(subset)
    summary = {"schema": "protocol-v2-c1-full-audit-1", **group_counts(rows),
               "valid_manifests": COUNT, "integer_count_matches": COUNT, "gpu_inference_calls": 0,
               "per_method_rho": cells, "crop_parameter": "removed area rho; not per-side c",
               "legacy_to_v2": dict(Counter(f"{r['legacy_evidence']} -> {r['evidence']}" for r in rows)),
               "audit_rows_sha256": sha256_file(OUTPUT / "audit_rows.jsonl"),
               "selected_cases_sha256": sha256_file(OUTPUT / "selected_cases.json"),
               "script_sha256": sha256_file(Path(__file__)), "acceptance_passed": True}
    atomic_json(OUTPUT / "audit_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_method_rho"}, indent=2), flush=True)


def checked_parents(cases, context, resume):
    parents = {c["image_index"]: c for c in cases}
    require(len(parents) == 200, "expected 200 reusable parents")
    for case in parents.values():
        require(sha256_file(case["parent_source"]) == case["parent_file_sha256"], "parent input changed")
    prepare_parents(cases, output=OUTPUT, signer=context["signer"], resume=resume)
    for case in parents.values():
        require(decoded_rgb_key(parent_path(OUTPUT, case["image_index"]))[0] == case["parent_pixel_key"],
                "parent signing changed pixels")


def run_audit(args):
    cases, _ = ready_cases()
    rows_path = OUTPUT / "audit_rows.jsonl"
    done = indexed_rows(rows_path, {c["case_id"] for c in cases}, args.resume)
    context = model_context()
    from protocol_v2_sign_pilot import create_signer
    context["signer"] = create_signer(ROOT / "certs/ec_chain.pem", ROOT / "certs/ec_key.pem")
    started = time.perf_counter()
    checked_parents(cases, context, args.resume)
    for case in cases:
        identity = case["case_id"]
        if identity in done:
            require(sha256_file(OUTPUT / "signed" / f"{identity}.png") == done[identity]["manifest_sha256"],
                    "completed signed file changed")
            continue
        ensure_signed(case, context, args.resume)
        row = fresh_audit(case, context)
        append_jsonl(rows_path, row)
        done[identity] = row
        if len(done) % 25 == 0 or len(done) == COUNT:
            atomic_json(OUTPUT / "audit_progress.json", {"completed": len(done), "total": COUNT,
                        "wall_seconds_current_session": time.perf_counter()-started})
            print(f"C1 full signed/audited {len(done)}/{COUNT}", flush=True)
    summarize(list(done.values()))
    atomic_json(OUTPUT / f"audit_environment-{time.time_ns()}.json", {
        "wall_seconds_current_session": time.perf_counter()-started, "gpu_inference_calls": 0,
        **{n: package_version(n) for n in ("torch", "numpy", "Pillow", "c2pa-python")}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run_audit(args)
    except Exception as error:
        record_failure("audit", error)
        raise


if __name__ == "__main__":
    main()
