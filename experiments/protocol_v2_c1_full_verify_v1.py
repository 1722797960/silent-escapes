"""Independent full CPU readback: every signed row, raw output and both policies."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from protocol_v2_cache import decoded_rgb_key, sha256_file
from protocol_v2_c1_full_v1 import (
    ROOT, OUTPUT, COUNT, UNIQUE, checked_entry, indexed_rows, model_context, record_failure,
)
from protocol_v2_c1_full_audit_v1 import fresh_audit, ready_cases, validate_audit_rows
from protocol_v2_c1_pilot_v1 import cache_hashes
from protocol_v2_c1_inventory_v1 import contained_file
from protocol_v2_c1_pilot_verify_v1 import require, precision_diagnostics, compare_decisions, check_precision_acceptance
from protocol_v2_main_curve_full_v1 import append_jsonl, package_version, parent_path
from protocol_v2_pilot_v1 import atomic_json, load_json, load_jsonl, mapped_mask_and_retention
from read_actions_v2 import inspect_asset


def verify_case(case, saved, warm, context):
    require(fresh_audit(case, context) == saved, "fresh signed-chain audit differs from saved row")
    require(decoded_rgb_key(parent_path(OUTPUT, case["image_index"]))[0] == case["parent_pixel_key"],
            "parent decoded pixels changed")
    inspection = inspect_asset(OUTPUT / "signed" / f"{case['case_id']}.png")
    entry = checked_entry(context, case, complete=True)
    require(entry["decoded_bits"] == warm["decoded_bits"], "cached decoded bits differ from raw receipt")
    mapped, retention = mapped_mask_and_retention(inspection["region"], inspection["geometry"])
    path = contained_file(ROOT, warm["raw_logits_relpath"])
    require(sha256_file(path) == warm["raw_sha256"], "raw float32 hash changed")
    with np.load(path, allow_pickle=False) as arrays:
        precision = precision_diagnostics(arrays["mask_logits"], entry["mask_logits"], mapped)
    outputs = compare_decisions(inspection, entry["decoded_bits"], retention, precision)
    require(outputs["cached"]["evidence"] == saved["evidence"], "cached evidence changed")
    require(outputs["cached"]["conditions"] == saved["conditions"], "cached policies changed")
    check_precision_acceptance(outputs, case["cache_hit"], precision)
    return {"case_id": case["case_id"], "attack_id": case["attack_id"], "pixel_key": case["pixel_key"],
            "manifest_sha256": saved["manifest_sha256"], "raw_sha256": warm["raw_sha256"],
            "verified_cache_hashes": cache_hashes(context["cache"], case["pixel_key"]),
            "comparison_scope": warm["comparison_scope"], **precision, "passed": True}


def verification_bindings():
    paths = ("audit_rows.jsonl", "audit_summary.json", "warm_rows.jsonl", "warm_summary.json",
             "selected_cases.json", "unique_pixels.json")
    return {name: sha256_file(OUTPUT / name) for name in paths}


def ensure_verification_inputs(resume):
    bindings = verification_bindings()
    path = OUTPUT / "verification_inputs.json"
    if path.exists():
        require(resume and load_json(path) == bindings, "verification resume inputs changed")
    else:
        atomic_json(path, bindings)
    return bindings


def verify(args):
    cases, pixels = ready_cases()
    bindings = ensure_verification_inputs(args.resume)
    audit = indexed_rows(OUTPUT / "audit_rows.jsonl", {c["case_id"] for c in cases}, True)
    validate_audit_rows(list(audit.values()))
    warm = indexed_rows(OUTPUT / "warm_rows.jsonl", {p["pixel_key"] for p in pixels}, True, "pixel_key")
    require(len(warm) == UNIQUE, "raw provenance incomplete")
    path = OUTPUT / "verification_rows.jsonl"
    done = indexed_rows(path, {c["case_id"] for c in cases}, args.resume)
    context = model_context()
    started = time.perf_counter()
    for case in cases:
        identity = case["case_id"]
        if identity in done:
            validate_completed(case, done[identity], audit[identity], warm[case["pixel_key"]], context)
            continue
        row = verify_case(case, audit[identity], warm[case["pixel_key"]], context)
        append_jsonl(path, row)
        done[identity] = row
        if len(done) % 25 == 0 or len(done) == COUNT:
            atomic_json(OUTPUT / "verification_progress.json", {"completed": len(done), "total": COUNT,
                        "wall_seconds_current_session": time.perf_counter()-started})
            print(f"C1 full independently verified {len(done)}/{COUNT}", flush=True)
    require(len(done) == COUNT and all(r["passed"] for r in done.values()), "full verification incomplete")
    write_verification_summary(done, path, bindings)


def write_verification_summary(done, path, bindings):
    summary = {"cases": COUNT, "unique_pixels": UNIQUE, "signed_assets_reopened": COUNT,
        "decoded_bits_identical": COUNT, "quantization_classification_changes": 0,
        "rerun_cache_classification_changes": 0, "gpu_inference_calls": 0, "acceptance_passed": True,
        "max_quantization_error": max(r["quantization_max_abs"] for r in done.values()),
        "max_rerun_cache_error": max(r["rerun_vs_cache_max_abs"] for r in done.values()),
        "verification_rows_sha256": sha256_file(path), "bindings": bindings,
        "script_sha256": sha256_file(Path(__file__)),
        "environment": {n: package_version(n) for n in ("torch", "numpy", "Pillow", "c2pa-python")}}
    atomic_json(OUTPUT / "verification_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def validate_completed(case, verified, audit, warm, context):
    require(verified["passed"] and verified["manifest_sha256"] == audit["manifest_sha256"], "resume acceptance drift")
    require(sha256_file(OUTPUT / "signed" / f"{case['case_id']}.png") == verified["manifest_sha256"],
            "verified signed file drift")
    require(sha256_file(contained_file(ROOT, warm["raw_logits_relpath"])) == verified["raw_sha256"],
            "verified raw file drift")
    entry = checked_entry(context, case, complete=True)
    require(cache_hashes(context["cache"], case["pixel_key"]) == verified["verified_cache_hashes"],
            "verified cache files drift")
    require(entry["decoded_bits"] == warm["decoded_bits"], "verified bits drift")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        verify(args)
    except Exception as error:
        record_failure("verify", error)
        raise


if __name__ == "__main__":
    main()
