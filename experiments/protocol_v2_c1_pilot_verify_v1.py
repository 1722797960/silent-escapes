"""Independent CPU readback: distinguish rerun differences from quantization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from protocol_v2_cache import PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_c1_pilot_v1 import ROOT, OUTPUT, cache_hashes, checked_cases, validate_cached
from protocol_v2_c1_pilot_audit_v1 import evidence_from_chain, conditions_from_chain
from protocol_v2_geometry_lie_v1 import geometry_gate
from protocol_v2_main_curve_full_v1 import package_version, parent_path
from protocol_v2_main_curve_full_v2 import signed_transform_matches
from protocol_v2_pilot_v1 import atomic_json, load_jsonl, logit_mean, mapped_mask_and_retention
from read_actions_v2 import inspect_asset


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def precision_diagnostics(original, cached, mapped):
    require(original.dtype == np.float32 and original.ndim == 2, "genuine raw float32 required")
    require(np.isfinite(original).all() and np.isfinite(cached).all(), "nonfinite logits")
    require(original.shape == cached.shape, "logit shape mismatch")
    quantized = original.astype(np.float16)
    return {"raw_mean": logit_mean(original, mapped), "rerun_quantized_mean": logit_mean(quantized, mapped),
            "cached_mean": logit_mean(cached, mapped),
            "quantization_max_abs": float(np.max(np.abs(original - quantized.astype(np.float32)))),
            "rerun_vs_cache_max_abs": float(np.max(np.abs(quantized.astype(np.float32)
                                                        - cached.astype(np.float32))))}


def verify_case(case, saved, warm, context):
    signed = context["output"] / "signed" / f"{case['case_id']}.png"
    inspection = inspect_asset(signed)
    require(inspection["manifest_valid"] and geometry_gate(inspection) == "accepted", "invalid signed chain")
    require(sha256_file(signed) == saved["manifest_sha256"], "signed file changed")
    require(decoded_rgb_key(signed)[0] == case["pixel_key"], "signed pixels changed")
    require(decoded_rgb_key(parent_path(context["output"], case["image_index"]))[0] == case["parent_pixel_key"],
            "parent pixels changed")
    require(inspection["region_assertion"]["status"] == "valid", "invalid signed reference")
    require(inspection["region"] == case["region_assertion"], "region readback mismatch")
    require(signed_transform_matches(case, inspection), "geometry readback mismatch")
    require(inspection["cropped_action_count"] == int(case["rho_requested"] > 0), "crop action count mismatch")
    entry = context["cache"].get(case["pixel_key"])
    require(entry is not None and entry["decoded_bits"] == warm["decoded_bits"], "decoded bits changed")
    if case["cache_hit"]:
        require(cache_hashes(context["cache"], case["pixel_key"]) == case["preserved_cache_hashes"],
                "preserved cache files changed")
    require(validate_cached(context, case, entry) == saved["match_count"] == warm["match_count"], "count mismatch")
    mapped, retention = mapped_mask_and_retention(inspection["region"], inspection["geometry"])
    raw_path = context["output"] / "raw_logits" / f"{case['case_id']}.npz"
    require(sha256_file(raw_path) == warm["raw_sha256"], "raw file changed")
    with np.load(raw_path, allow_pickle=False) as data:
        precision = precision_diagnostics(data["mask_logits"], entry["mask_logits"], mapped)
    require(abs(retention - saved["region_retention"]) < 1e-12, "retention mismatch")
    require(abs(precision["cached_mean"] - saved["logit_in_region"]) < 1e-7, "cached mean mismatch")
    outputs = compare_decisions(inspection, entry["decoded_bits"], retention, precision)
    require(outputs["cached"]["evidence"] == saved["evidence"], "saved evidence mismatch")
    require(outputs["cached"]["conditions"] == saved["conditions"], "saved policy mismatch")
    check_precision_acceptance(outputs, warm["cache_hit_before"], precision)
    return {"case_id": case["case_id"], "attack_id": case["attack_id"],
            "comparison_scope": warm["comparison_scope"], **precision, "passed": True}


def compare_decisions(inspection, bits, retention, precision):
    outputs = {}
    for label, field in (("raw", "raw_mean"), ("rerun_quantized", "rerun_quantized_mean"),
                         ("cached", "cached_mean")):
        evidence = evidence_from_chain(inspection, bits, retention, precision[field])
        outputs[label] = {"evidence": evidence["evidence"], "conditions": conditions_from_chain(evidence, inspection)}
    return outputs


def check_precision_acceptance(outputs, cache_hit, precision):
    require(outputs["raw"] == outputs["rerun_quantized"], "float16 quantization changed classification")
    require(outputs["raw"] == outputs["cached"], "rerun versus preserved cache changed classification")
    if not cache_hit:
        require(precision["rerun_vs_cache_max_abs"] == 0, "new cache is not exact raw-to-float16 output")


def joined_rows(output, cases):
    audit_rows = load_jsonl(output / "audit_rows.jsonl")
    warm_rows = load_jsonl(output / "warm_rows.jsonl")
    audit = {r["case_id"]: r for r in audit_rows}
    warm = {r["case_id"]: r for r in warm_rows}
    identities = {c["case_id"] for c in cases}
    require(len(audit_rows) == len(warm_rows) == 20 and set(audit) == set(warm) == identities,
            "expected exactly twenty joined rows")
    return audit, warm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/protocol_v2_c1_pilot_v1.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results/protocol_v2/full_pixel_cache_v1/cache")
    args = parser.parse_args()
    target = args.output_dir / "verification_summary.json"
    require(not target.exists(), "preserve the first verification record")
    cases = checked_cases(args)
    audit, warm = joined_rows(args.output_dir, cases)
    context = {"output": args.output_dir, "cache": PixelCache(args.cache_dir),
               "model_hash": sha256_file(ROOT / "ckpts/params.json"),
               "checkpoint_hash": sha256_file(ROOT / "ckpts/wam_mit.pth")}
    rows = []
    for case in cases:
        try:
            rows.append(verify_case(case, audit[case["case_id"]], warm[case["case_id"]], context))
        except RuntimeError as error:
            atomic_json(args.output_dir / "verification_failure.json",
                        {"case_id": case["case_id"], "error": str(error)})
            raise
    summary = {"cases": 20, "signed_assets_reopened": 20, "decoded_bits_identical": 20,
               "quantization_classification_changes": 0, "rerun_cache_classification_changes": 0,
               "max_quantization_error": max(r["quantization_max_abs"] for r in rows),
               "max_rerun_cache_error": max(r["rerun_vs_cache_max_abs"] for r in rows),
               "gpu_inference_calls": 0, "acceptance_passed": True, "population_estimate": False, "rows": rows,
               "verification_script_sha256": sha256_file(Path(__file__)),
               "cpu_environment": {n: package_version(n) for n in ("torch", "numpy", "Pillow", "c2pa-python")}}
    atomic_json(target, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
