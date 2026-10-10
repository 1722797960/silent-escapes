"""Read-only CPU recheck of signed pilot assets and cache quantization."""
from pathlib import Path

import numpy as np

from protocol_v2_cache import PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_common import audit_protocol_v2
from protocol_v2_geometry_lie_v1 import geometry_gate
from protocol_v2_main_curve_full_v1 import package_version, parent_path, trust_condition
from protocol_v2_main_curve_full_v2 import signed_transform_matches
from protocol_v2_pilot_v1 import atomic_json, load_json, load_jsonl, logit_mean, mapped_mask_and_retention
from protocol_v2_pixart_pilot_v1 import ROOT, validate_cached
from read_actions_v2 import inspect_asset


def verify_case(case, saved, warm, context):
    signed = context["output"] / "signed" / f"{case['case_id']}.png"
    inspection = inspect_asset(signed)
    assert inspection["manifest_valid"] and geometry_gate(inspection) == "accepted"
    assert sha256_file(signed) == saved["manifest_sha256"]
    assert decoded_rgb_key(signed)[0] == case["pixel_key"]
    assert decoded_rgb_key(parent_path(context["output"], case["image_index"]))[0] == case["parent_pixel_key"]
    assert inspection["region_assertion"]["status"] == "valid"
    assert inspection["region"] == case["region_assertion"]
    assert signed_transform_matches(case, inspection)
    entry = context["cache"].get(case["pixel_key"])
    assert entry is not None and entry["decoded_bits"] == warm["decoded_bits"]
    assert validate_cached(context, case, entry) == saved["match_count"] == warm["match_count"]
    mapped, retention = mapped_mask_and_retention(inspection["region"], inspection["geometry"])
    cached_mean = logit_mean(entry["mask_logits"], mapped)
    with np.load(context["output"] / "raw_logits" / f"{case['case_id']}.npz", allow_pickle=False) as data:
        original = data["mask_logits"]
        assert original.dtype == np.float32 and np.isfinite(original).all()
        assert original.shape == entry["mask_logits"].shape
        max_error = float(np.max(np.abs(original - entry["mask_logits"].astype(np.float32))))
        raw_mean = logit_mean(original, mapped)
    assert abs(retention - saved["region_retention"]) < 1e-12
    assert abs(cached_mean - saved["logit_in_region"]) < 1e-7
    raw = audit_protocol_v2(
        lineage=inspection["lineage_state"], reference_state=inspection["reference_source"],
        semantics=inspection["chain_semantics"], reference_bits=inspection["region_assertion"]["reference_bits"],
        decoded_bits=entry["decoded_bits"], required_matches=24,
        region_retention=retention, logit_in_region=raw_mean)
    assert raw["evidence"] == saved["evidence"], "float16 caching changed evidence classification"
    for trust in ("trusted-editor", "untrusted-editor"):
        condition = trust_condition(raw, gate="accepted", cropped_action_count=inspection["cropped_action_count"],
                                    editor_trust=trust)
        assert condition == saved["conditions"][trust], "fresh signed-chain policy differs"
    return {"case_id": case["case_id"], "max_logit_quantization_error": max_error,
            "raw_region_mean": raw_mean, "cached_region_mean": cached_mean,
            "region_mean_difference": abs(raw_mean - cached_mean), "passed": True}


def main():
    output = ROOT / "results/protocol_v2/pixart_pilot_v1"
    target = output / "verification_summary.json"
    if target.exists():
        raise FileExistsError("preserve the first verification record")
    cases = load_json(output / "selected_cases.json")
    audit = {r["case_id"]: r for r in load_jsonl(output / "audit_rows.jsonl")}
    warm = {r["case_id"]: r for r in load_jsonl(output / "warm_rows.jsonl")}
    assert len(cases) == len(audit) == len(warm) == 20
    context = dict(output=output, cache=PixelCache(ROOT / "results/protocol_v2/full_pixel_cache_v1/cache"),
                   model_hash=sha256_file(ROOT / "ckpts/params.json"),
                   checkpoint_hash=sha256_file(ROOT / "ckpts/wam_mit.pth"))
    rows = [verify_case(c, audit[c["case_id"]], warm[c["case_id"]], context) for c in cases]
    summary = dict(cases=20, signed_assets_reopened=20, decoded_bits_identical=20,
                   evidence_changes_due_to_quantization=0, policy_changes_due_to_quantization=0,
                   max_logit_quantization_error=max(r["max_logit_quantization_error"] for r in rows),
                   max_region_mean_difference=max(r["region_mean_difference"] for r in rows),
                   gpu_inference_calls=0, acceptance_passed=True, rows=rows,
                   verification_script_sha256=sha256_file(Path(__file__)),
                   cpu_environment={name: package_version(name)
                                    for name in ("torch", "numpy", "Pillow", "c2pa-python")})
    atomic_json(target, summary)
    import json
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
