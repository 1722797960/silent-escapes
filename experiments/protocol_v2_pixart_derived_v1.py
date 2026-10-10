"""CPU-only PixArt signed-chain sensitivity and legacy transition analysis."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from protocol_v2_cache import sha256_file
from protocol_v2_pixart_full_audit_v1 import fresh_audit
from protocol_v2_pixart_full_v1 import OUTPUT, ROOT, checked_cases, model_context
from protocol_v2_pilot_v1 import atomic_json, load_json, load_jsonl
from protocol_v2_sdxl_derived_v1 import (
    FLOORS, THRESHOLDS, ablations, bbox_retention, classify, report, summarize,
)
from read_actions_v2 import inspect_asset
from recompute_region_retention import region_metrics


DESTINATION = ROOT / "results/protocol_v2/pixart_derived_v1"
LEGACY = ROOT / "results/protocol_v2/pixart_legacy_provenance_v1"
LEGACY_HASH = "ba8cb143d8e56ba5042f54b029c631a111fed8758fc896d0a2791729bf2eea89"
OLD_NAMES = {"INTEGRITY_CLASH": "message-detected", "TAMPERED_REGION": "region-removed",
             "WATERMARK_SUPPRESSED": "signal-suppressed", "SILENT_OTHER": "silent-other"}
REVIEW = {"region-removed", "signal-suppressed"}


def unique_index(rows, key):
    indexed = {key(row): row for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError("duplicate comparison identity")
    return indexed


def transition_summary(rows):
    counts = Counter((r["legacy_evidence"], r["evidence"]) for r in rows)
    escaped = [r for r in rows if r["match_count"] < 24]
    return {"matrix": [{"from": a, "to": b, "count": n}
                       for (a, b), n in sorted(counts.items())],
            "changed_evidence": sum(n for (a, b), n in counts.items() if a != b),
            "review_to_silent": sum(r["legacy_evidence"] in REVIEW and
                                   r["evidence"] == "silent-other" for r in escaped),
            "silent_to_review": sum(r["legacy_evidence"] == "silent-other" and
                                   r["evidence"] in REVIEW for r in escaped),
            "legacy_reviews": sum(r["legacy_evidence"] in REVIEW for r in escaped),
            "v2_reviews": sum(r["evidence"] in REVIEW for r in escaped)}


def checked_legacy():
    script = ROOT / "experiments/recompute_region_retention.py"
    if sha256_file(script) != "4a575f6674a32e0df983f150de420ef6ff1c72fedce92235c96c84f0d669639f":
        raise RuntimeError("retained legacy correction script changed")
    path = LEGACY / "audit_retention.json"
    if sha256_file(path) != LEGACY_HASH:
        raise RuntimeError("legacy reconstruction provenance changed")
    bindings = {"audit.json": "audit_sha256", "embed_meta.json": "embed_meta_sha256"}
    selection = load_json(OUTPUT / "selection_summary.json")
    for filename, field in bindings.items():
        if sha256_file(LEGACY / filename) != selection[field]:
            raise RuntimeError("legacy reconstruction does not share frozen full-run inputs")
    legacy = load_json(path)
    if (legacy["threshold"], legacy["region_min_retention"], legacy["logit_mean_min"]) != (.75, .05, .5):
        raise RuntimeError("unexpected legacy operating point")
    index = unique_index(legacy["rows"], lambda r: (r["image"], round(r["crop_frac"], 2)))
    metadata = unique_index(load_json(LEGACY / "embed_meta.json")["images"], lambda r: r["input_image"])
    return index, metadata


def derived_row(case, saved, legacy, meta, context):
    if fresh_audit(case, context) != saved:
        raise RuntimeError(f"signed-chain/cache evidence changed: {case['case_id']}")
    inspection = inspect_asset(OUTPUT / "signed" / f"{case['case_id']}.png")
    _, retention = region_metrics(meta["bitmap64"], case["crop_frac"])
    old = {"match_count": saved["match_count"], "region_retention": retention,
           "logit_in_region": legacy["logit_in_region"]}
    if round(float(legacy["bit_accuracy"]) * 32) != saved["match_count"]:
        raise RuntimeError("legacy integer count changed")
    old_name = OLD_NAMES[legacy["audit_v2_region_aware"]]
    if classify(old, .05, .5) != old_name:
        raise RuntimeError("legacy unrounded-retention decision not reproducible")
    result = {k: saved[k] for k in ("case_id", "crop_frac", "match_count", "region_retention",
                                   "logit_in_region", "evidence", "manifest_sha256", "pixel_key")}
    result.update(image=case["image"], legacy_evidence=old_name,
                  legacy_retention_unrounded=retention, legacy_logit_saved=legacy["logit_in_region"],
                  bbox_retention=bbox_retention(inspection["region"], inspection["geometry"]))
    # An ordered input-swap diagnostic, not order-independent causal attribution.
    swapped = {**old, "region_retention": saved["region_retention"]}
    result["new_retention_old_saved_logit_evidence"] = classify(swapped, .05, .5)
    return result


def checked_inputs():
    cases = checked_cases()
    old, metadata = checked_legacy()
    audit_path = OUTPUT / "audit_rows.jsonl"
    audit_summary = load_json(OUTPUT / "audit_summary.json")
    verification = load_json(OUTPUT / "verification_summary.json")
    audit_hash = sha256_file(audit_path)
    if not (verification["acceptance_passed"] and verification["cases"] == 1800
            and verification["audit_rows_sha256"] == audit_hash == audit_summary["audit_rows_sha256"]):
        raise RuntimeError("independent full-run acceptance required")
    saved = unique_index(load_jsonl(audit_path), lambda r: r["case_id"])
    expected_old = {(c["image"], round(c["crop_frac"], 2)) for c in cases}
    if set(saved) != {c["case_id"] for c in cases} or set(old) != expected_old:
        raise RuntimeError("legacy/new comparison cohorts differ")
    return cases, old, metadata, saved, audit_hash


def validated_rows(cases, old, metadata, saved):
    context = model_context()
    rows = []
    for number, case in enumerate(cases, 1):
        identity = (case["image"], round(case["crop_frac"], 2))
        rows.append(derived_row(case, saved[case["case_id"]], old[identity], metadata[case["image"]], context))
        if number % 100 == 0:
            print(f"PixArt signed-chain/cache checks {number}/1800", flush=True)
    return rows


def build_summary(rows, audit_hash):
    default = summarize(rows, .05, .5)
    transitions = transition_summary(rows)
    if (default["escapes"], default["reviewed_escapes"], transitions["legacy_reviews"]) != (940, 869, 870):
        raise RuntimeError("default comparison does not reproduce frozen accepted results")
    payload = {"schema": "protocol-v2-pixart-derived-1", "default": default,
               "grid": [summarize(rows, f, t) for f in FLOORS for t in THRESHOLDS],
               "ablations": ablations(rows), "transitions": transitions,
               "source_rows_sha256": audit_hash, "legacy_reconstruction_sha256": LEGACY_HASH,
               "legacy_recompute_script_sha256": sha256_file(ROOT / "experiments/recompute_region_retention.py"),
               "script_sha256": sha256_file(Path(__file__)), "signed_chain_checks": len(rows),
               "pixel_cache_hits": len(rows), "gpu_inference_calls": 0, "signing_calls": 0,
               "trust_condition": "trusted-editor", "crop_parameter": "per-side c, not removed-area rho",
               "interval_scope": "row-level Wilson; repeated source images are not cluster-adjusted"}
    return payload


def write_results(rows, payload):
    transitions = payload["transitions"]
    atomic_json(DESTINATION / "summary.json", payload)
    atomic_json(DESTINATION / "rows.json", rows)
    atomic_json(DESTINATION / "changed_rows.json", [r for r in rows if r["legacy_evidence"] != r["evidence"]])
    text = report(payload).replace("SDXL v2", "PixArt v2")
    text += "\n## Legacy comparison\n\n"
    text += f"Old {transitions['legacy_reviews']}/940; v2 {transitions['v2_reviews']}/940. "
    text += f"{transitions['review_to_silent']} reviews become silent and "
    text += f"{transitions['silent_to_review']} silent cases become reviews.\n\n"
    text += "| Legacy evidence | V2 evidence | Cases |\n|---|---|---:|\n"
    for change in transitions["matrix"]:
        text += f"| {change['from']} | {change['to']} | {change['count']} |\n"
    text += ("\nThe legacy corrected audit was reconstructed on the server from hash-identical original "
             "audit/metadata and the retained CPU correction script. Its LF-normalized hash matches "
             "the local historical file; original files remain untouched. Legacy logits are saved "
             "rounded values, not new inference. V2 uses signed half-open pixel bounds with floor/ceil "
             "mask inclusion and fixed-64 sampling, rather than rounded normalized bounds and "
             "variable-window sampling. Per-row ordered input swaps separate decision changes "
             "when retention is replaced first; they are not order-independent causal attribution. "
             "Intervals are row-level Wilson, not source-cluster-adjusted uncertainty.\n")
    (DESTINATION / "report.md").write_text(text, encoding="utf-8")
    print(json.dumps({"default": payload["default"], "transitions": transitions,
                      "ablations": payload["ablations"]}), flush=True)


def main():
    DESTINATION.mkdir(parents=True, exist_ok=False)
    cases, old, metadata, saved, audit_hash = checked_inputs()
    rows = validated_rows(cases, old, metadata, saved)
    write_results(rows, build_summary(rows, audit_hash))


if __name__ == "__main__":
    main()
