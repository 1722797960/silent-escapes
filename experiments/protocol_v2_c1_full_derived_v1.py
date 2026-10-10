"""CPU-only curves, geometry-only mask-aware derivation and source-cluster CIs."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from protocol_v2_cache import sha256_file
from protocol_v2_c1_full_v1 import ROOT, OUTPUT, CONFIG, INVENTORY, checked_inputs, record_failure
from protocol_v2_c1_full_audit_v1 import REVIEW_EVIDENCE, group_counts, validate_audit_rows
from protocol_v2_c1_pilot_verify_v1 import require
from protocol_v2_geometry_lie_full_v1 import wilson_interval
from protocol_v2_pilot_v1 import atomic_json, load_json, load_jsonl


def mask_aware_rows(rows, selection):
    by_attack = {r["attack_id"]: r for r in rows}
    result = []
    for choice in selection:
        row = by_attack[choice["selected_attack_id"]]
        require(row["method"] in {"left", "right", "top", "bottom"}, "mask-aware must be an existing edge")
        require(row["pixel_key"] == choice["pixel_key"] and row["image_index"] == choice["image_index"]
                and row["rho_requested"] == choice["rho_requested"], "mask-aware frozen association drift")
        result.append({**row, "derived_method": "mask-aware"})
    require(len(result) == 1800 and len({r["attack_id"] for r in result}) == 1800, "mask-aware cohort incomplete")
    return result


def cell_statistics(rows):
    require(len(rows) == 200 and len({r["image_index"] for r in rows}) == 200,
            "each curve cell must contain one row per source")
    counts = group_counts(rows)
    counts["escape_rate_wilson95"] = wilson_interval(counts["escapes"], 200)
    counts["review_given_escape_wilson95"] = wilson_interval(counts["reviews"], counts["escapes"])
    counts["evidence_review_rate_wilson95"] = wilson_interval(
        sum(r["evidence"] in REVIEW_EVIDENCE for r in rows), 200)
    return counts


def clustered_review_interval(rows, seed, replicates):
    """Resample source images, retaining all dependent crop rows of each image."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["image_index"]].append(row)
    totals = []
    for image in sorted(groups):
        cohort = groups[image]
        totals.append((sum(r["match_count"] < 24 for r in cohort),
                       sum(r["match_count"] < 24 and r["evidence"] in REVIEW_EVIDENCE for r in cohort)))
    require(len(groups) == 200, "cluster interval requires the 200 source-image cohort")
    values = np.asarray(totals, dtype=np.int64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(replicates, len(values)))].sum(axis=1)
    valid = draws[:, 0] > 0
    ratios = draws[valid, 1] / draws[valid, 0]
    require(len(ratios) > 0, "no bootstrap replicate contains escapes")
    return {"method": "source-image cluster bootstrap percentile", "clusters": len(groups),
            "replicates": replicates, "valid_replicates": int(valid.sum()), "seed": seed,
            "interval95": np.quantile(ratios, [.025, .975]).tolist(),
            "estimand": "pooled reviewed escapes divided by pooled escapes"}


def paired_escape_difference(first, second, seed, replicates):
    """Paired source-level difference between two policies at one rho level."""
    left, right = ({r["image_index"]: r for r in group} for group in (first, second))
    require(len(left) == len(right) == 200 and set(left) == set(right), "paired source cohorts differ")
    deltas = np.asarray([int(right[i]["match_count"] < 24) - int(left[i]["match_count"] < 24)
                         for i in sorted(left)], dtype=np.int64)
    rng = np.random.default_rng(seed)
    means = deltas[rng.integers(0, 200, size=(replicates, 200))].mean(axis=1)
    return {"estimand": "mask-aware escape rate minus center escape rate", "difference": float(deltas.mean()),
            "paired_source_bootstrap_interval95": np.quantile(means, [.025, .975]).tolist(),
            "sources": 200, "seed": seed, "replicates": replicates}


def directional_transitions(rows):
    legacy = {"INTEGRITY_CLASH": "message-detected", "TAMPERED_REGION": "region-removed",
              "WATERMARK_SUPPRESSED": "signal-suppressed", "SILENT_OTHER": "silent-other"}
    transitions = Counter()
    changes = []
    for row in rows:
        old = legacy[row["legacy_evidence"]]
        transitions[f"{old} -> {row['evidence']}"] += 1
        if old != row["evidence"]:
            changes.append({"attack_id": row["attack_id"], "old": old, "new": row["evidence"]})
    return {"matrix": dict(transitions), "changed_records": len(changes), "changes": changes}


def validate_verified_sources():
    checked_inputs()
    summary = load_json(OUTPUT / "verification_summary.json")
    require(summary["acceptance_passed"] and summary["cases"] == 10800, "full signed verification required")
    for filename, digest in summary["bindings"].items():
        require(sha256_file(OUTPUT / filename) == digest, "verified input changed before derivation")
    require(sha256_file(OUTPUT / "verification_rows.jsonl") == summary["verification_rows_sha256"],
            "verification rows changed")


def derive():
    validate_verified_sources()
    target = OUTPUT / "derived"
    require(not target.exists(), "preserve first derived outputs")
    rows = load_jsonl(OUTPUT / "audit_rows.jsonl")
    validate_audit_rows(rows)
    config = load_json(CONFIG)
    aware = mask_aware_rows(rows, load_json(INVENTORY / "mask_aware_selection.json"))
    groups = defaultdict(list)
    for row in rows + aware:
        groups[(row.get("derived_method", row["method"]), row["rho_requested"])].append(row)
    cells = {f"{method}|{rho:.1f}": cell_statistics(group) for (method, rho), group in sorted(groups.items())}
    paired = {f"{rho:.1f}": paired_escape_difference(groups[("center", rho)], groups[("mask-aware", rho)],
                config["bootstrap_seed"], config["bootstrap_replicates"])
              for rho in sorted({r["rho_requested"] for r in rows})}
    summary = {"base_records": len(rows), "unique_pixels": 9800, "mask_aware_derived_records": len(aware),
               "mask_aware_is_independent_cohort": False, "per_method_rho": cells,
               "pooled_review_interval": clustered_review_interval(rows, config["bootstrap_seed"],
                                            config["bootstrap_replicates"]),
               "paired_center_vs_mask_aware": paired, "transitions": directional_transitions(rows),
               "gpu_inference_calls": 0, "signing_calls": 0, "acceptance_passed": True,
               "audit_rows_sha256": sha256_file(OUTPUT / "audit_rows.jsonl"),
               "verification_summary_sha256": sha256_file(OUTPUT / "verification_summary.json"),
               "script_sha256": sha256_file(Path(__file__))}
    target.mkdir()
    atomic_json(target / "summary.json", summary)
    atomic_json(target / "mask_aware_rows.json", aware)
    report = ["# C1 full verifier-v2 derived results", "", "10,800 signed base rows; 9,800 unique pixels.",
              "Mask-aware selects 1,800 existing edge rows by geometry, not new inference or signing.",
              "Cell Wilson intervals use one row per source; pooled intervals resample the 200 source clusters.",
              "The benchmark uses removed area rho and LANCZOS resize to original dimensions, not per-side c.",
              "Both trusted and untrusted editor policies remain separate; geometry signatures do not prove truth."]
    (target / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("C1 full derived curves, paired intervals and transitions completed", flush=True)


if __name__ == "__main__":
    try:
        derive()
    except Exception as error:
        record_failure("derived", error)
        raise
