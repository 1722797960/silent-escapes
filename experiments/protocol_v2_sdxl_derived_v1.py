"""CPU-only SDXL sensitivity and geometry-only ablations from verified assets.

Reads reference, region and crop transform back from the completed signed
chain; checks the exact pixel cache before deriving any new aggregate. No
signing or detector inference is performed. Existing outputs are preserved.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from protocol_v2_cache import PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_cache_warm_v1 import validate_entry
from protocol_v2_pilot_v1 import logit_mean, mapped_mask_and_retention
from read_actions_v2 import inspect_asset


ROOT = Path(__file__).resolve().parents[1]
FLOORS = (0.01, 0.05, 0.10, 0.20)
THRESHOLDS = (0.30, 0.50, 0.70)


def wilson_interval(successes: int, total: int) -> list[float | None]:
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1-rate) / total + z*z / (4*total*total)) / denominator
    return [100 * (center-half), 100 * (center+half)]


def validate_cohort(rows: list[dict]) -> None:
    counts = Counter(round(row["crop_frac"], 2) for row in rows)
    expected = Counter({round(i * .05, 2): 200 for i in range(9)})
    if len({row["case_id"] for row in rows}) != 1800 or counts != expected:
        raise RuntimeError("expected exactly 200 cases at each of nine per-side crop levels")


def classify(row: dict, floor: float, threshold: float) -> str:
    if row["match_count"] >= 24:
        return "message-detected"
    if row["region_retention"] < floor:
        return "region-removed"
    if row["logit_in_region"] < threshold:
        return "signal-suppressed"
    return "silent-other"


def bbox_retention(region: dict, geometry: dict) -> float:
    """Continuous normalized bbox area retained by the signed pixel crop."""
    bbox = [float(value) for value in region["bbox_csv"].split(",")]
    if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        raise ValueError("bbox must contain four finite values")
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("bbox is empty or outside normalized source bounds")
    if geometry["status"] == "missing":
        return 1.0
    if geometry["status"] != "valid":
        raise ValueError("bbox ablation requires valid signed geometry")
    transform = geometry["transform"]
    width, height = transform["source_size_px"]
    left, top, right, bottom = transform["crop_box_px"]
    retained = max(0.0, min(x1, right / width) - max(x0, left / width))
    retained *= max(0.0, min(y1, bottom / height) - max(y0, top / height))
    return retained / ((x1 - x0) * (y1 - y0))


def summarize(rows: list[dict], floor: float, threshold: float) -> dict:
    counts = Counter(classify(row, floor, threshold) for row in rows)
    escapes = len(rows) - counts["message-detected"]
    reviewed = counts["region-removed"] + counts["signal-suppressed"]
    controls = [row for row in rows if row["crop_frac"] == 0]
    control_reviews = sum(classify(row, floor, threshold) in
                          {"region-removed", "signal-suppressed"} for row in controls)
    return {
        "retention_floor": floor, "suppression_threshold": threshold,
        "assets": len(rows), "escapes": escapes, "reviewed_escapes": reviewed,
        "review_rate_pct": 100 * reviewed / escapes if escapes else None,
        "review_ci95_pct": wilson_interval(reviewed, escapes),
        "evidence_counts": dict(sorted(counts.items())),
        "no_edit_assets": len(controls), "no_edit_reviews": control_reviews,
        "no_edit_review_ci95_pct": wilson_interval(control_reviews, len(controls)),
    }


def ablations(rows: list[dict]) -> list[dict]:
    escaped = [row for row in rows if row["match_count"] < 24]
    severe = [row for row in escaped if row["crop_frac"] == 0.40]
    result = []
    for name in ("bbox-only", "bitmap-retention-only", "bitmap-plus-signal"):
        def reviewed(row: dict) -> bool:
            if name == "bbox-only":
                return row["bbox_retention"] < 0.05
            if name == "bitmap-retention-only":
                return row["region_retention"] < 0.05
            return classify(row, 0.05, 0.50) != "silent-other"
        count = sum(reviewed(row) for row in escaped)
        result.append({
            "condition": name, "reviewed_escapes": count, "escapes": len(escaped),
            "review_rate_pct": 100 * count / len(escaped),
            "review_ci95_pct": wilson_interval(count, len(escaped)),
            "c040_reviewed": sum(reviewed(row) for row in severe),
            "c040_escapes": len(severe),
        })
    return result


def validate_row(row: dict, full: Path, cache: PixelCache,
                 model_hash: str, checkpoint_hash: str) -> dict:
    asset = full / "signed" / f"{row['case_id']}.png"
    if sha256_file(asset) != row["manifest_sha256"]:
        raise RuntimeError(f"signed asset changed: {row['case_id']}")
    inspection = inspect_asset(asset)
    expected_status = "missing" if row["crop_frac"] == 0 else "valid"
    if not (inspection["manifest_valid"]
            and inspection["lineage_state"] == "ancestor"
            and inspection["reference_source"] == "signed-ancestor"
            and inspection["chain_semantics"] == "no-ai-disclosure-in-chain"
            and inspection["geometry"]["status"] == expected_status
            and inspection["cropped_action_count"] == (0 if row["crop_frac"] == 0 else 1)):
        raise RuntimeError(f"signed chain acceptance failed: {row['case_id']}")
    key, image = decoded_rgb_key(asset)
    if key != row["pixel_cache_key"]:
        raise RuntimeError(f"decoded pixels changed: {row['case_id']}")
    entry = cache.get(key)
    if entry is None:
        raise RuntimeError(f"cache miss; inference forbidden: {row['case_id']}")
    validate_entry(entry, key=key, image=image, model_sha256=model_hash,
                   checkpoint_sha256=checkpoint_hash)
    reference = inspection["region_assertion"]["reference_bits"]
    if len(reference) != 32 or len(entry["decoded_bits"]) != 32:
        raise RuntimeError("expected 32-bit signed reference and decoder output")
    matches = sum(a == b for a, b in zip(reference, entry["decoded_bits"]))
    mapped, retention = mapped_mask_and_retention(inspection["region"], inspection["geometry"])
    logit = logit_mean(entry["mask_logits"], mapped)
    if not (matches == row["match_count"] == row["legacy_match_count"]
            and math.isclose(retention, row["region_retention"], abs_tol=1e-12, rel_tol=0)
            and math.isclose(logit, row["logit_in_region"], abs_tol=1e-12, rel_tol=0)
            and classify(row, 0.05, 0.50) == row["trusted_evidence"]):
        raise RuntimeError(f"cached evidence differs from signed-chain result: {row['case_id']}")
    return {"case_id": row["case_id"], "crop_frac": row["crop_frac"],
            "match_count": matches, "region_retention": retention,
            "logit_in_region": logit,
            "bbox_retention": bbox_retention(inspection["region"], inspection["geometry"]),
            "manifest_sha256": row["manifest_sha256"], "pixel_cache_key": key}


def report(payload: dict) -> str:
    lines = ["# SDXL v2 derived analysis", "",
             "All inputs were re-read from valid signed chains and exact pixel caches.",
             "Results assume the editor's declared geometry is consumed (trusted-editor).",
             "No GPU inference or signing was performed. No-edit warnings are not benign FPR.",
             "", "## Threshold grid", "",
             "| Retention floor | Logit threshold | Reviewed / escapes | Rate [Wilson 95% CI] | No-edit reviews |",
             "|---:|---:|---:|---:|---:|"]
    for row in payload["grid"]:
        lo, hi = row["review_ci95_pct"]
        lines.append(f"| {row['retention_floor']:.0%} | {row['suppression_threshold']:.1f} | "
                     f"{row['reviewed_escapes']}/{row['escapes']} | {row['review_rate_pct']:.2f}% "
                     f"[{lo:.2f}, {hi:.2f}] | {row['no_edit_reviews']}/{row['no_edit_assets']} |")
    lines += ["", "## Geometry and signal ablations", "",
              "| Condition | Reviewed / escapes | Rate | c=0.40 reviewed / escapes |",
              "|---|---:|---:|---:|"]
    for row in payload["ablations"]:
        lines.append(f"| {row['condition']} | {row['reviewed_escapes']}/{row['escapes']} | "
                     f"{row['review_rate_pct']:.2f}% | {row['c040_reviewed']}/{row['c040_escapes']} |")
    lines += ["", "BBox-only uses continuous bbox intersection with signed half-open source-pixel",
              "crop coordinates, divided by original bbox area. Bitmap-only uses the v2",
              "floor/ceil mask-cell retention rule, with no signal branch. Their difference",
              "measures geometry-representation gain, not signal-check gain. The difference",
              "between bitmap-only and bitmap-plus-signal isolates the added signal branch.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-dir", type=Path, default=ROOT / "results/protocol_v2/main_curve_full_v2/full_v1")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results/protocol_v2/full_pixel_cache_v1/cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/protocol_v2/sdxl_derived_v1")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    full = args.full_dir.resolve(strict=True)
    source = full / "rows.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    validate_cohort(rows)
    model_hash = sha256_file(ROOT / "ckpts/params.json")
    checkpoint_hash = sha256_file(ROOT / "ckpts/wam_mit.pth")
    cache = PixelCache(args.cache_dir)
    validated = []
    for index, row in enumerate(rows, 1):
        validated.append(validate_row(row, full, cache, model_hash, checkpoint_hash))
        if index % 100 == 0:
            print(f"signed-chain/cache checks {index}/1800", flush=True)
    grid = [summarize(validated, floor, threshold) for floor in FLOORS for threshold in THRESHOLDS]
    default = summarize(validated, .05, .50)
    if (default["escapes"], default["reviewed_escapes"]) != (752, 684):
        raise RuntimeError("default v2 operating point mismatch")
    results = ablations(validated)
    payload = {"schema": "protocol-v2-sdxl-derived-1", "dataset": "SDXL per-side centered curve",
               "source_rows_sha256": sha256_file(source), "script_sha256": sha256_file(Path(__file__)),
               "params_sha256": model_hash, "checkpoint_sha256": checkpoint_hash,
               "crop_parameter": "fraction removed from each side c; not removed-area rho",
               "trust_condition": "trusted-editor", "gpu_inference_calls": 0, "signing_calls": 0,
               "signed_chain_checks": len(validated), "pixel_cache_hits": len(validated),
               "default": default, "grid": grid, "ablations": results}
    for name, value in (("summary.json", payload), ("rows.json", validated)):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    (output / "report.md").write_text(report(payload))
    print(json.dumps({"default": default, "ablations": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
