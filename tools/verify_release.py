#!/usr/bin/env python3
"""Validate the compact release and recompute headline paper values."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str):
    with (ROOT / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


def close(actual: float, expected: float, tolerance: float = 1e-3) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"expected {expected}, observed {actual}")


def curve_value(rows: list[dict], crop: float, field: str) -> float:
    row = next(item for item in rows if item["crop_frac"] == crop)
    return float(row[field])


def c1_value(rows: list[dict], method: str, rho: float, field: str) -> float:
    row = next(
        item for item in rows if item["method"] == method and item["rho"] == rho
    )
    return float(row[field])


def main() -> int:
    required = [
        "README.md",
        "DESIGN.md",
        "LICENSE",
        "REPRODUCIBILITY.md",
        "experiments/inpainting_common_v1.py",
        "experiments/inpainting_lama_v1.py",
        "experiments/inpainting_resign_v1.py",
        "experiments/inpainting_audit_v1.py",
        "experiments/summarize_inpainting_v1.py",
        "experiments/test_inpainting_common_v1.py",
        "results/summary_500.json",
        "results/regional_v2/summary.json",
        "results/regional_v2_pixart/summary.json",
        "results/c1_v1/full/attack_manifest.jsonl",
        "results/c1_v1/full/audit_rows.jsonl",
        "results/c1_v1/full/summary.json",
        "results/threshold_sensitivity_v1/threshold_sensitivity.json",
        "results/inpainting_e2e_v1/audit.csv",
        "results/inpainting_e2e_v1/summary.json",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    if missing:
        raise AssertionError(f"missing release files: {missing}")

    baseline = load("results/summary_500.json")
    close(float(baseline["wam_audit"]["metrics"]["TPR"]), 1.0)
    close(float(baseline["wam_audit"]["metrics"]["FPR"]), 0.0)

    sdxl = load("results/regional_v2/summary.json")
    pixart = load("results/regional_v2_pixart/summary.json")
    close(curve_value(sdxl, 0.0, "detection_rate_pct"), 99.0)
    close(curve_value(sdxl, 0.2, "detection_rate_pct"), 62.5)
    close(curve_value(sdxl, 0.4, "detection_rate_pct"), 0.5)
    close(curve_value(pixart, 0.0, "detection_rate_pct"), 98.0)
    close(curve_value(pixart, 0.2, "detection_rate_pct"), 51.0)
    close(curve_value(pixart, 0.4, "detection_rate_pct"), 0.5)

    c1 = load("results/c1_v1/full/summary.json")["summary"]
    close(c1_value(c1, "center", 0.4, "escape_rate_pct"), 17.0)
    close(c1_value(c1, "mask_aware", 0.4, "escape_rate_pct"), 55.0)

    expected_jsonl_rows = {
        "results/c1_v1/pilot/attack_manifest.jsonl": 360,
        "results/c1_v1/pilot/audit_rows.jsonl": 360,
        "results/c1_v1/full/attack_manifest.jsonl": 10_800,
        "results/c1_v1/full/audit_rows.jsonl": 10_800,
    }
    for relative, expected in expected_jsonl_rows.items():
        with (ROOT / relative).open(encoding="utf-8") as handle:
            observed = sum(1 for line in handle if line.strip())
        if observed != expected:
            raise AssertionError(f"{relative}: expected {expected} rows, observed {observed}")

    sensitivity = load(
        "results/threshold_sensitivity_v1/threshold_sensitivity.json"
    )["reference_point"]
    reference = {row["dataset"]: row for row in sensitivity}
    close(reference["SDXL centered"]["escape_flag_rate_pct"], 91.223)
    close(reference["PixArt-alpha centered"]["escape_flag_rate_pct"], 92.553)
    close(reference["SDXL mask-aware"]["escape_flag_rate_pct"], 93.435)

    inpainting = load("results/inpainting_e2e_v1/summary.json")
    if int(inpainting["n"]) != 200:
        raise AssertionError("inpainting summary must contain 200 assets")
    for invariant, observed in inpainting["invariants"].items():
        if int(observed) != 200:
            raise AssertionError(
                f"inpainting invariant {invariant}: expected 200, observed {observed}"
            )
    if int(inpainting["baseline_silent_escape"]["count"]) != 199:
        raise AssertionError("expected 199 baseline inpainting escapes")
    if int(inpainting["region_aware_recovery_among_escapes"]["count"]) != 199:
        raise AssertionError("expected all 199 inpainting escapes to be recovered")
    close(float(inpainting["median_bit_accuracy_before"]), 1.0)
    close(float(inpainting["median_bit_accuracy_after"]), 0.5)
    close(float(inpainting["median_logit_before"]), 0.96075)
    close(float(inpainting["median_logit_after"]), 0.0006)

    with (ROOT / "results/inpainting_e2e_v1/audit.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        inpainting_rows = list(csv.DictReader(handle))
    if len(inpainting_rows) != 200:
        raise AssertionError(
            f"inpainting audit: expected 200 rows, observed {len(inpainting_rows)}"
        )
    escaped = [row for row in inpainting_rows if row["baseline_audit"] == "SILENT_ESCAPE"]
    recovered = [
        row for row in escaped if row["region_aware_audit"] == "WATERMARK_SUPPRESSED"
    ]
    if len(escaped) != 199 or len(recovered) != 199:
        raise AssertionError(
            f"inpainting audit: expected 199/199 recovered escapes, observed "
            f"{len(recovered)}/{len(escaped)}"
        )

    blocked_suffixes = {".pth", ".pt", ".ckpt", ".safetensors"}
    blocked = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in blocked_suffixes
    ]
    if blocked:
        raise AssertionError(f"model weights must not be committed: {blocked}")

    oversized = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path.stat().st_size > 50 * 1024 * 1024
    ]
    if oversized:
        raise AssertionError(f"files larger than 50 MiB found: {oversized}")

    print("Release verification passed.")
    print("  SDXL regional detection: 99.0% -> 62.5% -> 0.5%")
    print("  PixArt-alpha detection:   98.0% -> 51.0% -> 0.5%")
    print("  rho=0.4 escape rate:      center 17.0%, mask-aware 55.0%")
    print("  defense flag rates:       91.223%, 92.553%, 93.435%")
    print("  exact-mask inpainting:    199/200 escapes, 199/199 recovered")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, StopIteration, json.JSONDecodeError) as error:
        print(f"Release verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
