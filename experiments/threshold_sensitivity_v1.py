"""Post-process saved audit rows into a defense-threshold sensitivity grid.

The script never reruns watermark inference.  It applies the region-aware
verdict rule to the saved per-asset region-retention and in-region mask-logit
values, after first asserting that the paper's reference operating point can
be reproduced exactly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "threshold_sensitivity_v1"
REGION_FLOORS = (0.01, 0.05, 0.10, 0.20)
SUPPRESSION_THRESHOLDS = (0.30, 0.50, 0.70)
DIRECTIONS = ("left", "right", "top", "bottom")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(successes: int, total: int,
                    z: float = 1.959963984540054) -> list[float | None]:
    if total == 0:
        return [None, None]
    rate = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (rate + z2 / (2.0 * total)) / denominator
    half_width = z * math.sqrt(
        rate * (1.0 - rate) / total + z2 / (4.0 * total * total)
    ) / denominator
    return [round(100.0 * (center - half_width), 3),
            round(100.0 * (center + half_width), 3)]


def derive_mask_aware(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the lowest-retention anchor without observing detector output."""
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["method"] in DIRECTIONS:
            key = (row["input_image"], float(row["rho_requested"]))
            grouped[key].append(row)

    priority = {method: index for index, method in enumerate(DIRECTIONS)}
    selected = []
    for key, candidates in grouped.items():
        if len(candidates) != len(DIRECTIONS):
            raise ValueError(
                f"mask-aware group {key!r} has {len(candidates)} rows; expected 4")
        selected.append(min(
            candidates,
            key=lambda row: (
                float(row["region_retention"]), priority[row["method"]])
        ))
    return selected


def classify(row: dict[str, Any], watermark_threshold: float,
             region_floor: float, suppression_threshold: float) -> str:
    if float(row["bit_accuracy"]) >= watermark_threshold:
        return "INTEGRITY_CLASH"
    if float(row["region_retention"]) < region_floor:
        return "TAMPERED_REGION"
    if float(row["logit_in_region"]) < suppression_threshold:
        return "WATERMARK_SUPPRESSED"
    return "SILENT_OTHER"


def is_uncropped(row: dict[str, Any], severity_field: str) -> bool:
    return abs(float(row[severity_field])) < 1e-12


def summarize(dataset: str, rows: list[dict[str, Any]], severity_field: str,
              watermark_threshold: float, region_floor: float,
              suppression_threshold: float) -> dict[str, Any]:
    verdicts = [
        classify(row, watermark_threshold, region_floor, suppression_threshold)
        for row in rows
    ]
    escapes = [
        verdict for verdict in verdicts if verdict != "INTEGRITY_CLASH"
    ]
    counts = {
        verdict: escapes.count(verdict)
        for verdict in ("TAMPERED_REGION", "WATERMARK_SUPPRESSED", "SILENT_OTHER")
    }
    flagged = counts["TAMPERED_REGION"] + counts["WATERMARK_SUPPRESSED"]

    uncropped_pairs = [
        (row, verdict) for row, verdict in zip(rows, verdicts)
        if is_uncropped(row, severity_field)
    ]
    uncropped_region_flags = sum(
        verdict in ("TAMPERED_REGION", "WATERMARK_SUPPRESSED")
        for _, verdict in uncropped_pairs
    )
    uncropped_decoder_misses = sum(
        float(row["bit_accuracy"]) < watermark_threshold
        for row, _ in uncropped_pairs
    )

    return {
        "dataset": dataset,
        "region_floor": region_floor,
        "suppression_threshold": suppression_threshold,
        "assets": len(rows),
        "silent_escapes": len(escapes),
        "tampered_region": counts["TAMPERED_REGION"],
        "watermark_suppressed": counts["WATERMARK_SUPPRESSED"],
        "silent_other": counts["SILENT_OTHER"],
        "flagged_escapes": flagged,
        "escape_flag_rate_pct": round(100.0 * flagged / len(escapes), 3),
        "escape_flag_ci95_pct": wilson_interval(flagged, len(escapes)),
        "uncropped_assets": len(uncropped_pairs),
        "uncropped_decoder_misses": uncropped_decoder_misses,
        "uncropped_region_flags": uncropped_region_flags,
        "uncropped_region_flag_rate_pct": round(
            100.0 * uncropped_region_flags / len(uncropped_pairs), 3),
        "uncropped_region_flag_ci95_pct": wilson_interval(
            uncropped_region_flags, len(uncropped_pairs)),
    }


def assert_reference(rows: list[dict[str, Any]], severity_field: str,
                     watermark_threshold: float, expected_escapes: int,
                     expected_flags: int, label: str) -> dict[str, Any]:
    result = summarize(
        label, rows, severity_field, watermark_threshold,
        region_floor=0.05, suppression_threshold=0.50)
    actual = (result["silent_escapes"], result["flagged_escapes"])
    expected = (expected_escapes, expected_flags)
    if actual != expected:
        raise RuntimeError(
            f"{label} reference mismatch: got escapes/flags={actual}, "
            f"expected={expected}")
    return result


def pct(value: float) -> str:
    return f"{value:.1f}%"


def write_markdown(path: Path, results: list[dict[str, Any]],
                   reference: list[dict[str, Any]]) -> None:
    lines = [
        "# Defense-threshold sensitivity (v1)",
        "",
        "Verdict priority: `INTEGRITY_CLASH` (decoder succeeds), then "
        "`TAMPERED_REGION`, then `WATERMARK_SUPPRESSED`, then `SILENT_OTHER`.",
        "The uncropped control rate is the fraction of severity-0 watermarked assets "
        "that receive either region-aware flag; it is not a benign-image FPR.",
        "",
        "## Reference-point reproduction",
        "",
        "| Dataset | Escapes | Flagged | Rate | Uncropped region flags |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in reference:
        lines.append(
            f"| {row['dataset']} | {row['silent_escapes']} | "
            f"{row['flagged_escapes']} | {pct(row['escape_flag_rate_pct'])} | "
            f"{row['uncropped_region_flags']}/{row['uncropped_assets']} |")

    for dataset in dict.fromkeys(row["dataset"] for row in results):
        lines.extend([
            "",
            f"## {dataset}",
            "",
            "| Region floor | Suppression threshold | Flagged/escapes | Flag rate "
            "[95% CI] | Tampered | Suppressed | Silent-other | Uncropped flags |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in (item for item in results if item["dataset"] == dataset):
            ci = row["escape_flag_ci95_pct"]
            marker = " **(reference)**" if (
                row["region_floor"] == 0.05
                and row["suppression_threshold"] == 0.50
            ) else ""
            lines.append(
                f"| {row['region_floor']:.0%} | {row['suppression_threshold']:.1f}"
                f"{marker} | {row['flagged_escapes']}/{row['silent_escapes']} | "
                f"{pct(row['escape_flag_rate_pct'])} "
                f"[{ci[0]:.1f}, {ci[1]:.1f}] | "
                f"{row['tampered_region']} | {row['watermark_suppressed']} | "
                f"{row['silent_other']} | {row['uncropped_region_flags']}/"
                f"{row['uncropped_assets']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "SDXL centered": ROOT / "results" / "defense_e2e" / "audit_retention.json",
        "PixArt-alpha centered": (
            ROOT / "results" / "defense_e2e_pixart" / "audit_retention.json"),
        "SDXL mask-aware": ROOT / "results" / "c1_v1" / "full" / "audit_rows.jsonl",
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise SystemExit("missing required inputs:\n" + "\n".join(missing))

    sdxl_payload = read_json(sources["SDXL centered"])
    pixart_payload = read_json(sources["PixArt-alpha centered"])
    c1_rows = read_jsonl(sources["SDXL mask-aware"])
    mask_aware_rows = derive_mask_aware(c1_rows)

    datasets = [
        ("SDXL centered", sdxl_payload["rows"], "crop_frac",
         float(sdxl_payload["threshold"]), 752, 686),
        ("PixArt-alpha centered", pixart_payload["rows"], "crop_frac",
         float(pixart_payload["threshold"]), 940, 870),
        ("SDXL mask-aware", mask_aware_rows, "rho_requested", 0.75, 914, 854),
    ]

    reference = []
    results = []
    for label, rows, severity_field, threshold, expected_escapes, expected_flags in datasets:
        reference.append(assert_reference(
            rows, severity_field, threshold, expected_escapes,
            expected_flags, label))
        for floor in REGION_FLOORS:
            for suppression in SUPPRESSION_THRESHOLDS:
                results.append(summarize(
                    label, rows, severity_field, threshold, floor, suppression))

    provenance = {
        label: {"path": str(path.resolve()), "sha256": sha256(path)}
        for label, path in sources.items()
    }
    payload = {
        "experiment": "Defense-threshold sensitivity",
        "version": "threshold_sensitivity_v1",
        "region_floors": list(REGION_FLOORS),
        "suppression_thresholds": list(SUPPRESSION_THRESHOLDS),
        "verdict_priority": [
            "INTEGRITY_CLASH", "TAMPERED_REGION",
            "WATERMARK_SUPPRESSED", "SILENT_OTHER"],
        "uncropped_metric_note": (
            "Severity-0 watermarked assets receiving TAMPERED_REGION or "
            "WATERMARK_SUPPRESSED; this is a no-edit control rate, not a "
            "benign-image false-positive rate."),
        "sources": provenance,
        "reference_point": reference,
        "grid": results,
    }
    (output_dir / "threshold_sensitivity.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    with (output_dir / "threshold_sensitivity.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    write_markdown(output_dir / "threshold_sensitivity.md", results, reference)
    print(f"reference baselines verified for {len(reference)} datasets")
    print(f"wrote {len(results)} grid rows to {output_dir}")


if __name__ == "__main__":
    main()
