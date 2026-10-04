"""Fail-closed structural verification for granularity-control artifacts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from granularity_control_common_v1 import load_config, load_jsonl, validate_conditions


ROOT = Path(__file__).resolve().parents[1]


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--c1-cache", default=str(ROOT / "results" / "c1_v1" / "full" / "audit_rows.jsonl"))
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    conditions = validate_conditions(config["conditions"])
    experiment = Path(args.experiment_dir).resolve()
    embed = load_jsonl(experiment / "embed_rows.jsonl")
    require(bool(embed), "embed_rows.jsonl is empty")
    image_names = {row["input_image"] for row in embed}
    require(len(image_names) == len(embed), "duplicate input image in embed rows")
    threshold = int(config["matched_bits_threshold"])
    total_bits = int(config["message_bits"])
    rhos = {float(value) for value in config["removed_area_fractions"]}
    expected_marked = len(embed) * len(rhos)

    condition_rows: dict[str, list[dict]] = {}
    geometry: dict[tuple[int, float], tuple[list[int], float]] = {}
    for condition in conditions:
        rows = load_jsonl(experiment / f"detection_{condition}.jsonl")
        require(len(rows) == expected_marked,
                f"{condition}: {len(rows)} rows, expected {expected_marked}")
        ids = [row["row_id"] for row in rows]
        require(len(ids) == len(set(ids)), f"{condition}: duplicate row_id")
        for row in rows:
            require(row["condition"] == condition, f"{condition}: mislabeled row")
            require(float(row["rho_requested"]) in rhos, f"{condition}: unknown rho")
            matches = int(row["matched_bits"])
            require(int(row["total_bits"]) == total_bits, f"{condition}: payload length mismatch")
            require(abs(float(row["bit_accuracy"]) - matches / total_bits) < 1e-12,
                    f"{condition}: bit accuracy mismatch")
            require(bool(row["detected"]) == (matches >= threshold),
                    f"{condition}: threshold mismatch")
            key = (int(row["image_index"]), float(row["rho_requested"]))
            current = (list(row["crop_box_px"]), float(row["rho_actual"]))
            if condition == "subject_region":
                geometry[key] = current
        condition_rows[condition] = rows

    # Geometry is compared after every condition is loaded, independent of order.
    for condition, rows in condition_rows.items():
        for row in rows:
            key = (int(row["image_index"]), float(row["rho_requested"]))
            expected_box, expected_rho = geometry[key]
            require(list(row["crop_box_px"]) == expected_box,
                    f"{condition}: crop box differs from subject cache at {key}")
            require(abs(float(row["rho_actual"]) - expected_rho) < 1e-7,
                    f"{condition}: actual rho differs from subject cache at {key}")

    clean = load_jsonl(experiment / "detection_clean_reference.jsonl")
    require(len(clean) == len(embed), f"clean_reference: {len(clean)} rows, expected {len(embed)}")
    require(len({row["row_id"] for row in clean}) == len(clean), "clean_reference: duplicate row_id")
    for row in clean:
        matches = int(row["matched_bits"])
        require(abs(float(row["bit_accuracy"]) - matches / total_bits) < 1e-12,
                "clean_reference: bit accuracy mismatch")
        require(bool(row["detected"]) == (matches >= threshold),
                "clean_reference: threshold mismatch")

    c1_rows = {
        (row["input_image"], float(row["rho_requested"])): row
        for row in load_jsonl(Path(args.c1_cache))
        if row.get("method") == "center"
        and row.get("input_image") in image_names
        and float(row["rho_requested"]) in rhos
    }
    require(len(c1_rows) == expected_marked, "C1 comparison cache is incomplete")
    for row in condition_rows["subject_region"]:
        cached = c1_rows[(row["input_image"], float(row["rho_requested"]))]
        require(bool(row["detected"]) == bool(cached["detected"]), "subject detected mismatch")
        require(abs(float(row["bit_accuracy"]) - float(cached["bit_accuracy"])) < 1e-12,
                "subject bit accuracy mismatch")

    tolerance = float(config["energy_match_relative_tolerance"])
    random_condition = next(value for value in conditions if value.startswith("random_"))
    for row in embed:
        require(bool(row["subject_hash_match"]), "subject RGB hash mismatch")
        require(float(row["energy_match_relative_error"]) <= tolerance,
                "energy match exceeds tolerance")
        require(len(row["message_bits"]) == total_bits, "reference message length mismatch")
        if random_condition == "random_compact_equal_area":
            require(not bool(row["random_translation"]["wrapped"]), "compact control wrapped")
            require(int(row["random_translation"]["foreground_pixels"])
                    == int(row["subject_mask_pixels"]), "compact control area mismatch")

    summary = json.loads((experiment / "summary.json").read_text(encoding="utf-8"))
    require(int(summary["image_count"]) == len(embed), "summary image count mismatch")
    require(summary["config"]["version"] == config["version"], "summary config mismatch")
    require(all(math.isfinite(float(row["bit_accuracy_mean"])) for row in summary["summary"]),
            "summary contains non-finite bit accuracy")

    report = {
        "schema": "granularity-control-verification-1",
        "passed": True,
        "config_version": config["version"],
        "images": len(embed),
        "marked_rows_per_condition": expected_marked,
        "marked_conditions": list(conditions),
        "clean_rows": len(clean),
        "checks": {
            "unique_ids": True,
            "integer_threshold": True,
            "crop_geometry_equal": True,
            "subject_cache_exact": True,
            "subject_rgb_hash": True,
            "energy_match_tolerance": True,
            "random_area_exact": True,
            "summary_finite": True,
        },
    }
    output = Path(args.output) if args.output else experiment / "verification.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
