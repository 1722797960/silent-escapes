"""Summarize benign edit flags, clean-negative FPR, and threshold costs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from benign_controls_common_v1 import classify_benign
from inpainting_common_v1 import wilson_interval


REGION_FLOORS = (0.01, 0.05, 0.10, 0.20)
SUPPRESSION_THRESHOLDS = (0.30, 0.50, 0.70)


def rate_record(successes: int, total: int) -> dict:
    low, high = wilson_interval(successes, total)
    return {
        "count": successes,
        "total": total,
        "rate": successes / total if total else 0.0,
        "wilson95": [low, high],
    }


def median_or_none(values: list[float | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.median(present) if present else None


def make_contact_sheet(experiment_dir: Path, signed_rows: list[dict]) -> None:
    by_image: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in signed_rows:
        by_image[row["input_image"]][row["condition"]] = row
    conditions = [
        "identity_resave",
        "jpeg_q90",
        "resize_90",
        "safe_crop_1pct",
        "clean_negative",
    ]
    tile = 180
    label_height = 34
    canvas = Image.new(
        "RGB",
        (tile * len(conditions), (tile + label_height) * len(by_image)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, (image_name, condition_rows) in enumerate(by_image.items()):
        y = row_index * (tile + label_height)
        for column, condition in enumerate(conditions):
            row = condition_rows[condition]
            path = experiment_dir / "unsigned" / condition / row["unsigned_output"]
            image = Image.open(path).convert("RGB")
            image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(image, (column * tile, y))
            draw.text((column * tile + 4, y + tile + 3), condition, fill="black")
        draw.text((4, y + tile + 18), image_name[:48], fill="black")
    canvas.save(experiment_dir / "contact_sheet.png")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=root / "results/benign_controls_v1_pilot",
    )
    args = parser.parse_args()

    audit = json.loads((args.experiment_dir / "audit.json").read_text(encoding="utf-8"))
    signed = json.loads(
        (args.experiment_dir / "signed_meta.json").read_text(encoding="utf-8")
    )
    rows = audit["rows"]
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    condition_summaries = []
    for condition, condition_rows in sorted(by_condition.items()):
        false_flags = sum(row["benign_region_false_flag"] for row in condition_rows)
        detector_fp = sum(row["detector_false_positive"] for row in condition_rows)
        decoder_misses = sum(row["decoder_miss"] for row in condition_rows)
        condition_summaries.append(
            {
                "condition": condition,
                "n": len(condition_rows),
                "watermarked": bool(condition_rows[0]["watermarked"]),
                "manifest_valid": sum(row["manifest_valid"] for row in condition_rows),
                "region_assertion_present": sum(
                    row["region_assertion_present"] for row in condition_rows
                ),
                "payload_hash_ok": sum(row["payload_hash_ok"] is True for row in condition_rows),
                "decoder_miss": rate_record(decoder_misses, len(condition_rows)),
                "benign_region_false_flag": rate_record(false_flags, len(condition_rows)),
                "clean_detector_false_positive": rate_record(detector_fp, len(condition_rows)),
                "median_bit_accuracy": median_or_none(
                    [row["bit_accuracy_after"] for row in condition_rows]
                ),
                "median_logit_in_region": median_or_none(
                    [row["logit_in_region_after"] for row in condition_rows]
                ),
                "min_region_retention": min(
                    (
                        float(row["region_retention"])
                        for row in condition_rows
                        if row["region_retention"] is not None
                    ),
                    default=None,
                ),
                "verdict_counts": dict(Counter(row["verdict"] for row in condition_rows)),
            }
        )

    grid = []
    for condition, condition_rows in sorted(by_condition.items()):
        if not condition_rows[0]["watermarked"]:
            continue
        for floor in REGION_FLOORS:
            for suppression in SUPPRESSION_THRESHOLDS:
                flags = 0
                for row in condition_rows:
                    _, false_flag = classify_benign(
                        expected_region_assertion=True,
                        manifest_valid=bool(row["manifest_valid"]),
                        assertion_present=bool(row["region_assertion_present"]),
                        payload_hash_ok=bool(row["payload_hash_ok"]),
                        bit_accuracy=float(row["bit_accuracy_after"]),
                        region_retention=float(row["region_retention"]),
                        logit_in_region=float(row["logit_in_region_after"]),
                        bit_threshold=float(audit["bit_threshold"]),
                        retention_floor=floor,
                        logit_threshold=suppression,
                    )
                    flags += false_flag
                grid.append(
                    {
                        "condition": condition,
                        "region_floor": floor,
                        "suppression_threshold": suppression,
                        "false_flags": flags,
                        "n": len(condition_rows),
                        "false_flag_rate": flags / len(condition_rows),
                        "wilson95": list(wilson_interval(flags, len(condition_rows))),
                    }
                )

    summary = {
        "experiment": audit["experiment"],
        "version": audit["version"],
        "pilot": len(by_condition.get("identity_resave", [])) < 200,
        "false_flag_definition": audit["false_flag_definition"],
        "reference_operating_point": {
            "bit_threshold": audit["bit_threshold"],
            "region_min_retention": audit["region_min_retention"],
            "logit_mean_min": audit["logit_mean_min"],
        },
        "conditions": condition_summaries,
        "threshold_grid": grid,
    }
    (args.experiment_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.experiment_dir / "audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_contact_sheet(args.experiment_dir, signed["rows"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
