"""Create machine-readable and visual summaries for inpainting_e2e_v1."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

from PIL import Image, ImageDraw

from inpainting_common_v1 import wilson_interval


def rate_record(successes: int, total: int) -> dict:
    low, high = wilson_interval(successes, total)
    return {
        "count": successes,
        "total": total,
        "rate": successes / total if total else 0.0,
        "wilson95": [low, high],
    }


def make_contact_sheet(
    root: Path, experiment_dir: Path, rows: list[dict], output: Path
) -> None:
    tile = 300
    header = 55
    canvas = Image.new("RGB", (tile * 3, (tile + header) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(rows):
        source = root / "results/defense_e2e/watermarked_signed_ra" / row["signed_source"]
        mask = root / "data/masks_200" / row["mask_image"]
        result = experiment_dir / "inpainted_unsigned" / row["unsigned_inpainted"]
        source_image = Image.open(source).convert("RGB")
        mask_image = Image.open(mask).convert("L")
        overlay = source_image.copy()
        red = Image.new("RGB", source_image.size, (255, 32, 32))
        overlay.paste(red, mask=mask_image.point(lambda value: 110 if value > 127 else 0))
        triplet = [source_image, overlay, Image.open(result).convert("RGB")]
        y = row_index * (tile + header)
        for column, image in enumerate(triplet):
            thumb = image.copy()
            thumb.thumbnail((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (column * tile, y))
        draw.text(
            (8, y + tile + 5),
            f"{row['input_image'][:62]}  coverage={row['mask_coverage']:.3f}",
            fill="black",
        )
        draw.text((tile * 2 + 8, y + tile + 25), "LaMa exact-mask output", fill="black")
    canvas.save(output)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=root / "results/inpainting_e2e_v1")
    args = parser.parse_args()
    audit = json.loads((args.experiment_dir / "audit.json").read_text(encoding="utf-8"))
    signed = json.loads((args.experiment_dir / "signed_meta.json").read_text(encoding="utf-8"))
    rows = audit["rows"]
    total = len(rows)
    escapes = [row for row in rows if row["baseline_audit"] == "SILENT_ESCAPE"]
    recovered = [row for row in escapes if row["region_aware_audit"] != "SILENT_OTHER"]
    suppressed = [row for row in escapes if row["region_aware_audit"] == "WATERMARK_SUPPRESSED"]
    silent_other = [row for row in escapes if row["region_aware_audit"] == "SILENT_OTHER"]
    full_attack_flags = [row for row in rows if row["region_aware_audit"] != "SILENT_OTHER"]

    summary = {
        "experiment": audit["experiment"],
        "n": total,
        "invariants": {
            "manifest_valid": sum(row["manifest_valid"] for row in rows),
            "region_assertion_present": sum(row["region_assertion_present"] for row in rows),
            "payload_hash_ok": sum(row["payload_hash_ok"] for row in rows),
            "geometry_unchanged": sum(row["geometry_unchanged"] for row in rows),
            "outside_pixels_byte_identical": sum(row["outside_diff_max"] == 0 for row in rows),
        },
        "baseline_silent_escape": rate_record(len(escapes), total),
        "region_aware_recovery_among_escapes": rate_record(len(recovered), len(escapes)),
        "watermark_suppressed_among_escapes": rate_record(len(suppressed), len(escapes)),
        "silent_other_among_escapes": rate_record(len(silent_other), len(escapes)),
        "overall_attack_flag": rate_record(len(full_attack_flags), total),
        "median_bit_accuracy_before": statistics.median(row["bit_accuracy_before"] for row in rows),
        "median_bit_accuracy_after": statistics.median(row["bit_accuracy_after"] for row in rows),
        "median_logit_before": statistics.median(row["logit_in_region_before"] for row in rows),
        "median_logit_after": statistics.median(row["logit_in_region_after"] for row in rows),
    }
    (args.experiment_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.experiment_dir / "audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    make_contact_sheet(
        root,
        args.experiment_dir,
        signed["images"],
        args.experiment_dir / "contact_sheet.png",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
