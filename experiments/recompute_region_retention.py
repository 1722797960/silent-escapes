#!/usr/bin/env python3
"""Reclassify a saved defense audit using original-region retention.

This is a deterministic, CPU-only post-processing step. It preserves the
original audit JSON and reuses its saved bit accuracies and regional logits.
"""

import argparse
import base64
import json
from collections import Counter
from pathlib import Path


def unpack_bitmap(bitmap64):
    raw = base64.b64decode(bitmap64)
    bits = []
    for value in raw:
        bits.extend((value >> shift) & 1 for shift in range(7, -1, -1))
    if len(bits) != 64 * 64:
        raise ValueError(f"expected 4096 mask bits, got {len(bits)}")
    return [bits[i:i + 64] for i in range(0, len(bits), 64)]


def region_metrics(bitmap64, crop_frac):
    grid = unpack_bitmap(bitmap64)
    lo = int(round(crop_frac * 64))
    hi = int(round((1.0 - crop_frac) * 64))
    original = sum(sum(row) for row in grid)
    kept = sum(sum(row[lo:hi]) for row in grid[lo:hi]) if hi > lo else 0
    crop_area = max((hi - lo) ** 2, 1)
    occupancy = kept / crop_area
    retention = kept / original if original else 0.0
    return occupancy, retention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-dir", required=True)
    parser.add_argument("--input-json", default="audit.json")
    parser.add_argument("--output-json", default="audit_retention.json")
    parser.add_argument("--region-min-retention", type=float, default=0.05)
    parser.add_argument("--logit-mean-min", type=float, default=0.5)
    args = parser.parse_args()

    root = Path(args.e2e_dir)
    source = json.loads((root / args.input_json).read_text(encoding="utf-8"))
    meta_rows = json.loads(
        (root / "embed_meta.json").read_text(encoding="utf-8"))["images"]
    metadata = {row["input_image"]: row for row in meta_rows}

    rows = []
    for old in source["rows"]:
        row = dict(old)
        occupancy, retention = region_metrics(
            metadata[row["image"]]["bitmap64"], row["crop_frac"])

        row["region_occupancy"] = round(occupancy, 5)
        row["region_retention"] = round(retention, 5)
        row["region_in_frame"] = round(retention, 5)

        if row["bit_accuracy"] >= source["threshold"]:
            verdict = "INTEGRITY_CLASH"
        elif not (row["region_assertion_present"] and row["payload_hash_ok"]):
            verdict = "ASSERTION_MISSING"
        elif retention < args.region_min_retention:
            verdict = "TAMPERED_REGION"
        elif row["logit_in_region"] < args.logit_mean_min:
            verdict = "WATERMARK_SUPPRESSED"
        else:
            verdict = "SILENT_OTHER"
        row["audit_v2_region_aware"] = verdict
        rows.append(row)

    output = {
        "source_audit": args.input_json,
        "threshold": source["threshold"],
        "region_metric": "retained_mask_pixels/original_mask_pixels",
        "region_min_retention": args.region_min_retention,
        "logit_mean_min": args.logit_mean_min,
        "rows": rows,
    }
    destination = root / args.output_json
    destination.write_text(json.dumps(output, indent=1), encoding="utf-8")

    silent = [r for r in rows if r["audit_v1_deployed"] == "SILENT_ESCAPE"]
    counts = Counter(r["audit_v2_region_aware"] for r in silent)
    flagged = counts["TAMPERED_REGION"] + counts["WATERMARK_SUPPRESSED"]
    print(f"wrote {destination}")
    print(f"silent escapes: {len(silent)}")
    print(f"flagged: {flagged}/{len(silent)} = {flagged / len(silent):.3%}")
    print(dict(counts))


if __name__ == "__main__":
    main()
