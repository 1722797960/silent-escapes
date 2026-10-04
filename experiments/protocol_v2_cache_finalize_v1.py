"""Merge and verify the three full pixel-cache warming shards."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from protocol_v2_cache import PixelCache, sha256_file
from protocol_v2_cache_warm_v1 import (
    PREPROCESSING_VERSION,
    collect_records,
    validate_entry,
)
from protocol_v2_pilot_v1 import atomic_json, load_jsonl


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-root", type=Path, default=ROOT)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--warm-output-dir", type=Path, required=True)
    parser.add_argument("--params-json", type=Path, default=ROOT / "ckpts/params.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "ckpts/wam_mit.pth")
    parser.add_argument("--expected-per-dataset", type=int, default=200)
    parser.add_argument("--c1-rho", type=float, default=0.4)
    args = parser.parse_args()

    e2e_root = args.e2e_root.resolve(strict=True)
    output = args.warm_output_dir.resolve(strict=True)
    cache = PixelCache(args.cache_dir.resolve(strict=True))
    model_hash = sha256_file(args.params_json.resolve(strict=True))
    checkpoint_hash = sha256_file(args.checkpoint.resolve(strict=True))
    expected = collect_records(
        e2e_root,
        c1_rho=args.c1_rho,
        expected_per_dataset=args.expected_per_dataset,
    )
    expected_by_key = {row["pixel_key"]: row for row in expected}

    rows: list[dict[str, Any]] = []
    for path in sorted(output.glob("rows.part-*.jsonl")):
        rows.extend(load_jsonl(path))
    unique = {row["pixel_key"]: row for row in rows}
    if len(unique) != len(rows):
        raise RuntimeError("duplicate pixel keys occur across warm-cache shard rows")
    if set(unique) != set(expected_by_key):
        missing = sorted(set(expected_by_key) - set(unique))
        extra = sorted(set(unique) - set(expected_by_key))
        raise RuntimeError(f"warm-cache coverage mismatch: missing={len(missing)}, extra={len(extra)}")

    status_counts: dict[str, int] = {}
    dataset_counts: dict[str, int] = {}
    for key in sorted(expected_by_key):
        record = expected_by_key[key]
        entry = cache.get(key)
        if entry is None:
            raise RuntimeError(f"cache entry missing after warm run: {key}")
        validate_entry(
            entry,
            key=key,
            image=record["image"],
            model_sha256=model_hash,
            checkpoint_sha256=checkpoint_hash,
        )
        row = unique[key]
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        dataset_counts[row["dataset"]] = dataset_counts.get(row["dataset"], 0) + 1

    merged = output / "rows.jsonl"
    temporary = merged.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(unique):
            handle.write(json.dumps(unique[key], ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, merged)
    summary = {
        "schema": "protocol-v2-cache-warm-final-summary-1",
        "expected_records": len(expected),
        "verified_records": len(unique),
        "dataset_counts": dataset_counts,
        "status_counts": status_counts,
        "model_sha256": model_hash,
        "checkpoint_sha256": checkpoint_hash,
        "preprocessing_version": PREPROCESSING_VERSION,
        "acceptance": {
            "complete_coverage": len(unique) == len(expected),
            "all_entries_readable": True,
            "model_and_checkpoint_match": True,
        },
    }
    atomic_json(output / "summary.json", summary)
    report = [
        "# Protocol v2 exact pixel-cache warm run",
        "",
        f"Verified decoded-RGB cache entries: {len(unique)}/{len(expected)}.",
        f"Entry sources: {json.dumps(status_counts, sort_keys=True)}.",
        "Every entry was re-opened and checked against the frozen params hash, ",
        "checkpoint hash, preprocessing version, image dimensions, bit payload, and 2D logits.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
