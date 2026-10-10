"""Cache WAM evidence for the paper's nine-level per-side crop curve.

This is deliberately separate from ``protocol_v2_main_curve_cache_v1``.  The
v1 runner targets C1's removed-area curve; this runner targets the headline
``defense_e2e`` curve whose crop parameter is the margin removed from *each*
side (c = 0.00, 0.05, ..., 0.40).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from protocol_v2_cache import PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_cache_warm_v1 import PREPROCESSING_VERSION, validate_entry
from protocol_v2_main_curve_cache_v1 import warm_shard
from protocol_v2_pilot_v1 import (
    atomic_json,
    load_json,
    load_jsonl,
    logit_mean,
    make_region_assertion,
    mapped_mask_and_retention,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CROPS = tuple(round(index * 0.05, 2) for index in range(9))


def collect_records(defense_dir: Path, *, expected_images: int) -> list[dict[str, Any]]:
    metadata = load_json(defense_dir / "embed_meta.json")["images"]
    meta_by_image = {row["input_image"]: row for row in metadata}
    index_by_image = {row["input_image"]: int(row["index"]) for row in metadata}
    audit_rows = load_json(defense_dir / "audit_retention.json")["rows"]
    audit_rows.sort(key=lambda row: (index_by_image[row["image"]], float(row["crop_frac"])))

    expected_count = expected_images * len(EXPECTED_CROPS)
    if len(metadata) != expected_images or len(audit_rows) != expected_count:
        raise RuntimeError(
            f"expected {expected_images} images/{expected_count} rows, observed "
            f"{len(metadata)}/{len(audit_rows)}"
        )
    counts = Counter(round(float(row["crop_frac"]), 2) for row in audit_rows)
    if counts != Counter({crop: expected_images for crop in EXPECTED_CROPS}):
        raise RuntimeError(f"unexpected per-side crop counts: {dict(sorted(counts.items()))}")

    root = defense_dir.parents[1]
    records: list[dict[str, Any]] = []
    seen_keys: dict[str, str] = {}
    for ordinal, row in enumerate(audit_rows):
        image_name = row["image"]
        meta = meta_by_image[image_name]
        image_index = index_by_image[image_name]
        crop = round(float(row["crop_frac"]), 2)
        source = (
            defense_dir
            / "attack_resign"
            / f"crop{round(crop * 100):02d}"
            / meta["signed_image"]
        ).resolve(strict=True)
        key, image = decoded_rgb_key(source)
        attack_id = f"sdxl_per_side_c{round(crop * 100):02d}__{image_index:04d}"
        if key in seen_keys:
            raise RuntimeError(f"duplicate decoded pixels: {attack_id} and {seen_keys[key]}")
        seen_keys[key] = attack_id
        with Image.open(defense_dir / "watermarked_unsigned" / meta["unsigned_image"]) as parent:
            original_size = [parent.width, parent.height]
        margin = int(round(original_size[0] * crop))
        crop_box = [margin, margin, original_size[0] - margin, original_size[1] - margin]
        records.append(
            {
                "ordinal": ordinal,
                "attack_id": attack_id,
                "image_index": image_index,
                # Kept for compatibility with the shared cache warmer.
                "rho_requested": crop,
                "crop_frac": crop,
                "source": source,
                "source_relpath": str(source.relative_to(root)),
                "pixel_key": key,
                "image": image,
                "original_size": original_size,
                "crop_box_px": crop_box,
                "legacy": row,
                "meta": meta,
            }
        )
    return records


def inventory(records: list[dict[str, Any]], cache: PixelCache) -> dict[str, Any]:
    per_crop: dict[str, dict[str, int]] = {}
    hits = 0
    for record in records:
        label = f"{record['crop_frac']:.2f}"
        bucket = per_crop.setdefault(
            label, {"rows": 0, "cache_hits": 0, "cache_misses": 0}
        )
        bucket["rows"] += 1
        if cache.contains(record["pixel_key"]):
            bucket["cache_hits"] += 1
            hits += 1
        else:
            bucket["cache_misses"] += 1
    return {
        "schema": "protocol-v2-per-side-main-curve-cache-inventory-2",
        "parameter": "per-side crop fraction c",
        "row_count": len(records),
        "unique_pixel_count": len({record["pixel_key"] for record in records}),
        "cache_hits": hits,
        "cache_misses": len(records) - hits,
        "gpu_inference_required": hits != len(records),
        "per_crop": per_crop,
    }


def finalize(
    records: list[dict[str, Any]],
    *,
    cache: PixelCache,
    output: Path,
    params_json: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    provenance_rows: list[dict[str, Any]] = []
    for path in sorted(output.glob("rows.part-*.jsonl")):
        provenance_rows.extend(load_jsonl(path))
    by_key = {row["pixel_key"]: row for row in provenance_rows}
    if len(provenance_rows) != len(records) or len(by_key) != len(records):
        raise RuntimeError(
            f"expected {len(records)} unique provenance rows, observed "
            f"{len(provenance_rows)}/{len(by_key)}"
        )

    model_hash = sha256_file(params_json)
    checkpoint_hash = sha256_file(checkpoint)
    bit_matches = retention_matches = suppression_matches = 0
    max_abs_logit_delta = 0.0
    comparison_rows: list[dict[str, Any]] = []
    per_crop: dict[str, dict[str, int]] = {}
    ordered_provenance: list[dict[str, Any]] = []
    for record in records:
        provenance = by_key[record["pixel_key"]]
        if provenance["attack_id"] != record["attack_id"]:
            raise RuntimeError(f"provenance mismatch: {record['attack_id']}")
        cached = cache.get(record["pixel_key"])
        if cached is None:
            raise RuntimeError(f"missing cache entry: {record['attack_id']}")
        validate_entry(
            cached,
            key=record["pixel_key"],
            image=record["image"],
            model_sha256=model_hash,
            checkpoint_sha256=checkpoint_hash,
        )

        old = record["legacy"]
        reference = "".join(str(int(bit)) for bit in record["meta"]["message_bits"])
        match_count = sum(a == b for a, b in zip(cached["decoded_bits"], reference))
        legacy_match_count = int(round(float(old["bit_accuracy"]) * len(reference)))
        bit_matches += int(match_count == legacy_match_count)

        crop = record["crop_frac"]
        geometry = {"status": "missing"} if crop == 0 else {
            "status": "valid",
            "transform": {
                "source_size_px": record["original_size"],
                "crop_box_px": record["crop_box_px"],
                "output_size_px": [record["image"]["width"], record["image"]["height"]],
            },
        }
        mapped, retention = mapped_mask_and_retention(
            make_region_assertion(record["meta"]), geometry
        )
        retention_matches += int(abs(retention - float(old["region_retention"])) <= 5.1e-6)
        regional_logit = logit_mean(cached["mask_logits"], mapped)
        old_logit = float(old["logit_in_region"])
        max_abs_logit_delta = max(max_abs_logit_delta, abs(regional_logit - old_logit))
        suppression_matches += int((regional_logit < 0.5) == (old_logit < 0.5))
        if not (math.isfinite(retention) and 0 <= retention <= 1
                and math.isfinite(regional_logit) and 0 <= regional_logit <= 1):
            raise RuntimeError(f"invalid v2 evidence values: {record['attack_id']}")
        comparison_rows.append({
            "attack_id": record["attack_id"],
            "crop_frac": crop,
            "legacy_retention": float(old["region_retention"]),
            "v2_retention": retention,
            "legacy_regional_logit": old_logit,
            "v2_regional_logit": regional_logit,
            "retention_class_changed": (retention < 0.05) != (float(old["region_retention"]) < 0.05),
            "suppression_class_changed": (regional_logit < 0.5) != (old_logit < 0.5),
        })
        bucket = per_crop.setdefault(
            f"{crop:.2f}",
            {"rows": 0, "legacy_bit_count_matches": 0, "suppression_class_matches": 0},
        )
        bucket["rows"] += 1
        bucket["legacy_bit_count_matches"] += int(match_count == legacy_match_count)
        bucket["suppression_class_matches"] += int(
            (regional_logit < 0.5) == (old_logit < 0.5)
        )
        ordered_provenance.append(provenance)

    checks = {
        "cache_entries": len(ordered_provenance) == len(records),
        "legacy_bit_count_matches": bit_matches == len(records),
        "finite_bounded_evidence": len(comparison_rows) == len(records),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"per-side cache acceptance failed: {failed}")

    merged = output / "rows.jsonl"
    with merged.with_suffix(".tmp").open("w", encoding="utf-8") as handle:
        for row in ordered_provenance:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    merged.with_suffix(".tmp").replace(merged)
    summary = {
        "schema": "protocol-v2-per-side-main-curve-cache-final-summary-2",
        "row_count": len(records),
        "unique_pixel_count": len({record["pixel_key"] for record in records}),
        "status_counts": dict(sorted(Counter(row["status"] for row in provenance_rows).items())),
        "legacy_bit_count_matches": bit_matches,
        "legacy_retention_matches": retention_matches,
        "suppression_class_matches": suppression_matches,
        "max_abs_logit_delta": max_abs_logit_delta,
        "retention_mapping": "source-pixel-half-open; floor lower/ceil upper mask-cell bounds",
        "legacy_geometry_comparison_is_diagnostic": True,
        "retention_class_changes": sum(row["retention_class_changed"] for row in comparison_rows),
        "suppression_class_changes": sum(row["suppression_class_changed"] for row in comparison_rows),
        "per_crop": per_crop,
        "model_sha256": model_hash,
        "checkpoint_sha256": checkpoint_hash,
        "preprocessing_version": PREPROCESSING_VERSION,
        "acceptance": checks,
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "geometry_comparison.json", {"rows": comparison_rows})
    (output / "report.md").write_text(
        "\n".join(
            [
                "# Protocol v2 per-side main-curve pixel cache",
                "",
                f"Rows and unique decoded pixels: {len(records)}.",
                f"Legacy integer bit-count matches: {bit_matches}/{len(records)}.",
                f"Legacy region-retention matches: {retention_matches}/{len(records)}.",
                f"Suppression-class matches: {suppression_matches}/{len(records)}.",
                f"Maximum absolute regional-logit delta: {max_abs_logit_delta:.8g}.",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defense-dir", type=Path, default=ROOT / "results/defense_e2e")
    parser.add_argument(
        "--cache-dir", type=Path,
        default=ROOT / "results/protocol_v2/full_pixel_cache_v1/cache",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results/protocol_v2/main_curve_cache_v2",
    )
    parser.add_argument("--wam-repo", type=Path)
    parser.add_argument("--params-json", type=Path, default=ROOT / "ckpts/params.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "ckpts/wam_mit.pth")
    parser.add_argument("--expected-images", type=int, default=200)
    parser.add_argument("--phase", choices=("inventory", "warm", "finalize"), default="inventory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")

    defense = args.defense_dir.resolve(strict=True)
    cache = PixelCache(args.cache_dir.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = collect_records(defense, expected_images=args.expected_images)
    if args.phase == "inventory":
        atomic_json(output / "inventory.json", inventory(records, cache))
        return
    if args.phase == "finalize":
        finalize(
            records,
            cache=cache,
            output=output,
            params_json=args.params_json.resolve(strict=True),
            checkpoint=args.checkpoint.resolve(strict=True),
        )
        return
    if args.wam_repo is None:
        parser.error("--wam-repo is required for --phase warm")
    warm_shard(
        records,
        cache=cache,
        output=output,
        params_json=args.params_json.resolve(strict=True),
        checkpoint=args.checkpoint.resolve(strict=True),
        wam_repo=args.wam_repo.resolve(strict=True),
        device=args.device,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
        limit=args.limit,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
