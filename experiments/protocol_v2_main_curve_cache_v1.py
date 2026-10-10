"""Inventory and populate protocol-v2 pixel evidence for the SDXL main curve.

The source set is the 200-image, nine-level centered-crop slice of C1.  Cache
keys are computed from decoded RGB pixels, so the resulting detector outputs
can be reused after C2PA signing without another WAM forward pass.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from protocol_v2_cache import PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_cache_warm_v1 import (
    PREPROCESSING_VERSION,
    append_jsonl,
    validate_entry,
)
from protocol_v2_pilot_v1 import (
    Detector,
    atomic_json,
    load_json,
    load_jsonl,
    logit_mean,
    make_region_assertion,
    mapped_mask_and_retention,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RHOS = tuple(round(index / 10, 1) for index in range(9))


def centered_curve_rows(c1_dir: Path) -> list[dict[str, Any]]:
    rows = [
        row
        for row in load_jsonl(c1_dir / "attack_manifest.jsonl")
        if row["method"] == "center"
    ]
    rows.sort(key=lambda row: (int(row["image_index"]), float(row["rho_requested"])))
    return rows


def collect_records(
    c1_dir: Path,
    *,
    expected_images: int,
) -> list[dict[str, Any]]:
    rows = centered_curve_rows(c1_dir)
    expected_count = expected_images * len(EXPECTED_RHOS)
    if len(rows) != expected_count:
        raise RuntimeError(f"expected {expected_count} centered-curve rows, observed {len(rows)}")
    counts = Counter(float(row["rho_requested"]) for row in rows)
    expected_counts = {rho: expected_images for rho in EXPECTED_RHOS}
    if dict(sorted(counts.items())) != expected_counts:
        raise RuntimeError(f"unexpected centered-curve rho counts: {dict(sorted(counts.items()))}")

    records: list[dict[str, Any]] = []
    seen_keys: dict[str, str] = {}
    for ordinal, row in enumerate(rows):
        source = (c1_dir / row["output_relpath"]).resolve(strict=True)
        key, image = decoded_rgb_key(source)
        if key in seen_keys:
            raise RuntimeError(
                f"centered curve requires unique pixels: {row['attack_id']} duplicates "
                f"{seen_keys[key]}"
            )
        seen_keys[key] = row["attack_id"]
        records.append(
            {
                "ordinal": ordinal,
                "attack_id": row["attack_id"],
                "image_index": int(row["image_index"]),
                "rho_requested": float(row["rho_requested"]),
                "source": source,
                "source_relpath": str(source.relative_to(c1_dir.parents[2])),
                "pixel_key": key,
                "image": image,
            }
        )
    return records


def inventory(records: list[dict[str, Any]], cache: PixelCache) -> dict[str, Any]:
    per_rho: dict[str, dict[str, int]] = {}
    hits = 0
    for record in records:
        rho = f"{record['rho_requested']:.1f}"
        bucket = per_rho.setdefault(rho, {"rows": 0, "cache_hits": 0, "cache_misses": 0})
        bucket["rows"] += 1
        if cache.contains(record["pixel_key"]):
            bucket["cache_hits"] += 1
            hits += 1
        else:
            bucket["cache_misses"] += 1
    return {
        "schema": "protocol-v2-main-curve-cache-inventory-1",
        "row_count": len(records),
        "unique_pixel_count": len({record["pixel_key"] for record in records}),
        "cache_hits": hits,
        "cache_misses": len(records) - hits,
        "gpu_inference_required": hits != len(records),
        "per_rho": per_rho,
    }


def warm_shard(
    records: list[dict[str, Any]],
    *,
    cache: PixelCache,
    output: Path,
    params_json: Path,
    checkpoint: Path,
    wam_repo: Path,
    device: str,
    shard_id: int,
    num_shards: int,
    limit: int | None,
    resume: bool,
) -> dict[str, Any]:
    rows_path = output / f"rows.part-{shard_id:03d}.jsonl"
    if rows_path.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite {rows_path}; pass --resume")
    completed = {
        row["pixel_key"] for row in load_jsonl(rows_path)
    } if rows_path.exists() else set()
    shard = [record for record in records if record["ordinal"] % num_shards == shard_id]
    if limit is not None:
        shard = shard[:limit]

    model_hash = sha256_file(params_json)
    checkpoint_hash = sha256_file(checkpoint)
    detector: Detector | None = None
    started = time.time()
    for position, record in enumerate(shard, start=1):
        key = record["pixel_key"]
        if key in completed:
            continue
        cached = cache.get(key)
        if cached is None:
            if detector is None:
                detector = Detector(
                    wam_repo=wam_repo,
                    params=params_json,
                    checkpoint=checkpoint,
                    device=device,
                )
            decoded_bits, mask_logits = detector.run(record["source"])
            cache.put(
                key=key,
                decoded_bits=decoded_bits,
                mask_logits=mask_logits,
                image=record["image"],
                model_sha256=model_hash,
                checkpoint_sha256=checkpoint_hash,
                preprocessing_version=PREPROCESSING_VERSION,
            )
            status = "gpu-inference"
        else:
            validate_entry(
                cached,
                key=key,
                image=record["image"],
                model_sha256=model_hash,
                checkpoint_sha256=checkpoint_hash,
            )
            status = "cache-hit"

        verified = cache.get(key)
        if verified is None:
            raise RuntimeError(f"cache write did not persist for {key}")
        validate_entry(
            verified,
            key=key,
            image=record["image"],
            model_sha256=model_hash,
            checkpoint_sha256=checkpoint_hash,
        )
        append_jsonl(
            rows_path,
            {
                "attack_id": record["attack_id"],
                "image_index": record["image_index"],
                "ordinal": record["ordinal"],
                "pixel_key": key,
                "rho_requested": record["rho_requested"],
                "source_relpath": record["source_relpath"],
                "status": status,
            },
        )
        completed.add(key)
        print(
            f"[{position}/{len(shard)}] {record['attack_id']}: {status}",
            flush=True,
        )

    rows = load_jsonl(rows_path) if rows_path.exists() else []
    status_counts = Counter(row["status"] for row in rows)
    summary = {
        "schema": "protocol-v2-main-curve-cache-warm-summary-1",
        "shard_id": shard_id,
        "num_shards": num_shards,
        "requested_records": len(shard),
        "completed_records": len({row["pixel_key"] for row in rows}),
        "status_counts": dict(sorted(status_counts.items())),
        "model_sha256": model_hash,
        "checkpoint_sha256": checkpoint_hash,
        "preprocessing_version": PREPROCESSING_VERSION,
        "device": device,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output / f"summary.part-{shard_id:03d}.json", summary)
    return summary


def finalize(
    records: list[dict[str, Any]],
    *,
    c1_dir: Path,
    cache: PixelCache,
    output: Path,
    params_json: Path,
    checkpoint: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output.glob("rows.part-*.jsonl")):
        rows.extend(load_jsonl(path))
    by_key = {row["pixel_key"]: row for row in rows}
    if len(rows) != len(records) or len(by_key) != len(records):
        raise RuntimeError(
            f"expected {len(records)} unique provenance rows, observed "
            f"{len(rows)} rows/{len(by_key)} keys"
        )

    root = c1_dir.parents[2]
    old_rows = {
        row["attack_id"]: row for row in load_jsonl(c1_dir / "audit_rows.jsonl")
    }
    metadata = load_json(root / "results/defense_e2e/embed_meta.json")["images"]
    meta_by_image = {row["input_image"]: row for row in metadata}
    model_hash = sha256_file(params_json)
    checkpoint_hash = sha256_file(checkpoint)
    bit_matches = 0
    retention_matches = 0
    suppression_matches = 0
    max_abs_logit_delta = 0.0
    per_rho: dict[str, dict[str, int]] = {}
    ordered_provenance: list[dict[str, Any]] = []

    for record in records:
        provenance = by_key[record["pixel_key"]]
        if provenance["attack_id"] != record["attack_id"]:
            raise RuntimeError(f"provenance attack mismatch for {record['attack_id']}")
        cached = cache.get(record["pixel_key"])
        if cached is None:
            raise RuntimeError(f"missing finalized cache entry: {record['attack_id']}")
        validate_entry(
            cached,
            key=record["pixel_key"],
            image=record["image"],
            model_sha256=model_hash,
            checkpoint_sha256=checkpoint_hash,
        )
        old = old_rows[record["attack_id"]]
        meta = meta_by_image[old["input_image"]]
        reference = "".join(str(int(bit)) for bit in meta["message_bits"])
        match_count = sum(
            decoded == expected
            for decoded, expected in zip(cached["decoded_bits"], reference)
        )
        legacy_match_count = int(round(float(old["bit_accuracy"]) * len(reference)))
        if match_count == legacy_match_count:
            bit_matches += 1

        mapped, retention = mapped_mask_and_retention(
            make_region_assertion(meta),
            {
                "status": "valid",
                "transform": {
                    "source_size_px": old["original_size"],
                    "crop_box_px": old["crop_box_px"],
                    "output_size_px": [record["image"]["width"], record["image"]["height"]],
                },
            },
        )
        if abs(retention - float(old["region_retention"])) <= 1e-8:
            retention_matches += 1
        regional_logit = logit_mean(cached["mask_logits"], mapped)
        old_logit = float(old["logit_in_region"])
        max_abs_logit_delta = max(max_abs_logit_delta, abs(regional_logit - old_logit))
        if (regional_logit < 0.5) == (old_logit < 0.5):
            suppression_matches += 1
        rho = f"{record['rho_requested']:.1f}"
        bucket = per_rho.setdefault(
            rho,
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
        "legacy_retention_matches": retention_matches == len(records),
        "suppression_class_matches": suppression_matches == len(records),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"main-curve cache acceptance failed: {failed}")
    merged = output / "rows.jsonl"
    temporary = merged.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in ordered_provenance:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(merged)
    summary = {
        "schema": "protocol-v2-main-curve-cache-final-summary-1",
        "row_count": len(records),
        "unique_pixel_count": len({record["pixel_key"] for record in records}),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "legacy_bit_count_matches": bit_matches,
        "legacy_retention_matches": retention_matches,
        "suppression_class_matches": suppression_matches,
        "max_abs_logit_delta": max_abs_logit_delta,
        "per_rho": per_rho,
        "model_sha256": model_hash,
        "checkpoint_sha256": checkpoint_hash,
        "preprocessing_version": PREPROCESSING_VERSION,
        "acceptance": checks,
    }
    atomic_json(output / "summary.json", summary)
    (output / "report.md").write_text(
        "\n".join(
            [
                "# Protocol v2 SDXL main-curve pixel cache",
                "",
                f"Rows and unique decoded pixels: {len(records)}.",
                f"Legacy integer bit-count matches: {bit_matches}/{len(records)}.",
                f"Legacy region-retention matches: {retention_matches}/{len(records)}.",
                f"Suppression-class matches: {suppression_matches}/{len(records)}.",
                f"Maximum absolute regional-logit delta: {max_abs_logit_delta:.8g}.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1-dir", type=Path, default=ROOT / "results/c1_v1/full")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "results/protocol_v2/full_pixel_cache_v1/cache",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/protocol_v2/main_curve_cache_v1",
    )
    parser.add_argument("--wam-repo", type=Path)
    parser.add_argument("--params-json", type=Path, default=ROOT / "ckpts/params.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "ckpts/wam_mit.pth")
    parser.add_argument("--expected-images", type=int, default=200)
    parser.add_argument(
        "--phase", choices=("inventory", "warm", "finalize"), default="inventory"
    )
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

    c1_dir = args.c1_dir.resolve(strict=True)
    cache = PixelCache(args.cache_dir.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = collect_records(c1_dir, expected_images=args.expected_images)
    if args.phase == "inventory":
        atomic_json(output / "inventory.json", inventory(records, cache))
        return
    if args.phase == "finalize":
        finalize(
            records,
            c1_dir=c1_dir,
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
