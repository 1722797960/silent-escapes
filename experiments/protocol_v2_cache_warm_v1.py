"""Populate the exact protocol-v2 pixel cache for the full evaluation matrix.

The cache key is derived from decoded RGB pixels, so C2PA signing and policy
changes never require another WAM forward pass for an unchanged image.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from protocol_v2_cache import PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_cache_inventory_v1 import dataset_paths
from protocol_v2_pilot_v1 import Detector, atomic_json, load_jsonl


ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_VERSION = "wam-default-transform-v1"


def collect_records(
    e2e_root: Path,
    *,
    c1_rho: float,
    expected_per_dataset: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    ordinal = 0
    for dataset, paths in dataset_paths(e2e_root, c1_rho=c1_rho).items():
        if len(paths) != expected_per_dataset:
            raise RuntimeError(
                f"{dataset} expected {expected_per_dataset} images, observed {len(paths)}"
            )
        for dataset_index, path in enumerate(paths):
            resolved = path.resolve(strict=True)
            key, image = decoded_rgb_key(resolved)
            if key in seen:
                raise RuntimeError(
                    f"full matrix requires unique decoded pixels; duplicate {key}: "
                    f"{seen[key]} and {resolved}"
                )
            seen[key] = resolved
            records.append(
                {
                    "ordinal": ordinal,
                    "dataset": dataset,
                    "dataset_index": dataset_index,
                    "source": resolved,
                    "source_relpath": str(resolved.relative_to(e2e_root)),
                    "pixel_key": key,
                    "image": image,
                }
            )
            ordinal += 1
    return records


def validate_entry(
    entry: dict[str, Any],
    *,
    key: str,
    image: dict[str, Any],
    model_sha256: str,
    checkpoint_sha256: str,
) -> None:
    if entry["key"] != key:
        raise RuntimeError(f"cache key mismatch for {key}")
    if entry["image"] != image:
        raise RuntimeError(f"cache image metadata mismatch for {key}")
    if entry["model_sha256"] != model_sha256:
        raise RuntimeError(f"cache params hash mismatch for {key}")
    if entry["checkpoint_sha256"] != checkpoint_sha256:
        raise RuntimeError(f"cache checkpoint hash mismatch for {key}")
    if entry["preprocessing_version"] != PREPROCESSING_VERSION:
        raise RuntimeError(f"cache preprocessing version mismatch for {key}")
    if not entry["decoded_bits"]:
        raise RuntimeError(f"empty decoded bit string for {key}")
    if entry["mask_logits"].ndim != 2:
        raise RuntimeError(f"non-2D mask logits for {key}")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-root", type=Path, default=ROOT)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wam-repo", type=Path, required=True)
    parser.add_argument("--params-json", type=Path, default=ROOT / "ckpts/params.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "ckpts/wam_mit.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expected-per-dataset", type=int, default=200)
    parser.add_argument("--c1-rho", type=float, default=0.4)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")

    started = time.time()
    e2e_root = args.e2e_root.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / f"rows.part-{args.shard_id:03d}.jsonl"
    if rows_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {rows_path}; use --resume")
    completed = {
        row["pixel_key"] for row in load_jsonl(rows_path)
    } if rows_path.exists() else set()

    records = collect_records(
        e2e_root,
        c1_rho=args.c1_rho,
        expected_per_dataset=args.expected_per_dataset,
    )
    shard = [row for row in records if row["ordinal"] % args.num_shards == args.shard_id]
    if args.limit is not None:
        shard = shard[: args.limit]

    base_cache = PixelCache(args.base_cache.resolve(strict=True))
    full_cache = PixelCache(args.cache_dir.resolve())
    model_hash = sha256_file(args.params_json.resolve(strict=True))
    checkpoint_hash = sha256_file(args.checkpoint.resolve(strict=True))
    detector: Detector | None = None

    for position, record in enumerate(shard, start=1):
        key = record["pixel_key"]
        if key in completed:
            continue
        cached = full_cache.get(key)
        if cached is not None:
            status = "full-cache-hit"
        else:
            cached = base_cache.get(key)
            if cached is not None:
                validate_entry(
                    cached,
                    key=key,
                    image=record["image"],
                    model_sha256=model_hash,
                    checkpoint_sha256=checkpoint_hash,
                )
                full_cache.put(
                    key=key,
                    decoded_bits=cached["decoded_bits"],
                    mask_logits=cached["mask_logits"],
                    image=record["image"],
                    model_sha256=model_hash,
                    checkpoint_sha256=checkpoint_hash,
                    preprocessing_version=PREPROCESSING_VERSION,
                )
                status = "copied-base-cache"
            else:
                if detector is None:
                    detector = Detector(
                        wam_repo=args.wam_repo.resolve(strict=True),
                        params=args.params_json.resolve(strict=True),
                        checkpoint=args.checkpoint.resolve(strict=True),
                        device=args.device,
                    )
                decoded_bits, mask_logits = detector.run(record["source"])
                full_cache.put(
                    key=key,
                    decoded_bits=decoded_bits,
                    mask_logits=mask_logits,
                    image=record["image"],
                    model_sha256=model_hash,
                    checkpoint_sha256=checkpoint_hash,
                    preprocessing_version=PREPROCESSING_VERSION,
                )
                status = "gpu-inference"

        verified = full_cache.get(key)
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
                "dataset": record["dataset"],
                "dataset_index": record["dataset_index"],
                "ordinal": record["ordinal"],
                "pixel_key": key,
                "source_relpath": record["source_relpath"],
                "status": status,
            },
        )
        print(
            f"[{position}/{len(shard)}] {record['dataset']} "
            f"{record['dataset_index']:03d}: {status}",
            flush=True,
        )

    rows = load_jsonl(rows_path) if rows_path.exists() else []
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    summary = {
        "schema": "protocol-v2-cache-warm-summary-1",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "requested_records": len(shard),
        "completed_records": len({row["pixel_key"] for row in rows}),
        "status_counts": status_counts,
        "model_sha256": model_hash,
        "checkpoint_sha256": checkpoint_hash,
        "preprocessing_version": PREPROCESSING_VERSION,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output / f"summary.part-{args.shard_id:03d}.json", summary)


if __name__ == "__main__":
    main()
