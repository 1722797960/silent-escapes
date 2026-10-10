"""CPU inventory of exact PixArt decoded-pixel caches; no signing or inference.

Legacy audit.json supplies frozen decoder counts ONLY. Its old occupancy-based
region metric is intentionally excluded. The source hash and this limitation
are recorded for a subsequent protocol-v2 migration.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image
from protocol_v2_cache import PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_cache_warm_v1 import validate_entry


ROOT = Path(__file__).resolve().parents[1]


def inventory_context(defense: Path, cache_dir: Path) -> dict:
    audit_file = defense / "audit.json"
    meta_file = defense / "embed_meta.json"
    audit = json.loads(audit_file.read_text())
    metadata = json.loads(meta_file.read_text())["images"]
    meta_by_name = {row["input_image"]: row for row in metadata}
    levels = {round(i * .05, 2): 200 for i in range(9)}
    counts = Counter(round(row["crop_frac"], 2) for row in audit["rows"])
    if len(meta_by_name) != 200 or counts != Counter(levels):
        raise RuntimeError("expected 200 PixArt images and 1,800 per-side crop rows")
    model_hash = sha256_file(ROOT / "ckpts/params.json")
    checkpoint_hash = sha256_file(ROOT / "ckpts/wam_mit.pth")
    return dict(defense=defense, audit_file=audit_file, meta_file=meta_file,
                audit=audit, metadata=meta_by_name, levels=levels,
                model_hash=model_hash, checkpoint_hash=checkpoint_hash,
                cache=PixelCache(cache_dir))


def frozen_reference(old: dict, meta: dict) -> tuple[str, int]:
    accuracy = float(old["bit_accuracy"])
    count = round(accuracy * 32)
    if not math.isclose(accuracy, count / 32, abs_tol=5.1e-5, rel_tol=0):
        raise RuntimeError("legacy accuracy is not a 32-bit count")
    reference = "".join(str(int(bit)) for bit in meta["message_bits"])
    if len(reference) != 32 or set(reference) - {"0", "1"}:
        raise RuntimeError("invalid frozen message")
    return reference, count


def cached_matches(context: dict, key: str, image: dict, reference: str) -> int | None:
    entry = context["cache"].get(key)
    if entry is None:
        return None
    validate_entry(entry, key=key, image=image, model_sha256=context["model_hash"],
                   checkpoint_sha256=context["checkpoint_hash"])
    if len(entry["decoded_bits"]) != 32:
        raise RuntimeError("invalid cache bit length")
    return sum(a == b for a, b in zip(reference, entry["decoded_bits"]))


def inventory_record(context: dict, old: dict) -> dict:
    defense = context["defense"]
    meta = context["metadata"][old["image"]]
    crop = round(old["crop_frac"], 2)
    path = defense / "attack_resign" / f"crop{round(crop * 100):02d}" / meta["signed_image"]
    key, image = decoded_rgb_key(path)
    reference, count = frozen_reference(old, meta)
    matches = cached_matches(context, key, image, reference)
    if matches is not None and matches != count:
        raise RuntimeError(f"cache/legacy match-count difference: {old['image']}: {matches} != {count}")
    with Image.open(defense / "watermarked_unsigned" / meta["unsigned_image"]) as parent:
        original_size = [parent.width, parent.height]
    if original_size[0] != original_size[1]:
        raise RuntimeError("legacy PixArt crop requires square input")
    margin = round(original_size[0] * crop)
    return {"attack_id": f"pixart_per_side_c{round(crop * 100):02d}__{int(meta['index']):04d}",
            "image": old["image"], "image_index": int(meta["index"]),
            "crop_frac": crop, "pixel_key": key, "image_size": image,
            "source_relpath": str(path.relative_to(ROOT)), "source_size_px": original_size,
            "crop_box_px": [margin, margin, original_size[0]-margin, original_size[1]-margin],
            "cache_hit": matches is not None, "cached_match_count": matches,
            "legacy_match_count": count}


def inventory_rows(context: dict) -> list[dict]:
    rows = []
    seen = set()
    for index, old in enumerate(context["audit"]["rows"], 1):
        identity = (old["image"], round(old["crop_frac"], 2))
        if identity in seen:
            raise RuntimeError(f"duplicate image/crop: {identity}")
        seen.add(identity)
        rows.append(inventory_record(context, old))
        if index % 200 == 0:
            print(f"PixArt CPU cache inventory {index}/1800", flush=True)
    return rows


def inventory_summary(context: dict, rows: list[dict]) -> dict:
    unique = len({row["pixel_key"] for row in rows})
    hits = sum(row["cache_hit"] for row in rows)
    escapes = sum(row["legacy_match_count"] < 24 for row in rows)
    if unique != 1800 or escapes != 940:
        raise RuntimeError(f"unexpected unique pixels/escapes: {unique}/{escapes}")
    per_crop = {}
    for crop in sorted(context["levels"]):
        cohort = [row for row in rows if row["crop_frac"] == crop]
        cohort_hits = sum(row["cache_hit"] for row in cohort)
        per_crop[f"{crop:.2f}"] = {"rows": len(cohort), "cache_hits": cohort_hits,
                                    "cache_misses": len(cohort)-cohort_hits,
                                    "legacy_escapes": sum(row["legacy_match_count"] < 24 for row in cohort)}
    return {"schema": "protocol-v2-pixart-cache-inventory-1", "rows": len(rows),
               "unique_pixels": unique, "cache_hits": hits, "cache_misses": len(rows)-hits,
               "cache_hit_match_count_checks": hits, "legacy_escapes": escapes,
               "gpu_inference_calls": 0, "signing_calls": 0,
               "legacy_region_metric_consumed": False,
               "server_retention_audit_exists": (context["defense"] / "audit_retention.json").is_file(),
               "audit_sha256": sha256_file(context["audit_file"]),
               "embed_meta_sha256": sha256_file(context["meta_file"]),
               "script_sha256": sha256_file(Path(__file__)), "params_sha256": context["model_hash"],
               "checkpoint_sha256": context["checkpoint_hash"], "per_crop": per_crop}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--defense-dir", type=Path, default=ROOT / "results/defense_e2e_pixart")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results/protocol_v2/full_pixel_cache_v1/cache")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/protocol_v2/pixart_inventory_v1")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    context = inventory_context(args.defense_dir.resolve(strict=True), args.cache_dir)
    rows = inventory_rows(context)
    payload = inventory_summary(context, rows)
    for name, value in (("summary.json", payload), ("rows.json", rows)):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    (output / "report.md").write_text(
        "# PixArt v2 migration inventory\n\n"
        f"Rows / unique decoded pixels: {len(rows)} / {payload['unique_pixels']}.\n\n"
        f"Exact cache hits: {payload['cache_hits']}; missing entries: {payload['cache_misses']}.\n\n"
        f"Frozen integer decoder escapes: {payload['legacy_escapes']}.\n\n"
        "No GPU inference, signing, or regional-evidence analysis was performed.\n"
        "Legacy audit.json contributes decoder counts only; its region occupancy is not retention.\n"
        "The local derived retention audit must not be mistaken for a server detector run.\n"
        "Missing cache entries need a boundary pilot before full warming.\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
