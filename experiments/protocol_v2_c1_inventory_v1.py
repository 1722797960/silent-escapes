"""CPU-only inventory of C1 pixels, frozen geometry and exact detector caches.

References/geometry here remain experiment-side inputs, not verifier claims.
Missing caches are listed, never populated; no signing or detector is called.
"""
from __future__ import annotations

import math
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from c1_attack_suite_v1 import VALID_METHODS, crop_box, mask_retention, stable_rng, unpack_mask
from protocol_v2_cache import PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_cache_warm_v1 import validate_entry
from protocol_v2_pilot_v1 import atomic_json, load_json, load_jsonl
from protocol_v2_pixart_inventory_v1 import frozen_reference


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/c1_v1/full"
DEFENSE = ROOT / "results/defense_e2e"
OUTPUT = ROOT / "results/protocol_v2/c1_inventory_v1"
CACHE = ROOT / "results/protocol_v2/full_pixel_cache_v1/cache"
EDGES = ("left", "right", "top", "bottom")


def expected_identities():
    return {f"{i:04d}|{m}|{rho:.3f}" for i in range(200)
            for m in VALID_METHODS for rho in (n / 10 for n in range(9))}


def contained_file(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise RuntimeError("input path escapes its declared dataset root")
    return path


def checked_pairs(attacks, audits):
    first = {r["attack_id"]: r for r in attacks}
    second = {r["attack_id"]: r for r in audits}
    expected = expected_identities()
    if len(attacks) != 10800 or len(audits) != 10800 or set(first) != expected or set(second) != expected:
        raise RuntimeError("expected complete unique 200 x 9 x 6 paired C1 records")
    for identity, attack in first.items():
        if any(second[identity].get(k) != value for k, value in attack.items()):
            raise RuntimeError("attack/audit metadata differs")
    return first, second


def checked_geometry(row, meta, config):
    width, height = row["original_size"]
    rho = row["rho_requested"]
    expected = crop_box(row["method"], rho, width, height,
                        stable_rng(config["random_seed"], meta["input_image"], rho))
    if list(expected) != row["crop_box_px"] or row["input_image"] != meta["input_image"]:
        raise RuntimeError("frozen crop geometry/seed differs")
    if row["signed_image"] != meta["signed_image"]:
        raise RuntimeError("signed source identity differs")
    left, top, right, bottom = expected
    actual = 1 - (right-left) * (bottom-top) / (width*height)
    retention = mask_retention(unpack_mask(meta["bitmap64"]), expected, width, height)
    if not (math.isclose(actual, row["rho_actual"], abs_tol=5.1e-9, rel_tol=0)
            and math.isclose(retention, row["mask_retention_geometry"], abs_tol=5.1e-9, rel_tol=0)):
        raise RuntimeError("recorded area or mask retention differs")
    return expected, retention


def cached_count(context, key, image, reference):
    if key not in context["cache_states"]:
        entry = context["cache"].get(key)
        if entry is not None:
            validate_entry(entry, key=key, image=image, model_sha256=context["model_hash"],
                           checkpoint_sha256=context["checkpoint_hash"])
            if len(entry["decoded_bits"]) != 32:
                raise RuntimeError("cached message is not 32 bits")
            context["cache_states"][key] = entry["decoded_bits"]
        else:
            context["cache_states"][key] = None
    bits = context["cache_states"][key]
    return None if bits is None else sum(a == b for a, b in zip(reference, bits))


def inventory_row(row, audit, meta, source_image, context):
    box, retention = checked_geometry(row, meta, context["config"])
    if tuple(row["original_size"]) != source_image.size:
        raise RuntimeError("original image dimensions differ")
    path = contained_file(SOURCE, row["output_relpath"])
    key, image = decoded_rgb_key(path)
    expected = source_image.crop(box).resize(source_image.size, Image.Resampling.LANCZOS)
    expected_key = hashlib.sha256(f"{expected.width}x{expected.height}:RGB:".encode()
                                  + expected.tobytes()).hexdigest()
    if key != expected_key:
        raise RuntimeError(f"saved pixels differ from frozen crop/resize: {row['attack_id']}")
    reference, legacy_count = frozen_reference(audit, meta)
    if bool(audit["detected"]) != (legacy_count >= 24):
        raise RuntimeError("legacy detector threshold is inconsistent")
    count = cached_count(context, key, image, reference)
    if count is not None and count != legacy_count:
        raise RuntimeError(f"cached/frozen integer count differs: {row['attack_id']}")
    return {**row, "pixel_key": key, "source_file_sha256": sha256_file(path),
            "image_size": image, "reference_bits": reference, "legacy_match_count": legacy_count,
            "cached_match_count": count, "cache_hit": count is not None,
            "region_retention_unrounded": retention,
            "parent_source_relpath": str(context["parent_path"].relative_to(ROOT)),
            "parent_file_sha256": context["parent_file_hash"], "parent_pixel_key": context["parent_key"]}


def grouped_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['method']}|{row['rho_requested']:.2f}"].append(row)
    return {name: {"records": len(group), "cache_hits": sum(r["cache_hit"] for r in group),
                   "legacy_escapes": sum(r["legacy_match_count"] < 24 for r in group)}
            for name, group in sorted(groups.items())}


def mask_aware_selection(rows):
    groups = defaultdict(dict)
    for row in rows:
        if row["method"] in EDGES:
            groups[(row["image_index"], row["rho_requested"])][row["method"]] = row
    selected = []
    for (image, rho), candidates in sorted(groups.items()):
        if set(candidates) != set(EDGES):
            raise RuntimeError("missing edge candidate")
        # Original suite tie order is left, right, top, bottom.
        chosen = min(EDGES, key=lambda m: candidates[m]["region_retention_unrounded"])
        row = candidates[chosen]
        selected.append({"image_index": image, "rho_requested": rho,
                         "selected_attack_id": row["attack_id"], "pixel_key": row["pixel_key"],
                         "cache_hit": row["cache_hit"], "tie_order": list(EDGES)})
    return selected


def inventory_context():
    attacks, audits = checked_pairs(load_jsonl(SOURCE / "attack_manifest.jsonl"),
                                    load_jsonl(SOURCE / "audit_rows.jsonl"))
    config = load_json(SOURCE / "config_effective.json")
    if (config["resize_filter"] != "LANCZOS" or config["watermark_threshold"] != .75
            or config["random_seed"] != 20260928 or config["effective_num_images"] != 200
            or config["effective_methods"] != list(VALID_METHODS)
            or config["effective_removed_area_fractions"] != [n/10 for n in range(9)]):
        raise RuntimeError("unexpected frozen C1 configuration")
    metadata = load_json(DEFENSE / "embed_meta.json")["images"][:200]
    if len(metadata) != 200 or len({m["input_image"] for m in metadata}) != 200:
        raise RuntimeError("expected 200 unique source images")
    context = {"cache": PixelCache(CACHE), "cache_states": {}, "config": config,
               "model_hash": sha256_file(ROOT / "ckpts/params.json"),
               "checkpoint_hash": sha256_file(ROOT / "ckpts/wam_mit.pth")}
    return attacks, audits, metadata, context


def inventory_rows(attacks, audits, metadata, context):
    rows = []
    for index, meta in enumerate(metadata):
        parent = contained_file(DEFENSE / "watermarked_signed_ra", meta["signed_image"])
        context.update(parent_path=parent, parent_file_hash=sha256_file(parent),
                       parent_key=decoded_rgb_key(parent)[0])
        with Image.open(parent) as source:
            source_rgb = source.convert("RGB")
        for rho in (n/10 for n in range(9)):
            for method in VALID_METHODS:
                identity = f"{index:04d}|{method}|{rho:.3f}"
                if attacks[identity]["image_index"] != index:
                    raise RuntimeError("C1 enumeration index changed")
                rows.append(inventory_row(attacks[identity], audits[identity], meta, source_rgb, context))
        print(f"C1 CPU inventory {len(rows)}/10800", flush=True)
    return rows


def historical_selection(audits):
    groups = defaultdict(list)
    for row in audits.values():
        if row["method"] in EDGES:
            groups[(row["input_image"], row["rho_requested"])].append(row)
    return {min(group, key=lambda r: (float(r["region_retention"]), EDGES.index(r["method"])))
            ["attack_id"] for group in groups.values()}


def summarize_inventory(rows, audits, context):
    mask_aware = mask_aware_selection(rows)
    if {r["selected_attack_id"] for r in mask_aware} != historical_selection(audits):
        raise RuntimeError("geometry-only mask-aware choices differ from historical selection")
    unique = {r["pixel_key"] for r in rows}
    unique_hits = sum(context["cache_states"][key] is not None for key in unique)
    counts = Counter(r["cache_hit"] for r in rows)
    missing = {r["pixel_key"]: r for r in reversed(rows) if not r["cache_hit"]}
    payload = {"schema": "protocol-v2-c1-inventory-1", "records": len(rows),
               "unique_pixels": len(unique), "record_cache_hits": counts[True],
               "record_cache_misses": counts[False], "unique_cache_hits": unique_hits,
               "unique_cache_misses": len(unique)-unique_hits, "paired_record_checks": len(rows),
               "decoded_pixel_reconstruction_checks": len(rows),
               "legacy_escapes": sum(r["legacy_match_count"] < 24 for r in rows),
               "mask_aware_derived_records": len(mask_aware), "per_method_rho": grouped_summary(rows),
               "gpu_inference_calls": 0, "signing_calls": 0, "manifest_validation_calls": 0,
               "params_sha256": context["model_hash"], "checkpoint_sha256": context["checkpoint_hash"],
               "script_sha256": sha256_file(Path(__file__)),
               "source_hashes": {name: sha256_file(SOURCE / name) for name in
                                  ("attack_manifest.jsonl", "audit_rows.jsonl", "config_effective.json")},
               "embed_meta_sha256": sha256_file(DEFENSE / "embed_meta.json"),
               "reference_scope": "experiment-side frozen metadata, not deployed verification",
               "geometry_scope": "experiment-side crop boxes; no v2 signing performed",
               "cache_scope": "shared full_pixel_cache_v1 only; no inference on missing keys",
               "resize_scope": "legacy C1 LANCZOS resize to original dimensions, not platform reframing"}
    return payload, list(missing.values()), mask_aware


def write_results(rows, payload, missing, mask_aware):
    for name, data in (("rows.json", rows), ("summary.json", payload),
                       ("missing_unique_pixels.json", missing), ("mask_aware_selection.json", mask_aware)):
        atomic_json(OUTPUT / name, data)
    unique = payload["unique_pixels"]
    hits, unique_hits = payload["record_cache_hits"], payload["unique_cache_hits"]
    text = (f"# C1 migration inventory\n\n{len(rows)} paired records; {unique} unique decoded RGB "
            f"pixels. Cache hits: {hits} records / {unique_hits} unique pixels. "
            f"Missing: {len(rows)-hits} records / {unique-unique_hits} unique pixels.\n\n"
            "All crop/seed/area/retention metadata and saved RGB pixels were reconstructed and checked. "
            "Every cache hit validates model, checkpoint, preprocessing and exact integer match count. "
            "Cache misses were not inferred. References and geometry remain experiment-side inventory "
            "inputs, not a deployed audit. No signing or C2PA validity check was performed.\n\n"
            "The six methods contribute 10,800 records. Mask-aware is 1,800 geometry-selected existing "
            "edge candidates, not a seventh independent pixel cohort. Zero-removal duplicates and "
            "other identical pixels are deduplicated by decoded RGB key. This is legacy C1 with "
            "LANCZOS resize to original dimensions; it is not the future non-square platform suite. "
            "Missing-cache warming and a signed v2 pilot require separate approval.\n")
    (OUTPUT / "report.md").write_text(text, encoding="utf-8")
    print({k: payload[k] for k in ("records", "unique_pixels", "unique_cache_hits", "unique_cache_misses")},
          flush=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    attacks, audits, metadata, context = inventory_context()
    rows = inventory_rows(attacks, audits, metadata, context)
    payload, missing, mask_aware = summarize_inventory(rows, audits, context)
    write_results(rows, payload, missing, mask_aware)


if __name__ == "__main__":
    main()
