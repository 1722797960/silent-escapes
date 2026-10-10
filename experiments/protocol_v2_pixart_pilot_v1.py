"""Freeze and warm exactly twenty PixArt boundary cases on the server.

Only --phase warm may construct the detector. This pilot is not a population
estimate. Never consumes legacy region occupancy as retention.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
from protocol_v2_cache import PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_cache_warm_v1 import PREPROCESSING_VERSION, validate_entry
from protocol_v2_main_curve_full_v1 import append_jsonl, package_version
from protocol_v2_pilot_v1 import (
    Detector, atomic_json, crop_transform, load_json, load_jsonl, make_region_assertion,
)

ROOT = Path(__file__).resolve().parents[1]


def add_selection(selected: dict, row: dict, reason: str) -> None:
    entry = selected.setdefault(row["attack_id"], {**row, "selection_reasons": []})
    entry["selection_reasons"].append(reason)


def mandatory_boundaries(rows: list[dict], metadata: dict) -> dict:
    selected = {}
    failures = [r for r in rows if r["crop_frac"] == 0 and r["legacy_match_count"] < 24]
    add_selection(selected, min(failures, key=lambda r: (r["legacy_match_count"], r["attack_id"])),
                  "no-edit decoder failure")
    successes = [r for r in rows if r["crop_frac"] == 0 and r["legacy_match_count"] >= 24]
    add_selection(selected, min(successes, key=lambda r: (metadata[r["image"]]["mask_coverage"], r["attack_id"])),
                  "successful no-edit control")
    for matches in (24, 23):
        candidates = [r for r in rows if r["legacy_match_count"] == matches]
        add_selection(selected, min(candidates, key=lambda r: r["attack_id"]),
                      f"integer boundary: {matches}/32")
    return selected


def extend_crop_coverage(selected: dict, rows: list[dict], metadata: dict) -> None:
    for crop in sorted({r["crop_frac"] for r in rows}):
        cohort = sorted([r for r in rows if r["crop_frac"] == crop],
                        key=lambda r: (metadata[r["image"]]["mask_coverage"], r["attack_id"]))
        choices = [cohort[0], cohort[-1]] + cohort
        for row in choices:
            if sum(r["crop_frac"] == crop for r in selected.values()) >= 2:
                break
            if row["attack_id"] not in selected:
                add_selection(selected, row, f"coverage-range / crop-level {crop:.2f}")


def extend_extras(selected: dict, rows: list[dict]) -> None:
    above = [r for r in rows if r["legacy_match_count"] == 25 and r["attack_id"] not in selected]
    if above and len(selected) < 20:
        add_selection(selected, min(above, key=lambda r: r["attack_id"]), "above-boundary: 25/32")
    failures = [r for r in rows if r["crop_frac"] == 0 and r["legacy_match_count"] < 24]
    for row in sorted(failures + rows, key=lambda r: (abs(r["legacy_match_count"]-24), r["attack_id"])):
        if len(selected) >= 20:
            break
        if row["attack_id"] not in selected:
            add_selection(selected, row, "additional near-threshold case")


def choose_boundaries(rows: list[dict], metadata: dict) -> list[dict]:
    """Two cases per crop plus two extras; diagnostic, never a rate estimate."""
    selected = mandatory_boundaries(rows, metadata)
    extend_crop_coverage(selected, rows, metadata)
    extend_extras(selected, rows)
    result = sorted(selected.values(), key=lambda r: (r["image_index"], r["crop_frac"]))
    if len(result) != 20 or len({r["crop_frac"] for r in result}) != 9:
        raise RuntimeError("pilot must contain 20 unique cases and all nine crop levels")
    return result


def source_context(args) -> dict:
    inventory = args.inventory_dir.resolve(strict=True)
    summary = load_json(inventory / "summary.json")
    defense = ROOT / "results/defense_e2e_pixart"
    audit_file, meta_file = defense / "audit.json", defense / "embed_meta.json"
    if sha256_file(audit_file) != summary["audit_sha256"] or sha256_file(meta_file) != summary["embed_meta_sha256"]:
        raise RuntimeError("PixArt source inputs changed since inventory")
    rows = load_json(inventory / "rows.json")
    if len(rows) != 1800 or sum(r["legacy_match_count"] < 24 for r in rows) != 940:
        raise RuntimeError("unexpected full PixArt cohort")
    metadata = {r["input_image"]: r for r in load_json(meta_file)["images"]}
    return dict(rows=rows, metadata=metadata, defense=defense, inventory=inventory,
                inventory_rows_sha256=sha256_file(inventory / "rows.json"))


def prepare_case(row: dict, context: dict, ordinal: int) -> dict:
    meta = context["metadata"][row["image"]]
    source = (ROOT / row["source_relpath"]).resolve(strict=True)
    key, image = decoded_rgb_key(source)
    if key != row["pixel_key"] or image != row["image_size"]:
        raise RuntimeError(f"pixels changed since inventory: {row['attack_id']}")
    parent = (context["defense"] / "watermarked_unsigned" / meta["unsigned_image"]).resolve(strict=True)
    parent_key, parent_image = decoded_rgb_key(parent)
    if [parent_image["width"], parent_image["height"]] != row["source_size_px"]:
        raise RuntimeError("parent size differs from frozen geometry")
    transform = None if row["crop_frac"] == 0 else crop_transform(
        row["source_size_px"], row["crop_box_px"], [image["width"], image["height"]])
    return {"ordinal": ordinal, "case_id": row["attack_id"], "attack_id": row["attack_id"],
            "image": row["image"], "image_index": row["image_index"], "crop_frac": row["crop_frac"],
            "child_source_relpath": row["source_relpath"], "parent_source_relpath": str(parent.relative_to(ROOT)),
            "source_file_sha256": sha256_file(source), "parent_file_sha256": sha256_file(parent),
            "parent_pixel_key": parent_key, "pixel_key": key, "image_size": image,
            "crop_transform": transform, "region_assertion": make_region_assertion(meta),
            "legacy_match_count": row["legacy_match_count"],
            "mask_coverage": meta["mask_coverage"], "selection_reasons": row["selection_reasons"]}


def select(args) -> None:
    context = source_context(args)
    rows = choose_boundaries(context["rows"], context["metadata"])
    cases = [prepare_case(row, context, ordinal) for ordinal, row in enumerate(rows)]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output_dir / "selected_cases.json", cases)
    atomic_json(args.output_dir / "selection_summary.json", {
        "case_count": len(cases), "inventory_rows_sha256": context["inventory_rows_sha256"],
        "config_sha256": sha256_file(args.config), "selection_script_sha256": sha256_file(Path(__file__)),
        "per_crop": dict(Counter(f"{r['crop_frac']:.2f}" for r in cases)),
        "legacy_match_counts": dict(Counter(str(r["legacy_match_count"]) for r in cases)),
        "expected_pilot_escapes": sum(r["legacy_match_count"] < 24 for r in cases),
        "population_estimate": False,
    })
    print(json.dumps(load_json(args.output_dir / "selection_summary.json"), indent=2), flush=True)


def checked_cases(args) -> list[dict]:
    selection = load_json(args.output_dir / "selection_summary.json")
    if selection["config_sha256"] != sha256_file(args.config):
        raise RuntimeError("pilot config changed after selection")
    cases = load_json(args.output_dir / "selected_cases.json")
    if len(cases) != 20 or len({r["case_id"] for r in cases}) != 20:
        raise RuntimeError("only the approved twenty-case pilot is supported")
    return cases


def validate_cached(context: dict, case: dict, entry: dict) -> int:
    validate_entry(entry, key=case["pixel_key"], image=case["image_size"],
                   model_sha256=context["model_hash"], checkpoint_sha256=context["checkpoint_hash"])
    reference = case["region_assertion"]["reference_bits"]
    if len(entry["decoded_bits"]) != 32:
        raise RuntimeError("cache decoder output must contain 32 bits")
    return sum(a == b for a, b in zip(reference, entry["decoded_bits"]))


def warm_case(context: dict, case: dict) -> dict:
    source = ROOT / case["child_source_relpath"]
    if sha256_file(source) != case["source_file_sha256"] or decoded_rgb_key(source)[0] != case["pixel_key"]:
        raise RuntimeError(f"source changed: {case['case_id']}")
    started = time.perf_counter()
    entry = context["cache"].get(case["pixel_key"])
    status = "cache-hit"
    if entry is None:
        if context["detector"] is None:
            model_started = time.perf_counter()
            context["detector"] = Detector(wam_repo=context["repo"], params=ROOT / "ckpts/params.json",
                                           checkpoint=ROOT / "ckpts/wam_mit.pth", device=context["device"])
            context["model_load_seconds"] = time.perf_counter()-model_started
            started = time.perf_counter()
        bits, logits = context["detector"].run(source)
        context["inference_calls"] += 1
        raw_dir = context["output"] / "raw_logits"
        raw_dir.mkdir(exist_ok=True)
        np.savez_compressed(raw_dir / f"{case['case_id']}.npz", mask_logits=np.asarray(logits, dtype=np.float32))
        context["cache"].put(key=case["pixel_key"], decoded_bits=bits, mask_logits=logits,
                             image=case["image_size"], model_sha256=context["model_hash"],
                             checkpoint_sha256=context["checkpoint_hash"], preprocessing_version=PREPROCESSING_VERSION)
        entry = context["cache"].get(case["pixel_key"])
        status = "gpu-inference"
    matches = validate_cached(context, case, entry)
    result = {"case_id": case["case_id"], "pixel_key": case["pixel_key"], "status": status,
              "match_count": matches, "legacy_match_count": case["legacy_match_count"],
              "decoded_bits": entry["decoded_bits"], "elapsed_seconds": time.perf_counter()-started}
    if matches != case["legacy_match_count"]:
        atomic_json(context["output"] / "failure.json", result)
        raise RuntimeError(f"STOP: decoder mismatch: {case['case_id']}: {matches}/{case['legacy_match_count']}")
    return result


def warm(args) -> None:
    cases = checked_cases(args)
    rows_path = args.output_dir / "warm_rows.jsonl"
    if rows_path.exists():
        raise FileExistsError("pilot warming refuses repeat runs; preserve original outputs")
    context = dict(cache=PixelCache(args.cache_dir), detector=None, device=args.device,
                   repo=args.wam_repo.resolve(strict=True), output=args.output_dir,
                   model_hash=sha256_file(ROOT / "ckpts/params.json"),
                   checkpoint_hash=sha256_file(ROOT / "ckpts/wam_mit.pth"),
                   model_load_seconds=0, inference_calls=0)
    started = time.perf_counter()
    for index, case in enumerate(cases, 1):
        row = warm_case(context, case)
        append_jsonl(rows_path, row)
        print(f"PixArt pilot warm {index}/20: {case['case_id']} matched {row['match_count']}/32", flush=True)
    rows = load_jsonl(rows_path)
    atomic_json(args.output_dir / "warm_summary.json", {
        "cases": len(rows), "inference_calls": context["inference_calls"],
        "cache_hits": sum(r["status"] == "cache-hit" for r in rows), "all_integer_counts_match": True,
        "wall_seconds": time.perf_counter()-started, "model_load_seconds": context["model_load_seconds"],
        "per_case_elapsed_seconds": [r["elapsed_seconds"] for r in rows], "device": args.device,
        "params_sha256": context["model_hash"], "checkpoint_sha256": context["checkpoint_hash"],
        "preprocessing_version": PREPROCESSING_VERSION, "python_platform": platform.platform(),
        "torch": package_version("torch"), "numpy": package_version("numpy"),
        "pilot_script_sha256": sha256_file(Path(__file__)),
        "selected_cases_sha256": sha256_file(args.output_dir / "selected_cases.json"),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("select", "warm"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/protocol_v2_pixart_pilot_v1.json")
    parser.add_argument("--inventory-dir", type=Path, default=ROOT / "results/protocol_v2/pixart_inventory_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/protocol_v2/pixart_pilot_v1")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results/protocol_v2/full_pixel_cache_v1/cache")
    parser.add_argument("--wam-repo", type=Path, default=ROOT / "third_party/watermark-anything")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_json(args.config)
    if (config["pilot_cases"], config["required_matches"], config["retention_floor"],
            config["suppression_threshold"]) != (20, 24, .05, .5):
        raise RuntimeError("pilot must use twenty cases and 24/32 integer matching")
    select(args) if args.phase == "select" else warm(args)


if __name__ == "__main__":
    main()
