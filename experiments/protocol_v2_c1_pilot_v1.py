"""Freeze twenty C1 boundary cases, then rerun exactly those pixels on server GPU.

Existing cache entries are never replaced. Genuine float32 outputs are private;
hit-cache comparisons include rerun effects, not just float16 quantization.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from protocol_v2_cache import PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_cache_warm_v1 import PREPROCESSING_VERSION
from protocol_v2_c1_inventory_v1 import contained_file
from protocol_v2_common import matching_bit_count
from protocol_v2_main_curve_full_v1 import append_jsonl, package_version
from protocol_v2_pilot_v1 import Detector, atomic_json, crop_transform, load_json, load_jsonl, make_region_assertion
from protocol_v2_pixart_pilot_v1 import validate_cached

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/protocol_v2/c1_migration_pilot_v1"
METHODS = {"center", "left", "right", "top", "bottom", "random"}
RHO_LEVELS = {n / 10 for n in range(9)}
MANDATORY = ("0046|center|0.000", "0117|center|0.000")


class BoundarySelection:
    def __init__(self):
        self.selected, self.used = {}, set()

    def add(self, row, reason):
        if row["attack_id"] in self.selected:
            self.selected[row["attack_id"]]["selection_reasons"].append(reason)
        elif row["pixel_key"] not in self.used:
            self.selected[row["attack_id"]] = {**row, "selection_reasons": [reason]}
            self.used.add(row["pixel_key"])

    def pick(self, candidates, reason, key=None):
        available = [r for r in candidates if r["pixel_key"] not in self.used]
        if not available:
            raise RuntimeError(f"no unique pixels available for boundary: {reason}")
        self.add(min(available, key=key or (lambda r: r["attack_id"])), reason)


def choose_boundaries(rows, metadata):
    selection = BoundarySelection()
    add, pick = selection.add, selection.pick
    by_id = {r["attack_id"]: r for r in rows}
    for identity in MANDATORY:
        add(by_id[identity], "mandatory no-edit decoder failure")
    pick([r for r in rows if r["rho_requested"] == 0 and r["legacy_match_count"] >= 24],
         "successful no-edit control")
    for matches in (24, 25):
        pick([r for r in rows if r["legacy_match_count"] == matches], f"integer boundary {matches}/32")
    select_regional_boundaries(rows, metadata, selection)
    fill_coverage(rows, selection.selected, selection.used, add)
    for row in sorted(rows, key=lambda r: (abs(r["legacy_match_count"] - 24), r["attack_id"])):
        if len(selection.selected) >= 20:
            break
        add(row, "additional integer boundary coverage")
    result = sorted(selection.selected.values(), key=lambda r: (r["image_index"], r["rho_requested"], r["method"]))
    validate_coverage(result)
    return result


def select_regional_boundaries(rows, metadata, selection):
    pick = selection.pick
    for direction in (-1, 1):
        pick(rows, "low source-mask coverage" if direction == 1 else "high source-mask coverage",
             key=lambda r: (direction * metadata[r["input_image"]]["mask_coverage"], r["attack_id"]))
    pick([r for r in rows if r["region_retention_unrounded"] == 0], "empty retained region")
    pick(rows, "nearest inventory retention to 0.05",
         key=lambda r: (abs(r["region_retention_unrounded"] - .05), r["attack_id"]))
    pick(rows, "nearest legacy regional mean to 0.5 (selection only)",
         key=lambda r: (abs(r["legacy_logit_in_region"] - .5), r["attack_id"]))


def coverage_tokens(row):
    return {("rho", row["rho_requested"]), ("method", row["method"]), ("hit", row["cache_hit"])}


def fill_coverage(rows, selected, used, add):
    required = ({("rho", v) for v in RHO_LEVELS} | {("method", m) for m in METHODS}
                | {("hit", False), ("hit", True)})
    while True:
        covered = set().union(*(coverage_tokens(r) for r in selected.values()))
        missing = required - covered
        if not missing:
            return
        if len(selected) >= 20:
            raise RuntimeError("twenty-case budget cannot cover declared boundaries")
        candidates = [r for r in rows if r["pixel_key"] not in used]
        best = min(candidates, key=lambda r: (-len(coverage_tokens(r) & missing), r["attack_id"]))
        if not coverage_tokens(best) & missing:
            raise RuntimeError("declared coverage unavailable")
        add(best, "joint rho/method/cache coverage")


def validate_coverage(rows):
    observed = {field: set(map(lambda row: row[field], rows)) for field in
                ("pixel_key", "attack_id", "rho_requested", "method", "cache_hit", "legacy_match_count")}
    checks = [len(rows) == 20, len(observed["pixel_key"]) == 20, len(observed["attack_id"]) == 20,
              observed["rho_requested"] == RHO_LEVELS, observed["method"] == METHODS,
              observed["cache_hit"] == {True, False}, set(MANDATORY) <= observed["attack_id"],
              {17, 23, 24, 25} <= observed["legacy_match_count"],
              any(r["region_retention_unrounded"] == 0 for r in rows),
              any(r["rho_requested"] == 0 and r["legacy_match_count"] >= 24 for r in rows)]
    if not all(checks):
        raise RuntimeError(f"boundary coverage failed: {checks}")


def checked_config(args):
    config = load_json(args.config)
    if (config["pilot_cases"], config["max_inference_calls"], config["required_matches"],
            config["retention_floor"], config["suppression_threshold"]) != (20, 20, 24, .05, .5):
        raise RuntimeError("only approved twenty-case, unchanged-threshold pilot is supported")
    return config


def cache_hashes(cache, key):
    directory = cache.root / key[:2]
    return {suffix: sha256_file(directory / f"{key}.{suffix}") for suffix in ("json", "npz")}


def source_context(args):
    config = checked_config(args)
    inventory = args.inventory_dir.resolve(strict=True)
    for name in ("rows", "summary"):
        if sha256_file(inventory / f"{name}.json") != config[f"inventory_{name}_sha256"]:
            raise RuntimeError("accepted C1 inventory changed")
    summary = load_json(inventory / "summary.json")
    for name, expected in summary["source_hashes"].items():
        if sha256_file(ROOT / "results/c1_v1/full" / name) != expected:
            raise RuntimeError("frozen C1 source changed")
    meta_path = ROOT / "results/defense_e2e/embed_meta.json"
    if sha256_file(meta_path) != summary["embed_meta_sha256"]:
        raise RuntimeError("embedding metadata changed")
    old = {r["attack_id"]: r for r in load_jsonl(ROOT / "results/c1_v1/full/audit_rows.jsonl")}
    rows = load_json(inventory / "rows.json")
    metadata = {m["input_image"]: m for m in load_json(meta_path)["images"][:200]}
    for row in rows:
        row["legacy_logit_in_region"] = old[row["attack_id"]]["logit_in_region"]
    return {"rows": rows, "metadata": metadata, "old": old, "config": config}


def prepare_case(row, context, ordinal):
    meta = context["metadata"][row["input_image"]]
    source = contained_file(ROOT / "results/c1_v1/full", row["output_relpath"])
    parent = contained_file(ROOT / "results/defense_e2e/watermarked_unsigned", meta["unsigned_image"])
    key, image = decoded_rgb_key(source)
    parent_key, parent_image = decoded_rgb_key(parent)
    if (key != row["pixel_key"] or sha256_file(source) != row["source_file_sha256"]
            or image != row["image_size"] or parent_key != row["parent_pixel_key"]
            or [parent_image["width"], parent_image["height"]] != row["original_size"]):
        raise RuntimeError("C1 source pixels/hash/dimensions changed")
    transform = None if row["rho_requested"] == 0 else crop_transform(
        row["original_size"], row["crop_box_px"], [image["width"], image["height"]])
    region = make_region_assertion(meta)
    if region["reference_bits"] != row["reference_bits"]:
        raise RuntimeError("inventory reference changed")
    return {**row, "ordinal": ordinal, "case_id": f"c1_{ordinal:02d}__{row['pixel_key'][:12]}",
            "image": row["input_image"], "child_source_relpath": str(source.relative_to(ROOT)),
            "parent_source_relpath": str(parent.relative_to(ROOT)), "parent_file_sha256": sha256_file(parent),
            "region_assertion": region, "crop_transform": transform,
            "legacy_evidence": context["old"][row["attack_id"]]["audit_v2_region_aware"]}


def select(args):
    context = source_context(args)
    rows = choose_boundaries(context["rows"], context["metadata"])
    cases = [prepare_case(r, context, i) for i, r in enumerate(rows)]
    cache = PixelCache(args.cache_dir)
    for case in cases:
        entry = cache.get(case["pixel_key"])
        if (entry is not None) != case["cache_hit"]:
            raise RuntimeError("cache coverage changed after inventory")
        case["preserved_cache_hashes"] = cache_hashes(cache, case["pixel_key"]) if entry is not None else None
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output_dir / "selected_cases.json", cases)
    summary = {"cases": 20, "unique_pixels": 20, "population_estimate": False,
               "config_sha256": sha256_file(args.config),
               "selected_cases_sha256": sha256_file(args.output_dir / "selected_cases.json"),
               "inventory_rows_sha256": context["config"]["inventory_rows_sha256"],
               "params_sha256": load_json(args.inventory_dir / "summary.json")["params_sha256"],
               "checkpoint_sha256": load_json(args.inventory_dir / "summary.json")["checkpoint_sha256"],
               "cache_hits": sum(r["cache_hit"] for r in cases),
               "cache_misses": sum(not r["cache_hit"] for r in cases),
               "per_method": dict(Counter(r["method"] for r in cases)),
               "per_rho": dict(Counter(f"{r['rho_requested']:.1f}" for r in cases)),
               "selection_script_sha256": sha256_file(Path(__file__))}
    atomic_json(args.output_dir / "selection_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def checked_cases(args):
    checked_config(args)
    summary = load_json(args.output_dir / "selection_summary.json")
    if summary["config_sha256"] != sha256_file(args.config):
        raise RuntimeError("config changed after selection")
    if summary["selected_cases_sha256"] != sha256_file(args.output_dir / "selected_cases.json"):
        raise RuntimeError("selected cases changed after selection")
    for filename, field in (("params.json", "params_sha256"), ("wam_mit.pth", "checkpoint_sha256")):
        if sha256_file(ROOT / "ckpts" / filename) != summary[field]:
            raise RuntimeError("model/checkpoint changed since accepted inventory")
    cases = load_json(args.output_dir / "selected_cases.json")
    validate_coverage(cases)
    return cases


def check_decoded(case, bits, entry):
    count = matching_bit_count(case["region_assertion"]["reference_bits"], bits)
    if len(bits) != 32 or count != case["legacy_match_count"]:
        raise RuntimeError(f"historical integer mismatch: {case['attack_id']}: {count}")
    if entry is not None and bits != entry["decoded_bits"]:
        raise RuntimeError(f"cached decoded bits changed on rerun: {case['attack_id']}")
    return count


def checked_warm_input(context, case):
    source = contained_file(ROOT, case["child_source_relpath"])
    if sha256_file(source) != case["source_file_sha256"] or decoded_rgb_key(source)[0] != case["pixel_key"]:
        raise RuntimeError("source changed after selection")
    cache = context["cache"]
    entry = cache.get(case["pixel_key"])
    if (entry is not None) != case["cache_hit"]:
        raise RuntimeError("selected cache-hit state changed; review before inference")
    if entry is not None:
        validate_cached(context, case, entry)
        if cache_hashes(cache, case["pixel_key"]) != case["preserved_cache_hashes"]:
            raise RuntimeError("existing cache changed after selection")
    if context["inference_calls"] >= 20:
        raise RuntimeError("approved inference budget exhausted")
    return source, entry


def save_raw(context, case, logits):
    logits = np.asarray(logits)
    if logits.dtype != np.float32 or logits.ndim != 2 or not np.isfinite(logits).all():
        raise RuntimeError("detector must emit genuine finite two-dimensional float32 output")
    if np.min(logits) < 0 or np.max(logits) > 1:
        raise RuntimeError("mask probabilities outside [0,1]")
    raw_path = context["output"] / "raw_logits" / f"{case['case_id']}.npz"
    with raw_path.open("xb") as handle:
        np.savez_compressed(handle, mask_logits=logits)
    return raw_path


def warm_case(context, case):
    source, entry = checked_warm_input(context, case)
    cache = context["cache"]
    started = time.perf_counter()
    bits, logits = context["detector"].run(source)
    context["inference_calls"] += 1
    raw_path = save_raw(context, case, logits)
    diagnostic = {"case_id": case["case_id"], "attack_id": case["attack_id"], "decoded_bits": bits,
                  "legacy_match_count": case["legacy_match_count"], "cache_hit_before": entry is not None,
                  "raw_sha256": sha256_file(raw_path)}
    try:
        count = check_decoded(case, bits, entry)
    except RuntimeError:
        atomic_json(context["output"] / "warm_failure.json", diagnostic)
        raise
    if entry is None:
        cache.put(key=case["pixel_key"], decoded_bits=bits, mask_logits=logits, image=case["image_size"],
                  model_sha256=context["model_hash"], checkpoint_sha256=context["checkpoint_hash"],
                  preprocessing_version=PREPROCESSING_VERSION)
    cached = cache.get(case["pixel_key"])
    validate_cached(context, case, cached)
    if entry is not None and cache_hashes(cache, case["pixel_key"]) != case["preserved_cache_hashes"]:
        raise RuntimeError("existing cache was modified during rerun")
    return {**diagnostic, "match_count": count, "pixel_key": case["pixel_key"],
            "comparison_scope": "rerun-and-quantization" if entry is not None else "quantization-only",
            "elapsed_seconds": time.perf_counter() - started}


def warm(args):
    cases = checked_cases(args)
    rows_path = args.output_dir / "warm_rows.jsonl"
    raw_dir = args.output_dir / "raw_logits"
    if rows_path.exists() or raw_dir.exists():
        raise FileExistsError("preserve first pilot warming outputs; no implicit repeat")
    raw_dir.mkdir()
    context = {"cache": PixelCache(args.cache_dir), "output": args.output_dir, "inference_calls": 0,
               "model_hash": sha256_file(ROOT / "ckpts/params.json"),
               "checkpoint_hash": sha256_file(ROOT / "ckpts/wam_mit.pth")}
    started = time.perf_counter()
    context["detector"] = Detector(wam_repo=args.wam_repo.resolve(strict=True), params=ROOT / "ckpts/params.json",
                                   checkpoint=ROOT / "ckpts/wam_mit.pth", device=args.device)
    model_seconds = time.perf_counter() - started
    for index, case in enumerate(cases, 1):
        append_jsonl(rows_path, warm_case(context, case))
        print(f"C1 pilot inference {index}/20", flush=True)
    rows = load_jsonl(rows_path)
    atomic_json(args.output_dir / "warm_summary.json", {
        "cases": len(rows), "inference_calls": context["inference_calls"], "all_integer_counts_match": True,
        "cache_hits_preserved": sum(r["cache_hit_before"] for r in rows),
        "new_cache_entries": sum(not r["cache_hit_before"] for r in rows),
        "all_hit_bits_identical": True, "raw_float32_cases": len(rows), "wall_seconds": time.perf_counter()-started,
        "model_load_seconds": model_seconds, "device": args.device,
        "params_sha256": context["model_hash"], "checkpoint_sha256": context["checkpoint_hash"],
        "preprocessing_version": PREPROCESSING_VERSION,
        "environment": {n: package_version(n) for n in ("torch", "numpy", "Pillow")},
        "pilot_script_sha256": sha256_file(Path(__file__)),
        "selected_cases_sha256": sha256_file(args.output_dir / "selected_cases.json")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("select", "warm"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/protocol_v2_c1_pilot_v1.json")
    parser.add_argument("--inventory-dir", type=Path, default=ROOT / "results/protocol_v2/c1_inventory_v1")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "results/protocol_v2/full_pixel_cache_v1/cache")
    parser.add_argument("--wam-repo", type=Path, default=ROOT / "third_party/watermark-anything")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    select(args) if args.phase == "select" else warm(args)


if __name__ == "__main__":
    main()
