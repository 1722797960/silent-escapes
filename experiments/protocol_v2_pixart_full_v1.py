"""Freeze/warm the complete PixArt per-side curve; no signing in this CLI."""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from protocol_v2_cache import PixelCache, sha256_file
from protocol_v2_main_curve_full_v1 import append_jsonl, package_version
from protocol_v2_pilot_v1 import atomic_json, load_json, load_jsonl
from protocol_v2_pixart_pilot_v1 import ROOT, prepare_case, source_context, validate_cached, warm_case

OUTPUT = ROOT / "results/protocol_v2/pixart_full_v1"
PILOT = ROOT / "results/protocol_v2/pixart_pilot_v1"
CONFIG = ROOT / "configs/protocol_v2_pixart_full_v1.json"
CACHE = ROOT / "results/protocol_v2/full_pixel_cache_v1/cache"


def validate_config(config):
    expected = (1800, 200, 940, 24, .05, .5)
    actual = tuple(config[k] for k in ("expected_cases", "expected_images", "expected_legacy_escapes",
                                     "required_matches", "retention_floor", "suppression_threshold"))
    if actual != expected or config["crop_parameter"] != "fraction removed from each side":
        raise RuntimeError("only the approved 1800-case per-side PixArt protocol is supported")


def validate_identities(cases):
    if len(cases) != 1800 or len({c["case_id"] for c in cases}) != 1800:
        raise RuntimeError("expected 1800 unique case IDs")
    if len({c["pixel_key"] for c in cases}) != 1800 or len({c["image_index"] for c in cases}) != 200:
        raise RuntimeError("expected 1800 unique pixels from 200 source images")


def validate_cohort(cases):
    validate_identities(cases)
    expected = Counter({round(i * .05, 2): 200 for i in range(9)})
    if Counter(c["crop_frac"] for c in cases) != expected:
        raise RuntimeError("not the complete nine-level per-side curve")
    if sum(c["legacy_match_count"] < 24 for c in cases) != 940:
        raise RuntimeError("unexpected frozen legacy escape count")
    by_image = {}
    for case in cases:
        by_image.setdefault(case["image_index"], set()).add(case["crop_frac"])
    for levels in by_image.values():
        if levels != set(expected):
            raise RuntimeError("a source image lacks crop levels")


def model_context(output=OUTPUT):
    return dict(output=output, cache=PixelCache(CACHE),
                model_hash=sha256_file(ROOT / "ckpts/params.json"),
                checkpoint_hash=sha256_file(ROOT / "ckpts/wam_mit.pth"))


def freeze(args):
    validate_config(load_json(CONFIG))
    if not load_json(PILOT / "audit_summary.json")["acceptance_passed"]:
        raise RuntimeError("pilot audit acceptance required")
    if not load_json(PILOT / "verification_summary.json")["acceptance_passed"]:
        raise RuntimeError("independent pilot acceptance required")
    context = source_context(args)
    rows = sorted(context["rows"], key=lambda r: (r["image_index"], r["crop_frac"]))
    cases = []
    for ordinal, row in enumerate(rows):
        cases.append(prepare_case({**row, "selection_reasons": ["complete frozen PixArt cohort"]}, context, ordinal))
        if (ordinal + 1) % 200 == 0:
            print(f"PixArt full freeze {ordinal+1}/1800", flush=True)
    validate_cohort(cases)
    cached = model_context()
    hits = 0
    for case in cases:
        entry = cached["cache"].get(case["pixel_key"])
        if entry is not None:
            if validate_cached(cached, case, entry) != case["legacy_match_count"]:
                raise RuntimeError("existing cached count differs from frozen legacy")
            hits += 1
    OUTPUT.mkdir(parents=True, exist_ok=False)
    atomic_json(OUTPUT / "selected_cases.json", cases)
    atomic_json(OUTPUT / "selection_summary.json", {
        "cases": 1800, "images": 200, "legacy_escapes": 940,
        "initial_cache_hits": hits, "initial_cache_misses": 1800-hits,
        "config_sha256": sha256_file(CONFIG),
        "selected_cases_sha256": sha256_file(OUTPUT / "selected_cases.json"),
        "inventory_rows_sha256": context["inventory_rows_sha256"],
        "audit_sha256": sha256_file(context["defense"] / "audit.json"),
        "embed_meta_sha256": sha256_file(context["defense"] / "embed_meta.json"),
        "params_sha256": cached["model_hash"], "checkpoint_sha256": cached["checkpoint_hash"],
        "pilot_audit_sha256": sha256_file(PILOT / "audit_summary.json"),
        "pilot_verification_sha256": sha256_file(PILOT / "verification_summary.json"),
        "selection_script_sha256": sha256_file(Path(__file__)),
        "legacy_region_metric_consumed": False,
    })


def checked_cases():
    validate_config(load_json(CONFIG))
    summary = load_json(OUTPUT / "selection_summary.json")
    bindings = {CONFIG: "config_sha256", OUTPUT / "selected_cases.json": "selected_cases_sha256",
                ROOT / "ckpts/params.json": "params_sha256", ROOT / "ckpts/wam_mit.pth": "checkpoint_sha256",
                ROOT / "results/defense_e2e_pixart/audit.json": "audit_sha256",
                ROOT / "results/defense_e2e_pixart/embed_meta.json": "embed_meta_sha256",
                PILOT / "audit_summary.json": "pilot_audit_sha256",
                PILOT / "verification_summary.json": "pilot_verification_sha256"}
    for path, field in bindings.items():
        if sha256_file(path) != summary[field]:
            raise RuntimeError(f"frozen input changed: {path}")
    cases = load_json(OUTPUT / "selected_cases.json")
    validate_cohort(cases)
    return cases


def indexed_rows(path, cases, resume):
    if path.exists() and not resume:
        raise FileExistsError("existing rows require explicit --resume")
    rows = load_jsonl(path) if path.exists() else []
    indexed = {r["case_id"]: r for r in rows}
    if len(indexed) != len(rows) or set(indexed) - {c["case_id"] for c in cases}:
        raise RuntimeError("duplicate or out-of-cohort resume records")
    return indexed


def raw_path(case):
    candidates = [OUTPUT / "raw_logits" / f"{case['case_id']}.npz",
                  PILOT / "raw_logits" / f"{case['case_id']}.npz"]
    available = [p for p in candidates if p.is_file()]
    if not available:
        raise RuntimeError(f"raw float32 provenance missing: {case['case_id']}")
    return available[0]


def warm(args):
    cases = checked_cases()
    path = OUTPUT / "warm_rows.jsonl"
    rows = indexed_rows(path, cases, args.resume)
    context = model_context()
    context.update(detector=None, device=args.device, repo=args.wam_repo.resolve(strict=True),
                   inference_calls=0, model_load_seconds=0)
    started = time.perf_counter()
    for case in cases:
        if case["case_id"] in rows:
            saved = rows[case["case_id"]]
            entry = context["cache"].get(case["pixel_key"])
            if entry is None or entry["decoded_bits"] != saved["decoded_bits"]:
                raise RuntimeError("completed cache changed during resume")
            if validate_cached(context, case, entry) != saved["match_count"]:
                raise RuntimeError("resume integer count changed")
            continue
        row = warm_case(context, case)
        raw = raw_path(case)
        row.update(raw_logits_relpath=str(raw.relative_to(ROOT)), raw_logits_sha256=sha256_file(raw))
        append_jsonl(path, row)
        rows[case["case_id"]] = row
        if len(rows) % 25 == 0 or len(rows) == 1800:
            atomic_json(OUTPUT / "warm_progress.json", {"completed": len(rows), "total": 1800,
                        "session_inference_calls": context["inference_calls"],
                        "session_seconds": time.perf_counter()-started})
            print(f"PixArt full warm {len(rows)}/1800", flush=True)
    atomic_json(OUTPUT / "warm_summary.json", {
        "cases": len(rows), "all_integer_counts_match": True,
        "inference_rows": sum(r["status"] == "gpu-inference" for r in rows.values()),
        "cache_hit_rows": sum(r["status"] == "cache-hit" for r in rows.values()),
        "session_inference_calls": context["inference_calls"],
        "wall_seconds_current_session": time.perf_counter()-started,
        "torch": package_version("torch"), "numpy": package_version("numpy"), "device": args.device,
        "script_sha256": sha256_file(Path(__file__)),
        "selected_cases_sha256": sha256_file(OUTPUT / "selected_cases.json"),
        "warm_rows_sha256": sha256_file(path), "acceptance_passed": len(rows) == 1800})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("freeze", "warm"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inventory-dir", type=Path, default=ROOT / "results/protocol_v2/pixart_inventory_v1")
    parser.add_argument("--wam-repo", type=Path, default=ROOT / "third_party/watermark-anything")
    args = parser.parse_args()
    try:
        freeze(args) if args.phase == "freeze" else warm(args)
    except Exception as error:
        if OUTPUT.exists():
            atomic_json(OUTPUT / f"failure-{time.time_ns()}.json", {"phase": args.phase, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
