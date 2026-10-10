"""Independent C1 full freeze/warm entry point; preserve old cache and raw pilot.

Inference-start journal entries reserve budget before running the GPU. An
interrupted, incomplete inference is a stop condition, not permission to rerun.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from protocol_v2_cache import PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_cache_warm_v1 import PREPROCESSING_VERSION
from protocol_v2_c1_inventory_v1 import contained_file, expected_identities
from protocol_v2_c1_pilot_v1 import METHODS, RHO_LEVELS, cache_hashes, check_decoded
from protocol_v2_c1_pilot_verify_v1 import require
from protocol_v2_main_curve_full_v1 import append_jsonl, package_version
from protocol_v2_pilot_v1 import Detector, atomic_json, crop_transform, load_json, load_jsonl, make_region_assertion
from protocol_v2_pixart_pilot_v1 import validate_cached

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/protocol_v2/c1_full_v1"
CONFIG = ROOT / "configs/protocol_v2_c1_full_v1.json"
INVENTORY = ROOT / "results/protocol_v2/c1_inventory_v1"
PILOT = ROOT / "results/protocol_v2/c1_migration_pilot_v1"
CACHE = ROOT / "results/protocol_v2/full_pixel_cache_v1/cache"
COUNT = 10800
UNIQUE = 9800
MAX_INFERENCES = 9780
WAM_ENV_ROOT = Path("/root/miniconda3/envs/wam")
CODE_FILES = (
    "protocol_v2_c1_full_v1.py", "protocol_v2_c1_full_audit_v1.py",
    "protocol_v2_c1_full_verify_v1.py", "protocol_v2_c1_full_derived_v1.py",
    "protocol_v2_cache.py", "protocol_v2_pilot_v1.py", "protocol_v2_main_curve_full_v1.py",
    "protocol_v2_c1_pilot_v1.py", "protocol_v2_c1_pilot_audit_v1.py",
    "protocol_v2_c1_pilot_verify_v1.py", "read_actions_v2.py",
)


def validate_config(config):
    fields = ("expected_records", "expected_images", "expected_unique_pixels", "expected_legacy_escapes",
              "max_new_inference_calls", "expected_pilot_raw_reuse", "expected_initial_unique_cache_hits",
              "required_matches", "retention_floor", "suppression_threshold")
    require(tuple(config[k] for k in fields) == (10800, 200, 9800, 2774, 9780, 20, 2012, 24, .05, .5),
            "only the approved unchanged-threshold full C1 cohort is supported")
    require(config["crop_parameter"] == "removed area rho; not per-side c", "rho/c convention mismatch")


def validate_cohort(cases):
    require(len(cases) == COUNT and {r["attack_id"] for r in cases} == expected_identities(), "C1 cohort incomplete")
    require(len({r["case_id"] for r in cases}) == COUNT, "duplicate case IDs")
    require(len({r["pixel_key"] for r in cases}) == UNIQUE, "unexpected unique pixel cohort")
    require(sum(r["legacy_match_count"] < 24 for r in cases) == 2774, "legacy escape cohort changed")
    for row in cases:
        expected = f"{row['image_index']:04d}|{row['method']}|{row['rho_requested']:.3f}"
        require(row["attack_id"] == expected, "C1 identity/geometry association changed")


def model_context():
    return {"output": OUTPUT, "cache": PixelCache(CACHE), "model_hash": sha256_file(ROOT / "ckpts/params.json"),
            "checkpoint_hash": sha256_file(ROOT / "ckpts/wam_mit.pth")}


def source_bindings(config):
    bindings = {INVENTORY / "rows.json": config["inventory_rows_sha256"],
                INVENTORY / "summary.json": config["inventory_summary_sha256"],
                INVENTORY / "mask_aware_selection.json": config["mask_aware_selection_sha256"],
                PILOT / "selected_cases.json": config["pilot_selected_sha256"],
                PILOT / "warm_rows.jsonl": config["pilot_warm_rows_sha256"],
                PILOT / "verification_summary.json": config["pilot_verification_sha256"]}
    for path, expected in bindings.items():
        require(sha256_file(path) == expected, f"accepted input changed: {path}")
    inventory = load_json(INVENTORY / "summary.json")
    for filename, digest in inventory["source_hashes"].items():
        path = ROOT / "results/c1_v1/full" / filename
        require(sha256_file(path) == digest, f"C1 historical source changed: {filename}")
        bindings[path] = digest
    meta = ROOT / "results/defense_e2e/embed_meta.json"
    require(sha256_file(meta) == inventory["embed_meta_sha256"], "embedding metadata changed")
    bindings[meta] = inventory["embed_meta_sha256"]
    require(load_json(PILOT / "verification_summary.json")["acceptance_passed"], "pilot verification required")
    return bindings, inventory


def pilot_raw_index():
    cases = load_json(PILOT / "selected_cases.json")
    warm = {r["case_id"]: r for r in load_jsonl(PILOT / "warm_rows.jsonl")}
    indexed = {}
    for case in cases:
        row = warm[case["case_id"]]
        path = PILOT / "raw_logits" / f"{case['case_id']}.npz"
        require(sha256_file(path) == row["raw_sha256"], "accepted pilot raw array changed")
        indexed[case["pixel_key"]] = {"raw_logits_relpath": str(path.relative_to(ROOT)),
                                    "raw_sha256": row["raw_sha256"], "decoded_bits": row["decoded_bits"]}
    require(len(indexed) == 20, "expected twenty unique accepted pilot raw arrays")
    return indexed


def prepared_parents(metadata):
    parents = {}
    for meta in metadata:
        path = contained_file(ROOT / "results/defense_e2e/watermarked_unsigned", meta["unsigned_image"])
        key, image = decoded_rgb_key(path)
        parents[meta["input_image"]] = {"path": path, "key": key, "image": image, "hash": sha256_file(path)}
    return parents


def frozen_case(row, meta, parent, old):
    source = contained_file(ROOT / "results/c1_v1/full", row["output_relpath"])
    require(sha256_file(source) == row["source_file_sha256"], "frozen C1 pixels changed")
    require(parent["key"] == row["parent_pixel_key"], "unsigned source differs from inventory parent pixels")
    region = make_region_assertion(meta)
    require(region["reference_bits"] == row["reference_bits"], "reference metadata changed")
    transform = None if row["rho_requested"] == 0 else crop_transform(
        row["original_size"], row["crop_box_px"], [row["image_size"]["width"], row["image_size"]["height"]])
    return {**row, "case_id": f"c1_{row['image_index']:04d}_{row['method']}_rho{round(row['rho_requested']*100):03d}",
            "image": row["input_image"], "child_source_relpath": str(source.relative_to(ROOT)),
            "parent_source_relpath": str(parent["path"].relative_to(ROOT)), "parent_file_sha256": parent["hash"],
            "region_assertion": region, "crop_transform": transform,
            "legacy_evidence": old["audit_v2_region_aware"]}


def freeze():
    config = load_json(CONFIG)
    validate_config(config)
    bindings, inventory = source_bindings(config)
    require(not OUTPUT.exists(), "preserve existing full C1 output directory")
    rows = load_json(INVENTORY / "rows.json")
    metadata = load_json(ROOT / "results/defense_e2e/embed_meta.json")["images"][:200]
    meta = {m["input_image"]: m for m in metadata}
    old = {r["attack_id"]: r for r in load_jsonl(ROOT / "results/c1_v1/full/audit_rows.jsonl")}
    parents = prepared_parents(metadata)
    cases = []
    for ordinal, row in enumerate(rows):
        cases.append({**frozen_case(row, meta[row["input_image"]], parents[row["input_image"]], old[row["attack_id"]]),
                      "ordinal": ordinal})
        if (ordinal + 1) % 200 == 0:
            print(f"C1 full input freeze {ordinal+1}/{COUNT}", flush=True)
    validate_cohort(cases)
    pixels, stats = freeze_pixels(cases, pilot_raw_index(), inventory)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    atomic_json(OUTPUT / "selected_cases.json", cases)
    atomic_json(OUTPUT / "unique_pixels.json", pixels)
    summary = {"cases": COUNT, "unique_pixels": UNIQUE, **stats,
               "bindings": {str(p.relative_to(ROOT)): h for p, h in bindings.items()},
               "config_sha256": sha256_file(CONFIG),
               "selected_cases_sha256": sha256_file(OUTPUT / "selected_cases.json"),
               "unique_pixels_sha256": sha256_file(OUTPUT / "unique_pixels.json"),
               "params_sha256": inventory["params_sha256"], "checkpoint_sha256": inventory["checkpoint_sha256"],
               "code_bindings": {name: sha256_file(ROOT / "experiments" / name) for name in CODE_FILES},
               "freeze_script_sha256": sha256_file(Path(__file__)), "legacy_region_metric_consumed": False}
    atomic_json(OUTPUT / "selection_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "bindings"}, indent=2), flush=True)


def freeze_pixels(cases, raw_index, inventory):
    context = model_context()
    require(context["model_hash"] == inventory["params_sha256"]
            and context["checkpoint_hash"] == inventory["checkpoint_sha256"], "model binding changed")
    pixels = {}
    for case in cases:
        prior = pixels.get(case["pixel_key"])
        if prior is not None:
            require(prior["region_assertion"] == case["region_assertion"]
                    and prior["legacy_match_count"] == case["legacy_match_count"],
                    "same pixels, incompatible reference")
            continue
        entry = context["cache"].get(case["pixel_key"])
        if entry is not None:
            require(validate_cached(context, case, entry) == case["legacy_match_count"], "cached integer count drift")
        pixels[case["pixel_key"]] = {**case, "cache_hit": entry is not None,
            "preserved_cache_hashes": cache_hashes(context["cache"], case["pixel_key"]) if entry is not None else None,
            "pilot_raw": raw_index.get(case["pixel_key"])}
    require(len(pixels) == UNIQUE and set(raw_index) <= set(pixels), "pixel/raw cohort mismatch")
    hits = sum(p["cache_hit"] for p in pixels.values())
    require(hits == 2012, "cache coverage changed since approved scope; review before running")
    return list(pixels.values()), {"initial_cache_hits": hits, "initial_cache_misses": UNIQUE-hits,
                                 "pilot_raw_reused": 20, "new_inference_budget": MAX_INFERENCES}


def checked_inputs():
    validate_config(load_json(CONFIG))
    summary = load_json(OUTPUT / "selection_summary.json")
    bindings = {ROOT / p: h for p, h in summary["bindings"].items()}
    bindings.update({ROOT / "experiments" / name: digest for name, digest in summary["code_bindings"].items()})
    bindings.update({CONFIG: summary["config_sha256"], OUTPUT / "selected_cases.json": summary["selected_cases_sha256"],
                     OUTPUT / "unique_pixels.json": summary["unique_pixels_sha256"],
                     ROOT / "ckpts/params.json": summary["params_sha256"],
                     ROOT / "ckpts/wam_mit.pth": summary["checkpoint_sha256"]})
    for path, digest in bindings.items():
        require(sha256_file(path) == digest, f"frozen binding changed: {path}")
    cases, pixels = load_json(OUTPUT / "selected_cases.json"), load_json(OUTPUT / "unique_pixels.json")
    validate_cohort(cases)
    require(len(pixels) == UNIQUE and {p["pixel_key"] for p in pixels} == {c["pixel_key"] for c in cases},
            "unique pixel index changed")
    return cases, pixels


def indexed_rows(path, identities, resume, field="case_id"):
    if path.exists() and not resume:
        raise FileExistsError(f"existing rows require explicit --resume: {path}")
    rows = load_jsonl(path) if path.exists() else []
    indexed = {r[field]: r for r in rows}
    require(len(rows) == len(indexed) and set(indexed) <= set(identities), "duplicate/out-of-cohort resume rows")
    return indexed


def checked_entry(context, case, *, complete=False):
    entry = context["cache"].get(case["pixel_key"])
    if entry is not None:
        require(validate_cached(context, case, entry) == case["legacy_match_count"], "cache integer count drift")
    if case["cache_hit"]:
        require(entry is not None, "old cache missing")
        require(cache_hashes(context["cache"], case["pixel_key"]) == case["preserved_cache_hashes"],
                "old cache modified")
    elif complete:
        require(entry is not None, "completed inference cache missing")
    return entry


def inference_starts(path, identities):
    rows = load_jsonl(path) if path.exists() else []
    keys = [r["pixel_key"] for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) <= set(identities), "invalid inference budget journal")
    require(len(keys) <= MAX_INFERENCES, "inference budget exceeded")
    return set(keys)


def validate_raw(path):
    with np.load(path, allow_pickle=False) as arrays:
        value = arrays["mask_logits"]
        require(value.dtype == np.float32 and value.ndim == 2 and np.isfinite(value).all(), "invalid genuine float32")
        require(np.min(value) >= 0 and np.max(value) <= 1, "raw probabilities outside [0,1]")
    return value


def materialize_cache(context, case, provenance):
    path = contained_file(ROOT, provenance["raw_logits_relpath"])
    require(sha256_file(path) == provenance["raw_sha256"], "raw provenance drift")
    raw = validate_raw(path)
    entry = checked_entry(context, case)
    check_decoded(case, provenance["decoded_bits"], entry)
    if entry is None:
        require(not case["cache_hit"], "cannot rebuild a missing old cache silently")
        require_new_cache_slot(context["cache"], case["pixel_key"])
        context["cache"].put(key=case["pixel_key"], decoded_bits=provenance["decoded_bits"], mask_logits=raw,
            image=case["image_size"], model_sha256=context["model_hash"], checkpoint_sha256=context["checkpoint_hash"],
            preprocessing_version=PREPROCESSING_VERSION)
    checked_entry(context, case, complete=True)


def require_new_cache_slot(cache, key):
    directory = cache.root / key[:2]
    paths = (directory / f"{key}.json", directory / f"{key}.npz",
             directory / f"{key}.json.tmp", directory / f"{key}.tmp.npz")
    require(not any(path.exists() for path in paths), "partial/unindexed cache requires review; preserve files")


def infer_unique(context, case, starts):
    key = case["pixel_key"]
    source = contained_file(ROOT / "results/c1_v1/full", case["output_relpath"])
    require(sha256_file(source) == case["source_file_sha256"] and decoded_rgb_key(source)[0] == key,
            "source pixel drift")
    require(key not in starts and len(starts) < MAX_INFERENCES, "incomplete attempt/budget prevents reinference")
    require(not (OUTPUT / "raw_logits" / f"{key}.npz").exists(), "unindexed raw output requires review")
    if context.get("detector") is None:
        context["detector"] = Detector(wam_repo=context["repo"], params=ROOT / "ckpts/params.json",
                                       checkpoint=ROOT / "ckpts/wam_mit.pth", device=context["device"])
    entry = checked_entry(context, case)
    require(case["cache_hit"] or entry is None, "a previously missing cache was populated outside this run")
    append_jsonl(OUTPUT / "inference_starts.jsonl", {"pixel_key": key, "started_at_ns": time.time_ns()})
    starts.add(key)
    bits, logits = context["detector"].run(source)
    raw_path = OUTPUT / "raw_logits" / f"{key}.npz"
    with raw_path.open("xb") as handle:
        np.savez_compressed(handle, mask_logits=logits)
    # Reserve/output provenance survives failure; no count or reference is adjusted.
    provenance = {"pixel_key": key, "case_id": case["case_id"], "decoded_bits": bits,
                  "raw_logits_relpath": str(raw_path.relative_to(ROOT)), "raw_sha256": sha256_file(raw_path),
                  "cache_hit_before": case["cache_hit"], "status": "new-inference"}
    atomic_json(OUTPUT / "inference_receipts" / f"{key}.json", provenance)
    check_decoded(case, bits, entry)
    return provenance


def obtain_raw(context, case, starts):
    key = case["pixel_key"]
    if case["pilot_raw"] is not None:
        return {**case["pilot_raw"], "pixel_key": key, "case_id": case["case_id"],
                "cache_hit_before": True, "status": "pilot-raw-reuse"}
    receipt = OUTPUT / "inference_receipts" / f"{key}.json"
    if receipt.exists():
        require(context["resume"] and key in starts, "orphan inference requires explicit resume")
        provenance = load_json(receipt)
        require(provenance["pixel_key"] == key and provenance["case_id"] == case["case_id"],
                "receipt identity mismatch")
        return provenance
    return infer_unique(context, case, starts)


def warm(args):
    _, pixels = checked_inputs()
    identities = {p["pixel_key"] for p in pixels}
    path = OUTPUT / "warm_rows.jsonl"
    done = indexed_rows(path, identities, args.resume, "pixel_key")
    starts = inference_starts(OUTPUT / "inference_starts.jsonl", identities)
    context = model_context()
    context.update(repo=args.wam_repo.resolve(strict=True), device=args.device, resume=args.resume, detector=None)
    (OUTPUT / "raw_logits").mkdir(exist_ok=True)
    started = time.perf_counter()
    for case in pixels:
        key = case["pixel_key"]
        if key in done:
            materialize_cache(context, case, done[key])
            continue
        row = obtain_raw(context, case, starts)
        materialize_cache(context, case, row)
        row["match_count"] = case["legacy_match_count"]
        row["comparison_scope"] = "rerun-and-quantization" if case["cache_hit"] else "quantization-only"
        append_jsonl(path, row)
        done[key] = row
        if len(done) % 25 == 0 or len(done) == UNIQUE:
            progress = {"completed_unique_pixels": len(done), "total_unique_pixels": UNIQUE,
                        "new_inference_starts": len(starts), "max_new_inferences": MAX_INFERENCES,
                        "wall_seconds_current_session": time.perf_counter()-started}
            atomic_json(OUTPUT / "warm_progress.json", progress)
            print(f"C1 full warm {len(done)}/{UNIQUE}; GPU attempts {len(starts)}/{MAX_INFERENCES}", flush=True)
    require(len(done) == UNIQUE and len(starts) == MAX_INFERENCES, "warming coverage/budget acceptance failed")
    summary = {"unique_pixels": UNIQUE, "new_inference_calls": len(starts), "pilot_raw_reused": 20,
               "preserved_initial_cache_entries": 2012, "new_cache_entries": 7788,
               "all_integer_counts_match": True, "acceptance_passed": True,
               "warm_rows_sha256": sha256_file(path),
               "inference_starts_sha256": sha256_file(OUTPUT / "inference_starts.jsonl"),
               "wall_seconds_current_session": time.perf_counter()-started, "device": args.device,
               "environment": {n: package_version(n) for n in ("torch", "numpy", "Pillow")},
               "script_sha256": sha256_file(Path(__file__))}
    atomic_json(OUTPUT / "warm_summary.json", summary)


def record_failure(phase, error):
    if OUTPUT.exists():
        atomic_json(OUTPUT / f"failure-{phase}-{time.time_ns()}.json", {"phase": phase, "error": str(error)})


def pipeline(args):
    """Approved stages run sequentially; a failed gate prevents all later stages."""
    require(args.wam_python is not None, "pipeline requires explicit WAM interpreter")
    wam_python = contained_file(WAM_ENV_ROOT, args.wam_python)
    # Keep the venv entry path: resolving its symlink would launch base Python.
    cpu_python = Path(sys.executable).absolute()
    require(cpu_python.is_file(), "current CPU interpreter is unavailable")
    self_script = contained_file(ROOT, Path(__file__).resolve())
    require(not args.resume, "pipeline recovery requires separately reviewed stage --resume commands")
    if not OUTPUT.exists():
        freeze()
    checked_inputs()
    steps = [
        ("warm", [str(wam_python), str(self_script), "--phase", "warm", "--device", args.device]),
        ("audit", [str(cpu_python), str(ROOT / "experiments/protocol_v2_c1_full_audit_v1.py")]),
        ("verify", [str(cpu_python), str(ROOT / "experiments/protocol_v2_c1_full_verify_v1.py")]),
        ("derived", [str(cpu_python), str(ROOT / "experiments/protocol_v2_c1_full_derived_v1.py")]),
    ]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for phase, command in steps:
        atomic_json(OUTPUT / "pipeline_progress.json", {"phase": phase, "state": "running", "time_ns": time.time_ns()})
        print(f"C1 full pipeline starting {phase}", flush=True)
        with (OUTPUT / f"{phase}.log").open("x", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    atomic_json(OUTPUT / "pipeline_progress.json", {"phase": "complete", "state": "automated-acceptance-passed",
                                                    "time_ns": time.time_ns()})
    print("C1 full pipeline all automated stages completed; sync/user acceptance remain", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("freeze", "warm", "pipeline"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wam-repo", type=Path, default=ROOT / "third_party/watermark-anything")
    parser.add_argument("--wam-python", type=Path)
    args = parser.parse_args()
    try:
        {"freeze": freeze, "warm": lambda: warm(args), "pipeline": lambda: pipeline(args)}[args.phase]()
    except Exception as error:
        record_failure(args.phase, error)
        raise


if __name__ == "__main__":
    main()
