"""Run the full 1,600-image protocol-v2 manifest-semantics matrix.

All detector outputs must already exist in the exact decoded-RGB pixel cache.
This runner performs C2PA signing, chain inspection, evidence classification,
and disclosure-conditioned policy evaluation without loading the WAM model.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from protocol_v2_cache import PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_cache_inventory_v1 import dataset_paths, mask_aware_rows
from protocol_v2_common import audit_protocol_v2
from protocol_v2_manifest_matrix_v1 import CHAIN_RETAINED, expected_policy
from protocol_v2_pilot_v1 import (
    atomic_json,
    crop_transform,
    load_json,
    load_jsonl,
    logit_mean,
    make_region_assertion,
    mapped_mask_and_retention,
    safe_crop_box,
    sanitize_pixels,
)
from read_actions_v2 import inspect_asset


ROOT = Path(__file__).resolve().parents[1]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def center_crop_box(width: int, height: int, fraction: float) -> list[int]:
    amount_x = int(round(width * fraction))
    amount_y = int(round(height * fraction))
    return [amount_x, amount_y, width - amount_x, height - amount_y]


def _image_size(path: Path) -> list[int]:
    with Image.open(path) as image:
        return [image.width, image.height]


def resolve_full_cases(
    e2e_root: Path,
    *,
    c1_rho: float,
    expected_per_dataset: int,
) -> list[dict[str, Any]]:
    results = e2e_root / "results"
    defense = results / "defense_e2e"
    benign = results / "benign_controls_v1_full"
    centered = results / "benign_centered_crops_v1"
    inpainting = results / "inpainting_e2e_v1_full"
    c1 = results / "c1_v1" / "full"

    source_meta = load_json(defense / "embed_meta.json")["images"]
    meta_by_image = {row["input_image"]: row for row in source_meta}
    benign_rows = load_json(benign / "signed_meta.json")["rows"]
    benign_by_path = {
        (row["condition"], row["unsigned_output"]): row for row in benign_rows
    }
    centered_rows = load_json(centered / "prepare_meta.json")["rows"]
    centered_by_path = {
        (row["condition"], row["signed_image"]): row for row in centered_rows
    }
    inpaint_payload = load_json(inpainting / "signed_meta.json")
    inpaint_rows = inpaint_payload.get("images", inpaint_payload.get("rows", []))
    inpaint_by_path = {row["unsigned_inpainted"]: row for row in inpaint_rows}
    selected_c1 = mask_aware_rows(c1, rho=c1_rho)
    c1_by_path = {
        str((c1 / row["output_relpath"]).resolve()): row for row in selected_c1
    }

    resolved: list[dict[str, Any]] = []
    ordinal = 0
    for dataset, paths in dataset_paths(e2e_root, c1_rho=c1_rho).items():
        if len(paths) != expected_per_dataset:
            raise RuntimeError(
                f"{dataset} expected {expected_per_dataset} images, observed {len(paths)}"
            )
        for dataset_index, path in enumerate(paths):
            pixels = path.resolve(strict=True)
            transform = None
            if dataset in {"identity_resave", "jpeg_q90", "resize_90", "safe_crop_1pct"}:
                row = benign_by_path[(dataset, pixels.name)]
                image_name = row["input_image"]
                if dataset == "safe_crop_1pct":
                    width, height = int(row["width"]), int(row["height"])
                    transform = crop_transform(
                        [width, height],
                        safe_crop_box(row["crop_side"], width, height),
                        [width, height],
                    )
            elif dataset.startswith("centered_crop_"):
                condition = "crop10" if "10pct" in dataset else "crop20"
                row = centered_by_path[(condition, pixels.name)]
                image_name = row["image"]
                parent_size = _image_size(
                    defense / "watermarked_unsigned" / meta_by_image[image_name]["unsigned_image"]
                )
                transform = crop_transform(
                    parent_size,
                    center_crop_box(*parent_size, float(row["crop_frac_per_side"])),
                    _image_size(pixels),
                )
            elif dataset == "lama_inpainting":
                row = inpaint_by_path[pixels.name]
                image_name = row["input_image"]
            elif dataset.startswith("mask_aware_crop_rho_"):
                row = c1_by_path[str(pixels)]
                image_name = row["input_image"]
                transform = crop_transform(
                    row["original_size"], row["crop_box_px"], _image_size(pixels)
                )
            else:
                raise ValueError(f"unknown full-matrix dataset: {dataset}")

            meta = meta_by_image[image_name]
            parent = (defense / "watermarked_unsigned" / meta["unsigned_image"]).resolve(strict=True)
            pixel_key, _ = decoded_rgb_key(pixels)
            resolved.append(
                {
                    "ordinal": ordinal,
                    "case_id": f"{dataset}__{dataset_index:03d}",
                    "dataset": dataset,
                    "dataset_index": dataset_index,
                    "image": image_name,
                    "child_source": str(pixels),
                    "child_source_relpath": str(pixels.relative_to(e2e_root)),
                    "parent_source": str(parent),
                    "parent_source_relpath": str(parent.relative_to(e2e_root)),
                    "pixel_key": pixel_key,
                    "crop_transform": transform,
                    "region_assertion": make_region_assertion(meta),
                }
            )
            ordinal += 1
    if len(resolved) != 8 * expected_per_dataset:
        raise RuntimeError(f"expected {8 * expected_per_dataset} cases, observed {len(resolved)}")
    if len({row["case_id"] for row in resolved}) != len(resolved):
        raise RuntimeError("full-matrix case IDs are not unique")
    if len({row["pixel_key"] for row in resolved}) != len(resolved):
        raise RuntimeError("full-matrix decoded pixel arrays are not unique")
    return resolved


def parent_path(output: Path, image_name: str, semantics: str) -> Path:
    return output / "parents" / semantics / f"{Path(image_name).stem}.png"


def prepare_parents(
    cases: list[dict[str, Any]],
    *,
    output: Path,
    signer: Any,
    resume: bool,
) -> None:
    from protocol_v2_sign_pilot import parent_manifest, sign_file

    by_image: dict[str, dict[str, Any]] = {}
    for case in cases:
        prior = by_image.setdefault(case["image"], case)
        if prior["region_assertion"] != case["region_assertion"]:
            raise RuntimeError(f"region assertion changed across datasets: {case['image']}")
    rows: list[dict[str, Any]] = []
    for image_name in sorted(by_image):
        case = by_image[image_name]
        for semantics in ("ai", "capture", "digital-creation"):
            destination = parent_path(output, image_name, semantics)
            if not destination.exists():
                sign_file(
                    Path(case["parent_source"]),
                    destination,
                    parent_manifest(destination.name, semantics, case["region_assertion"]),
                    signer,
                )
            elif not resume:
                raise FileExistsError(f"refusing to reuse {destination}; pass --resume")
            inspection = inspect_asset(destination)
            if not inspection["manifest_valid"]:
                raise RuntimeError(f"invalid prepared parent: {destination}")
            expected_semantics = (
                "ai-disclosed-in-chain" if semantics == "ai" else "no-ai-disclosure-in-chain"
            )
            if inspection["chain_semantics"] != expected_semantics:
                raise RuntimeError(f"parent semantics mismatch: {destination}")
            rows.append(
                {
                    "image": image_name,
                    "semantics": semantics,
                    "signed_parent": str(destination.relative_to(output)),
                    "sha256": sha256_file(destination),
                }
            )
        if len(rows) % 75 == 0:
            print(f"prepared/verified parents: {len(rows)}/600", flush=True)
    atomic_json(output / "parent_index.json", {"count": len(rows), "rows": rows})


def sign_child(
    *,
    case: dict[str, Any],
    condition: dict[str, Any],
    pixels: Path,
    signed: Path,
    parent: Path | None,
    signer: Any,
) -> None:
    from protocol_v2_sign_pilot import child_actions, created_action, sign_file

    request = {
        "crop_transform": case["crop_transform"],
        "child_disclosure": condition["child_disclosure"],
    }
    if condition["topology"] == "cut-chain":
        actions = [
            created_action(condition["parent_semantics"], "ProtocolV2FreshRoot/1.0"),
            *child_actions(request, with_parent=False),
        ]
        manifest = {
            "claim_generator": "protocol_v2_full/cut-chain",
            "format": "image/png",
            "title": signed.name,
            "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": actions}}],
        }
        sign_file(pixels, signed, manifest, signer)
        return
    if parent is None:
        raise ValueError("ancestor-reference signing requires a parent")
    manifest = {
        "claim_generator": "protocol_v2_full/child",
        "format": "image/png",
        "title": signed.name,
        "assertions": [{
            "label": "c2pa.actions.v2",
            "data": {"actions": child_actions(request, with_parent=True)},
        }],
    }
    sign_file(pixels, signed, manifest, signer, parent)


def run_matrix(
    cases: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    *,
    output: Path,
    cache: PixelCache,
    signer: Any,
    shard_id: int,
    num_shards: int,
    resume: bool,
) -> None:
    rows_path = output / f"rows.part-{shard_id:03d}.jsonl"
    if rows_path.exists() and not resume:
        raise FileExistsError(f"refusing to append to {rows_path}; pass --resume")
    existing = load_jsonl(rows_path) if rows_path.exists() else []
    completed = {(row["base_case_id"], row["condition"]) for row in existing}
    shard = [case for case in cases if case["ordinal"] % num_shards == shard_id]
    total = len(shard) * len(conditions)
    done = len(completed)

    for case in shard:
        pixels = output / "pixels" / f"{case['case_id']}.png"
        sanitize_pixels(Path(case["child_source"]), pixels)
        pixel_key, _ = decoded_rgb_key(pixels)
        if pixel_key != case["pixel_key"]:
            raise RuntimeError(f"sanitizing changed decoded pixels: {case['case_id']}")
        cached = cache.get(pixel_key)
        if cached is None:
            raise RuntimeError(
                f"exact pixel-cache miss; GPU inference is forbidden: {case['case_id']}"
            )

        for condition in conditions:
            condition_id = condition["id"]
            pair = (case["case_id"], condition_id)
            if pair in completed:
                continue
            run_id = f"{case['case_id']}__{condition_id}"
            signed = output / "signed" / condition_id / f"{case['case_id']}.png"
            parent = None
            if condition["topology"] != "cut-chain":
                parent = parent_path(output, case["image"], condition["parent_semantics"])
                if not parent.is_file():
                    raise FileNotFoundError(f"prepared parent missing: {parent}")
            request = {
                "run_id": run_id,
                "base_case_id": case["case_id"],
                "condition": condition,
                "child_pixels": str(pixels.relative_to(output)),
                "parent": str(parent.relative_to(output)) if parent else None,
                "crop_transform": case["crop_transform"],
            }
            request_path = output / "requests" / condition_id / f"{case['case_id']}.json"
            if request_path.exists() and load_json(request_path) != request:
                raise RuntimeError(f"resume request mismatch: {request_path}")
            if not request_path.exists():
                atomic_json(request_path, request)
            if not signed.exists():
                sign_child(
                    case=case,
                    condition=condition,
                    pixels=pixels,
                    signed=signed,
                    parent=parent,
                    signer=signer,
                )
            elif not resume:
                raise FileExistsError(f"refusing to reuse {signed}; pass --resume")

            inspection = inspect_asset(signed)
            if not inspection["manifest_valid"]:
                raise RuntimeError(f"invalid signed manifest: {run_id}")
            signed_key, _ = decoded_rgb_key(signed)
            if signed_key != pixel_key:
                raise RuntimeError(f"signing changed decoded pixels: {run_id}")

            reference_state = inspection["reference_source"]
            if reference_state in {"signed-current", "signed-ancestor"}:
                mapped, retention = mapped_mask_and_retention(
                    inspection["region"], inspection["geometry"]
                )
                regional_logit = logit_mean(cached["mask_logits"], mapped)
                reference_bits = inspection["region_assertion"]["reference_bits"]
                decoded_bits = cached["decoded_bits"]
            else:
                retention = None
                regional_logit = None
                reference_bits = None
                decoded_bits = None
            audit = audit_protocol_v2(
                lineage=inspection["lineage_state"],
                reference_state=reference_state,
                semantics=inspection["chain_semantics"],
                reference_bits=reference_bits,
                decoded_bits=decoded_bits,
                required_matches=24,
                region_retention=retention,
                logit_in_region=regional_logit,
            )
            expected_state_match = (
                audit["lineage"] == condition["expected_lineage"]
                and audit["semantics"] == condition["expected_semantics"]
            )
            expected_verdict = expected_policy(
                lineage=audit["lineage"],
                evidence=audit["evidence"],
                semantics=audit["semantics"],
            )
            expected_policy_match = audit["policy_verdict"] == expected_verdict
            if not expected_state_match or not expected_policy_match:
                raise RuntimeError(
                    f"unexpected protocol state for {run_id}: {audit}; expected={condition}"
                )
            append_jsonl(
                rows_path,
                {
                    "run_id": run_id,
                    "base_case_id": case["case_id"],
                    "dataset": case["dataset"],
                    "dataset_index": case["dataset_index"],
                    "image": case["image"],
                    "condition": condition_id,
                    "manifest_sha256": sha256_file(signed),
                    "manifest_valid": True,
                    "pixel_cache_key": pixel_key,
                    "pixel_cache_hit": True,
                    "gpu_inference_calls": 0,
                    "region_retention": retention,
                    "logit_in_region": regional_logit,
                    "expected_state_match": expected_state_match,
                    "expected_policy_match": expected_policy_match,
                    **audit,
                },
            )
            completed.add(pair)
            done += 1
            if done % 25 == 0 or done == total:
                print(f"shard {shard_id}: audited {done}/{total}", flush=True)


def summarize_full(output: Path, *, expected_cases: int, expected_conditions: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output.glob("rows.part-*.jsonl")):
        rows.extend(load_jsonl(path))
    unique = {(row["base_case_id"], row["condition"]): row for row in rows}
    expected_rows = expected_cases * expected_conditions
    if len(unique) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} unique rows, observed {len(unique)}")
    ordered = [unique[key] for key in sorted(unique)]
    merged = output / "rows.jsonl"
    temporary = merged.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, merged)

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    summary: dict[str, Any] = {
        "schema": "protocol-v2-manifest-matrix-full-summary-1",
        "case_count": expected_cases,
        "condition_count": expected_conditions,
        "row_count": len(ordered),
        "manifest_valid": sum(row["manifest_valid"] for row in ordered),
        "pixel_cache_hits": sum(row["pixel_cache_hit"] for row in ordered),
        "gpu_inference_calls": sum(row["gpu_inference_calls"] for row in ordered),
        "expected_state_matches": sum(row["expected_state_match"] for row in ordered),
        "expected_policy_matches": sum(row["expected_policy_match"] for row in ordered),
        "evidence_invariant_cases": 0,
        "datasets": {},
        "evidence_states_by_condition": {},
        "policy_verdicts_by_condition": {},
    }
    for row in ordered:
        by_case[row["base_case_id"]].append(row)
        dataset = summary["datasets"].setdefault(row["dataset"], {"rows": 0, "cases": set()})
        dataset["rows"] += 1
        dataset["cases"].add(row["base_case_id"])
        for target, field in [
            ("evidence_states_by_condition", "evidence"),
            ("policy_verdicts_by_condition", "policy_verdict"),
        ]:
            bucket = summary[target].setdefault(row["condition"], {})
            value = row[field]
            bucket[value] = bucket.get(value, 0) + 1
    for dataset in summary["datasets"].values():
        dataset["cases"] = len(dataset["cases"])
    for case_rows in by_case.values():
        retained = [row for row in case_rows if row["condition"] in CHAIN_RETAINED]
        if len(retained) == len(CHAIN_RETAINED) and len({row["evidence"] for row in retained}) == 1:
            summary["evidence_invariant_cases"] += 1

    checks = {
        "manifest_valid": summary["manifest_valid"] == expected_rows,
        "pixel_cache_hits": summary["pixel_cache_hits"] == expected_rows,
        "gpu_inference_calls": summary["gpu_inference_calls"] == 0,
        "expected_state_matches": summary["expected_state_matches"] == expected_rows,
        "expected_policy_matches": summary["expected_policy_matches"] == expected_rows,
        "evidence_invariant_cases": summary["evidence_invariant_cases"] == expected_cases,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"full manifest-matrix acceptance failed: {failed}")
    summary["acceptance"] = checks
    atomic_json(output / "summary.json", summary)
    return summary


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/protocol_v2_manifest_matrix.json")
    parser.add_argument("--e2e-root", type=Path, default=ROOT)
    parser.add_argument("--pixel-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cert", type=Path, default=ROOT / "certs/ec_chain.pem")
    parser.add_argument("--key", type=Path, default=ROOT / "certs/ec_key.pem")
    parser.add_argument("--expected-per-dataset", type=int, default=200)
    parser.add_argument("--c1-rho", type=float, default=0.4)
    parser.add_argument("--phase", choices=("parents", "matrix", "summarize", "all"), default="all")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")

    e2e_root = args.e2e_root.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_json(args.config.resolve(strict=True))
    conditions = list(config["conditions"])
    cases = resolve_full_cases(
        e2e_root,
        c1_rho=args.c1_rho,
        expected_per_dataset=args.expected_per_dataset,
    )
    if args.limit is not None:
        cases = cases[: args.limit]
    public_selection = [
        {key: value for key, value in case.items() if key not in {"region_assertion", "child_source", "parent_source"}}
        for case in cases
    ]
    selection_path = output / "selected_cases.json"
    if selection_path.exists() and load_json(selection_path) != public_selection:
        raise RuntimeError("full-matrix selected cases changed during resume")
    if not selection_path.exists():
        atomic_json(selection_path, public_selection)

    signer = None
    if args.phase in {"parents", "matrix", "all"}:
        from protocol_v2_sign_pilot import create_signer

        signer = create_signer(args.cert.resolve(strict=True), args.key.resolve(strict=True))
    if args.phase in {"parents", "all"}:
        assert signer is not None
        prepare_parents(cases, output=output, signer=signer, resume=args.resume)
        if args.phase == "parents":
            return
    if args.phase in {"matrix", "all"}:
        assert signer is not None
        run_matrix(
            cases,
            conditions,
            output=output,
            cache=PixelCache(args.pixel_cache.resolve(strict=True)),
            signer=signer,
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            resume=args.resume,
        )
        if args.phase == "matrix":
            return
    if args.phase in {"summarize", "all"}:
        summary = summarize_full(
            output,
            expected_cases=len(cases),
            expected_conditions=len(conditions),
        )
        atomic_json(
            output / "environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "c2pa": package_version("c2pa-python"),
                "numpy": package_version("numpy"),
                "pillow": package_version("Pillow"),
                "torch": package_version("torch"),
                "gpu_inference_allowed": False,
                "gpu_inference_calls": 0,
            },
        )
        report = [
            "# Protocol v2 full manifest-semantics matrix",
            "",
            f"Cases: {summary['case_count']}; conditions: {summary['condition_count']}; rows: {summary['row_count']}.",
            f"Valid manifests and exact cache hits: {summary['manifest_valid']}/{summary['row_count']}.",
            "Evidence invariance across retained-chain semantics: "
            f"{summary['evidence_invariant_cases']}/{summary['case_count']} cases.",
            "GPU inference calls during this matrix: 0.",
        ]
        (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
