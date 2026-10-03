"""Stress-test protocol-v2 against missing, invalid, and false crop geometry.

The runner reuses decoded bits and two-dimensional logits from the completed
boundary pilot. It never loads the detector. The adaptive condition is an
explicit defense-aware upper bound: it selects a syntactically valid claimed
crop box that maximizes the verifier's regional logit while retaining at least
the configured fraction of the signed region.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from protocol_v2_cache import ManifestCache, PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_common import audit_protocol_v2
from protocol_v2_pilot_v1 import (
    atomic_json,
    inspect_signed,
    load_json,
    load_jsonl,
    logit_mean,
    mapped_mask_and_retention,
)
from read_actions_v2 import parse_crop_transform


ROOT = Path(__file__).resolve().parents[1]
SIGN_HELPER = ROOT / "experiments" / "protocol_v2_sign_pilot.py"


def csv_ints(value: str, expected: int) -> list[int]:
    parts = [int(piece.strip()) for piece in value.split(",")]
    if len(parts) != expected:
        raise ValueError(f"expected {expected} CSV integers, observed {len(parts)}")
    return parts


def with_box(transform: dict[str, str], box: Iterable[int]) -> dict[str, str]:
    result = dict(transform)
    result["crop_box_px"] = ",".join(str(int(value)) for value in box)
    return result


def shifted_transform(transform: dict[str, str], fraction: float) -> dict[str, str]:
    source_w, source_h = csv_ints(transform["source_size_px"], 2)
    x0, y0, x1, y1 = csv_ints(transform["crop_box_px"], 4)
    crop_w, crop_h = x1 - x0, y1 - y0
    dx = max(1, round(source_w * fraction))
    dy = max(1, round(source_h * fraction))
    new_x0 = min(x0 + dx, source_w - crop_w)
    if new_x0 == x0:
        new_x0 = max(0, x0 - dx)
    new_y0 = min(y0 + dy, source_h - crop_h)
    if new_y0 == y0:
        new_y0 = max(0, y0 - dy)
    if (new_x0, new_y0) == (x0, y0):
        raise ValueError("crop box cannot be shifted inside the declared source")
    return with_box(transform, [new_x0, new_y0, new_x0 + crop_w, new_y0 + crop_h])


def scaled_transform(transform: dict[str, str], factor: float) -> dict[str, str]:
    if not 0 < factor < 1:
        raise ValueError("scale factor must be between zero and one")
    source_w, source_h = csv_ints(transform["source_size_px"], 2)
    x0, y0, x1, y1 = csv_ints(transform["crop_box_px"], 4)
    crop_w, crop_h = x1 - x0, y1 - y0
    new_w = max(1, min(source_w, round(crop_w * factor)))
    new_h = max(1, min(source_h, round(crop_h * factor)))
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    new_x0 = min(max(0, round(center_x - new_w / 2)), source_w - new_w)
    new_y0 = min(max(0, round(center_y - new_h / 2)), source_h - new_h)
    return with_box(transform, [new_x0, new_y0, new_x0 + new_w, new_y0 + new_h])


def out_of_bounds_transform(transform: dict[str, str]) -> dict[str, str]:
    source_w, _ = csv_ints(transform["source_size_px"], 2)
    x0, y0, _, y1 = csv_ints(transform["crop_box_px"], 4)
    return with_box(transform, [x0, y0, source_w + 1, y1])


def is_full_frame_transform(transform: dict[str, str]) -> bool:
    source_w, source_h = csv_ints(transform["source_size_px"], 2)
    return csv_ints(transform["crop_box_px"], 4) == [0, 0, source_w, source_h]


def grid_positions(limit: int, count: int) -> list[int]:
    if limit < 0:
        raise ValueError("grid limit cannot be negative")
    if limit == 0:
        return [0]
    if count < 2:
        raise ValueError("grid count must be at least two")
    return sorted({round(index * limit / (count - 1)) for index in range(count)})


def adaptive_high_logit_transform(
    transform: dict[str, str],
    *,
    region: dict[str, Any],
    mask_logits: Any,
    grid_points: int,
    retention_floor: float,
) -> tuple[dict[str, str], dict[str, Any]]:
    source_w, source_h = csv_ints(transform["source_size_px"], 2)
    true_box = csv_ints(transform["crop_box_px"], 4)
    crop_w, crop_h = true_box[2] - true_box[0], true_box[3] - true_box[1]
    candidates: list[tuple[float, float, list[int], dict[str, str]]] = []
    for y0 in grid_positions(source_h - crop_h, grid_points):
        for x0 in grid_positions(source_w - crop_w, grid_points):
            box = [x0, y0, x0 + crop_w, y0 + crop_h]
            if box == true_box:
                continue
            candidate = with_box(transform, box)
            parsed = parse_crop_transform({"com.example.crop_transform": candidate})
            if parsed["status"] != "valid":
                raise AssertionError(f"generated invalid adaptive candidate: {parsed}")
            mapped, retention = mapped_mask_and_retention(region, parsed)
            score = logit_mean(mask_logits, mapped)
            if retention >= retention_floor:
                candidates.append((score, retention, box, candidate))
    if not candidates:
        raise RuntimeError("no false geometry candidate satisfies the retention floor")
    # Sorting makes ties deterministic: highest score, then highest retention,
    # then lexicographically smallest crop box.
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    score, retention, box, candidate = candidates[0]
    return candidate, {
        "candidate_count": len(candidates),
        "selected_box_px": box,
        "selected_logit": score,
        "selected_retention": retention,
        "selection_uses_detector_logits": True,
    }


def condition_transform(
    condition: str,
    true_transform: dict[str, str],
    *,
    region: dict[str, Any],
    mask_logits: Any,
    config: dict[str, Any],
) -> tuple[dict[str, str] | None, bool, dict[str, Any] | None]:
    if condition == "correct":
        return dict(true_transform), False, None
    if condition == "missing":
        return None, True, None
    if condition == "out-of-bounds":
        return out_of_bounds_transform(true_transform), False, None
    if condition == "shifted":
        return shifted_transform(true_transform, float(config["shift_fraction_of_source"])), False, None
    if condition == "scaled":
        return scaled_transform(true_transform, float(config["scale_factor"])), False, None
    if condition == "adaptive-high-logit":
        transformed, selection = adaptive_high_logit_transform(
            true_transform,
            region=region,
            mask_logits=mask_logits,
            grid_points=int(config["adaptive_grid_points_per_axis"]),
            retention_floor=float(config["retention_floor"]),
        )
        return transformed, False, selection
    raise ValueError(f"unknown geometry condition: {condition}")


def run_signer(
    request_path: Path,
    *,
    output: Path,
    input_roots: list[Path],
    c2pa_python: Path,
    cert: Path,
    key: Path,
) -> None:
    command = [
        str(c2pa_python), str(SIGN_HELPER), "--request", str(request_path),
        "--cert", str(cert), "--key", str(key), "--output-root", str(output),
    ]
    for root in input_roots:
        command.extend(["--input-root", str(root)])
    completed = subprocess.run(command, capture_output=True, text=True, check=False, shell=False)
    if completed.returncode != 0:
        raise RuntimeError(f"signing failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def geometry_gate(inspection: dict[str, Any]) -> str:
    if inspection["lineage_state"] in {"discontinuity", "no-manifest"}:
        return "not-reached"
    geometry = inspection["geometry"]
    if geometry["status"] == "invalid":
        return "reject-invalid"
    if geometry["status"] == "missing" and inspection["cropped_action_count"] == 1:
        return "reject-missing"
    return "accepted"


def summarize(rows: list[dict[str, Any]], *, expected_rows: int) -> dict[str, Any]:
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, observed {len(rows)}")
    summary: dict[str, Any] = {
        "row_count": len(rows),
        "case_count": len({row["base_case_id"] for row in rows}),
        "gpu_inference_calls": 0,
        "pixel_cache_hits": sum(row["pixel_cache_hit"] for row in rows),
        "signed_manifest_count": sum(row["manifest_valid"] is True for row in rows),
        "not_applicable_count": sum(not row["condition_applicable"] for row in rows),
        "correct_baseline_matches": sum(row["correct_baseline_match"] is True for row in rows),
        "correct_baseline_comparisons": sum(row["correct_baseline_match"] is not None for row in rows),
        "geometry_gates": {},
        "geometry_statuses": {},
        "policy_verdicts_by_condition": {},
        "adaptive_new_silent_other": 0,
    }
    by_pair = {(row["base_case_id"], row["condition"]): row for row in rows}
    geometry_challenges = [
        row for row in rows
        if row["condition"] in {"missing", "out-of-bounds"}
        and row["geometry_gate"] != "not-reached"
    ]
    summary["malformed_geometry_challenges"] = len(geometry_challenges)
    summary["malformed_geometry_rejected"] = sum(
        row["policy_verdict"] == "geometry-rejected" for row in geometry_challenges
    )
    adaptive_gains: list[float] = []
    adaptive_policy_changes = 0
    for row in rows:
        gate = row["geometry_gate"]
        status = row["geometry_status"]
        summary["geometry_gates"][gate] = summary["geometry_gates"].get(gate, 0) + 1
        summary["geometry_statuses"][status] = summary["geometry_statuses"].get(status, 0) + 1
        condition = row["condition"]
        verdict = row["policy_verdict"]
        bucket = summary["policy_verdicts_by_condition"].setdefault(condition, {})
        bucket[verdict] = bucket.get(verdict, 0) + 1
        if (
            condition == "adaptive-high-logit"
            and verdict == "silent-other"
            and row["baseline_policy_verdict"] != "silent-other"
        ):
            summary["adaptive_new_silent_other"] += 1
        if (
            condition == "adaptive-high-logit"
            and row["condition_applicable"]
            and row["geometry_gate"] == "accepted"
        ):
            correct = by_pair[(row["base_case_id"], "correct")]
            if row["logit_in_region"] is not None and correct["logit_in_region"] is not None:
                adaptive_gains.append(row["logit_in_region"] - correct["logit_in_region"])
            if verdict != row["baseline_policy_verdict"]:
                adaptive_policy_changes += 1
    summary["adaptive_policy_changes"] = adaptive_policy_changes
    summary["adaptive_logit_comparisons"] = len(adaptive_gains)
    summary["adaptive_logit_gain_mean"] = (
        sum(adaptive_gains) / len(adaptive_gains) if adaptive_gains else None
    )
    summary["adaptive_logit_gain_min"] = min(adaptive_gains) if adaptive_gains else None
    summary["adaptive_logit_gain_max"] = max(adaptive_gains) if adaptive_gains else None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/protocol_v2_geometry_lie.json")
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--e2e-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/protocol_v2/geometry_lie_v1")
    parser.add_argument("--c2pa-python", type=Path, required=True)
    parser.add_argument("--cert", type=Path, default=ROOT / "certs/ec_chain.pem")
    parser.add_argument("--key", type=Path, default=ROOT / "certs/ec_key.pem")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    pilot = args.pilot_dir.resolve(strict=True)
    e2e_root = args.e2e_root.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_selection = {row["case_id"]: row for row in load_json(pilot / "selected_samples.json")}
    base_rows = {row["case_id"]: row for row in load_jsonl(pilot / "audit_rows.jsonl")}
    requested_ids = list(config["case_ids"])
    if len(requested_ids) != int(config["expected_case_count"]):
        raise ValueError("configured case count does not match expected_case_count")
    missing_ids = [case_id for case_id in requested_ids if case_id not in base_selection]
    if missing_ids:
        raise ValueError(f"cases missing from base pilot: {missing_ids}")
    selected_ids = requested_ids[: args.limit] if args.limit is not None else requested_ids
    conditions = list(config["conditions"])
    rows_path = output / "rows.jsonl"
    existing = load_jsonl(rows_path) if rows_path.exists() else []
    if existing and not args.resume:
        raise FileExistsError(f"refusing to append to {rows_path}; use --resume")
    completed = {(row["base_case_id"], row["condition"]) for row in existing}
    pixel_cache = PixelCache(pilot / "pixel_cache")
    manifest_cache = ManifestCache(output / "manifest_cache")

    for case_id in selected_ids:
        selection = base_selection[case_id]
        true_transform = selection.get("crop_transform")
        if true_transform is None:
            raise ValueError(f"geometry study case has no real crop transform: {case_id}")
        base_request = load_json(pilot / "requests" / f"{case_id}.json")
        pixels = pilot / "pixels" / f"{case_id}.png"
        pixel_key, _ = decoded_rgb_key(pixels)
        cached = pixel_cache.get(pixel_key)
        if cached is None:
            raise RuntimeError(f"pixel cache miss would require forbidden GPU inference: {case_id}")
        region = base_request["region_assertion"]
        for condition in conditions:
            pair = (case_id, condition)
            if pair in completed:
                continue
            if (
                is_full_frame_transform(true_transform)
                and condition in config["full_frame_not_applicable_conditions"]
            ):
                row = {
                    "run_id": f"{case_id}__{condition}",
                    "base_case_id": case_id,
                    "condition": condition,
                    "condition_applicable": False,
                    "not_applicable_reason": "full-frame crop has no alternative equal-size position",
                    "category": selection["category"],
                    "topology": base_request["topology"],
                    "signed_asset": None,
                    "manifest_sha256": None,
                    "manifest_valid": None,
                    "pixel_cache_key": pixel_key,
                    "pixel_cache_hit": True,
                    "gpu_inference_calls": 0,
                    "true_transform": true_transform,
                    "declared_transform": None,
                    "adaptive_selection": None,
                    "geometry_gate": "not-applicable",
                    "geometry_status": "not-applicable",
                    "geometry_errors": [],
                    "region_retention": None,
                    "logit_in_region": None,
                    "baseline_policy_verdict": base_rows[case_id]["policy_verdict"],
                    "correct_baseline_match": None,
                    "lineage": base_rows[case_id]["lineage"],
                    "reference": base_rows[case_id]["reference"],
                    "semantics": base_rows[case_id]["semantics"],
                    "match_count": None,
                    "bit_length": None,
                    "required_matches": int(config["required_matches"]),
                    "evidence": "not-applicable",
                    "policy_verdict": "not-applicable",
                }
                with rows_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                completed.add(pair)
                print(f"recorded {row['run_id']} -> not-applicable", flush=True)
                continue
            declared, missing_geometry, adaptive = condition_transform(
                condition,
                true_transform,
                region=region,
                mask_logits=cached["mask_logits"],
                config=config,
            )
            run_id = f"{case_id}__{condition}"
            signed = output / "signed" / f"{run_id}.png"
            request_path = output / "requests" / f"{run_id}.json"
            request = {
                "case_id": run_id,
                "topology": base_request["topology"],
                "manifest_semantics": base_request["manifest_semantics"],
                "child_pixels": str(pixels),
                "child_output": str(signed),
                "parent_pixels": base_request["parent_pixels"],
                "parent_output": str(output / "parents" / f"{case_id}-parent.png"),
                "crop_transform": declared,
                "crop_action_without_transform": missing_geometry,
                "region_assertion": region,
            }
            request_path.parent.mkdir(parents=True, exist_ok=True)
            if request_path.exists() and load_json(request_path) != request:
                raise RuntimeError(f"resume request mismatch: {request_path}")
            if not request_path.exists():
                atomic_json(request_path, request)
            if not signed.exists():
                run_signer(
                    request_path,
                    output=output,
                    input_roots=[pilot, e2e_root],
                    c2pa_python=args.c2pa_python,
                    cert=args.cert,
                    key=args.key,
                )
            inspection = inspect_signed(signed, args.c2pa_python)
            if not inspection["manifest_valid"]:
                raise RuntimeError(f"invalid signed manifest: {run_id}")
            signed_key, _ = decoded_rgb_key(signed)
            if signed_key != pixel_key:
                raise RuntimeError(f"signing changed decoded pixels: {run_id}")
            gate = geometry_gate(inspection)
            if gate == "accepted":
                reference = inspection["reference_source"]
                if reference in {"signed-current", "signed-ancestor"}:
                    mapped, retention = mapped_mask_and_retention(inspection["region"], inspection["geometry"])
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
                    reference_state=reference,
                    semantics=inspection["chain_semantics"],
                    reference_bits=reference_bits,
                    decoded_bits=decoded_bits,
                    required_matches=int(config["required_matches"]),
                    region_retention=retention,
                    logit_in_region=regional_logit,
                )
            elif gate == "not-reached":
                retention = None
                regional_logit = None
                audit = audit_protocol_v2(
                    lineage=inspection["lineage_state"],
                    reference_state=inspection["reference_source"],
                    semantics=inspection["chain_semantics"],
                    reference_bits=None,
                    decoded_bits=None,
                    required_matches=int(config["required_matches"]),
                    region_retention=None,
                    logit_in_region=None,
                )
            else:
                retention = None
                regional_logit = None
                audit = {
                    "lineage": inspection["lineage_state"],
                    "reference": inspection["reference_source"],
                    "semantics": inspection["chain_semantics"],
                    "match_count": None,
                    "bit_length": None,
                    "required_matches": int(config["required_matches"]),
                    "evidence": "not-evaluated-invalid-geometry",
                    "policy_verdict": "geometry-rejected",
                }
            baseline = base_rows[case_id]
            correct_match = None
            if condition == "correct":
                correct_match = all(
                    audit[key] == baseline[key]
                    for key in ("lineage", "reference", "semantics", "evidence", "policy_verdict")
                )
                if correct_match and retention is not None:
                    correct_match = abs(retention - baseline["region_retention"]) <= 1e-12
                if correct_match and regional_logit is not None:
                    correct_match = abs(regional_logit - baseline["logit_in_region"]) <= 1e-6
                if not correct_match:
                    raise RuntimeError(f"correct geometry does not reproduce baseline: {case_id}")
            manifest_sha = sha256_file(signed)
            cache_payload = {
                "run_id": run_id,
                "geometry_gate": gate,
                "geometry_status": inspection["geometry"]["status"],
                "policy_verdict": audit["policy_verdict"],
                "pixel_cache_key": pixel_key,
            }
            existing_manifest = manifest_cache.get(manifest_sha)
            if existing_manifest is None:
                manifest_cache.put(manifest_sha256=manifest_sha, payload=cache_payload)
            elif any(existing_manifest.get(key) != value for key, value in cache_payload.items()):
                raise RuntimeError(f"manifest cache conflict: {run_id}")
            row = {
                "run_id": run_id,
                "base_case_id": case_id,
                "condition": condition,
                "condition_applicable": True,
                "not_applicable_reason": None,
                "category": selection["category"],
                "topology": base_request["topology"],
                "signed_asset": str(signed.relative_to(output)),
                "manifest_sha256": manifest_sha,
                "manifest_valid": True,
                "pixel_cache_key": pixel_key,
                "pixel_cache_hit": True,
                "gpu_inference_calls": 0,
                "true_transform": true_transform,
                "declared_transform": declared,
                "adaptive_selection": adaptive,
                "geometry_gate": gate,
                "geometry_status": inspection["geometry"]["status"],
                "geometry_errors": inspection["geometry"]["errors"],
                "region_retention": retention,
                "logit_in_region": regional_logit,
                "baseline_policy_verdict": baseline["policy_verdict"],
                "correct_baseline_match": correct_match,
                **audit,
            }
            with rows_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            completed.add(pair)
            print(f"audited {run_id} -> {audit['policy_verdict']}", flush=True)

    rows = load_jsonl(rows_path)
    expected_rows = len(selected_ids) * len(conditions)
    summary = summarize(rows, expected_rows=expected_rows)
    if args.limit is None:
        if summary["signed_manifest_count"] != int(config["expected_signed_manifest_count"]):
            raise RuntimeError("signed manifest count does not match the frozen design")
        if summary["not_applicable_count"] != expected_rows - int(config["expected_signed_manifest_count"]):
            raise RuntimeError("not-applicable count does not match the frozen design")
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "environment.json", {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": package_version("numpy"),
        "pillow": package_version("Pillow"),
        "torch": package_version("torch"),
        "base_pilot_environment": load_json(pilot / "environment.json"),
        "gpu_inference_allowed": False,
        "gpu_inference_calls": 0,
    })
    report = [
        "# Protocol v2 declared-geometry stress test",
        "",
        f"Cases: {summary['case_count']}; rows: {summary['row_count']}; "
        f"signed manifests: {summary['signed_manifest_count']}; "
        f"not applicable: {summary['not_applicable_count']}; "
        f"pixel-cache hits: {summary['pixel_cache_hits']}; GPU inference calls: 0.",
        "",
        "This is a deliberate protocol stress test, not a statistical estimate.",
        f"Correct declarations reproduce {summary['correct_baseline_matches']}/"
        f"{summary['correct_baseline_comparisons']} baseline verdicts. Missing or invalid "
        f"geometry is rejected in {summary['malformed_geometry_rejected']}/"
        f"{summary['malformed_geometry_challenges']} cases that reach the geometry gate.",
        f"The defense-aware adaptive condition creates {summary['adaptive_new_silent_other']} "
        f"new silent-other verdicts and changes {summary['adaptive_policy_changes']} policy verdicts.",
        "The adaptive condition maximizes the regional logit among false equal-size boxes on "
        "the configured grid and is reported as an upper-bound attack.",
        "Equal-size shifted and adaptive declarations are not applicable to three full-frame controls.",
        "A cut-chain case is retained as a negative control: the lineage gate precedes geometry.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
