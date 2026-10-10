"""Run the full protocol-v2 declared-geometry stress test.

The experiment evaluates two frozen 200-image SDXL cohorts under six declared
crop geometries. Detector bits and spatial logits must already exist in the
exact decoded-RGB cache; this runner performs no neural inference. The
``adaptive-high-logit`` condition is deliberately defense-aware and selects a
false, equal-size crop box that maximizes the verifier's regional logit.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from protocol_v2_cache import PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_common import audit_protocol_v2
from protocol_v2_geometry_lie_v1 import condition_transform, geometry_gate
from protocol_v2_manifest_matrix_full_v1 import resolve_full_cases
from protocol_v2_pilot_v1 import (
    atomic_json,
    load_json,
    load_jsonl,
    logit_mean,
    mapped_mask_and_retention,
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


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> dict[str, float | int | None]:
    """Return a two-sided Wilson interval for a binomial proportion."""
    if successes < 0 or total < 0 or successes > total:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= total")
    if total == 0:
        return {"successes": successes, "total": total, "rate": None, "low": None, "high": None}
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == total else min(1.0, center + half)
    return {
        "successes": successes,
        "total": total,
        "rate": rate,
        "low": low,
        "high": high,
    }


def transition_counts(
    rows: list[dict[str, Any]],
    *,
    baseline_field: str,
    observed_field: str,
) -> dict[str, dict[str, int]]:
    transitions: dict[str, dict[str, int]] = {}
    for row in rows:
        before = str(row[baseline_field])
        after = str(row[observed_field])
        bucket = transitions.setdefault(before, {})
        bucket[after] = bucket.get(after, 0) + 1
    return transitions


def select_cases(
    e2e_root: Path,
    *,
    datasets: list[str],
    expected_per_dataset: int,
    c1_rho: float,
) -> list[dict[str, Any]]:
    all_cases = resolve_full_cases(
        e2e_root,
        c1_rho=c1_rho,
        expected_per_dataset=expected_per_dataset,
    )
    selected = [case for case in all_cases if case["dataset"] in datasets]
    counts: dict[str, int] = defaultdict(int)
    for case in selected:
        counts[case["dataset"]] += 1
        if case["crop_transform"] is None:
            raise RuntimeError(f"selected geometry case has no crop transform: {case['case_id']}")
    expected = {dataset: expected_per_dataset for dataset in datasets}
    if dict(counts) != expected:
        raise RuntimeError(f"selected dataset counts differ from frozen design: {dict(counts)} != {expected}")
    for ordinal, case in enumerate(selected):
        case["geometry_ordinal"] = ordinal
    return selected


def public_selection(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    private = {"region_assertion", "child_source", "parent_source"}
    return [{key: value for key, value in case.items() if key not in private} for case in cases]


def parent_path(output: Path, image_name: str) -> Path:
    return output / "parents" / f"{Path(image_name).stem}.png"


def prepare_parents(
    cases: list[dict[str, Any]],
    *,
    output: Path,
    signer: Any,
    semantics: str,
    resume: bool,
) -> None:
    from protocol_v2_sign_pilot import parent_manifest, sign_file

    by_image: dict[str, dict[str, Any]] = {}
    for case in cases:
        prior = by_image.setdefault(case["image"], case)
        if prior["region_assertion"] != case["region_assertion"]:
            raise RuntimeError(f"region assertion changed across cohorts: {case['image']}")
        if prior["parent_source"] != case["parent_source"]:
            raise RuntimeError(f"parent source changed across cohorts: {case['image']}")
    rows: list[dict[str, Any]] = []
    for index, image_name in enumerate(sorted(by_image), start=1):
        case = by_image[image_name]
        destination = parent_path(output, image_name)
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
        if inspection["reference_source"] != "signed-current":
            raise RuntimeError(f"prepared parent has no signed current reference: {destination}")
        if inspection["chain_semantics"] != "no-ai-disclosure-in-chain":
            raise RuntimeError(f"prepared parent semantics mismatch: {destination}")
        rows.append(
            {
                "image": image_name,
                "signed_parent": str(destination.relative_to(output)),
                "sha256": sha256_file(destination),
            }
        )
        if index % 50 == 0 or index == len(by_image):
            print(f"prepared/verified parents: {index}/{len(by_image)}", flush=True)
    atomic_json(output / "parent_index.json", {"count": len(rows), "rows": rows})


def sign_geometry_child(
    *,
    pixels: Path,
    signed: Path,
    parent: Path,
    declared_transform: dict[str, str] | None,
    missing_geometry: bool,
    signer: Any,
) -> None:
    from protocol_v2_sign_pilot import child_actions, sign_file

    request = {
        "crop_transform": declared_transform,
        "crop_action_without_transform": missing_geometry,
        "child_disclosure": "none",
    }
    manifest = {
        "claim_generator": "protocol_v2_geometry_lie_full/child",
        "format": "image/png",
        "title": signed.name,
        "assertions": [
            {
                "label": "c2pa.actions.v2",
                "data": {"actions": child_actions(request, with_parent=True)},
            }
        ],
    }
    sign_file(pixels, signed, manifest, signer, parent)


def audit_signed(
    inspection: dict[str, Any],
    *,
    cached: dict[str, Any],
    required_matches: int,
) -> tuple[dict[str, Any], float | None, float | None, str]:
    gate = geometry_gate(inspection)
    if gate == "accepted":
        reference = inspection["reference_source"]
        if reference in {"signed-current", "signed-ancestor"}:
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
            reference_state=reference,
            semantics=inspection["chain_semantics"],
            reference_bits=reference_bits,
            decoded_bits=decoded_bits,
            required_matches=required_matches,
            region_retention=retention,
            logit_in_region=regional_logit,
        )
        return audit, retention, regional_logit, gate
    if gate == "not-reached":
        audit = audit_protocol_v2(
            lineage=inspection["lineage_state"],
            reference_state=inspection["reference_source"],
            semantics=inspection["chain_semantics"],
            reference_bits=None,
            decoded_bits=None,
            required_matches=required_matches,
            region_retention=None,
            logit_in_region=None,
        )
        return audit, None, None, gate
    audit = {
        "lineage": inspection["lineage_state"],
        "reference": inspection["reference_source"],
        "semantics": inspection["chain_semantics"],
        "match_count": None,
        "bit_length": None,
        "required_matches": required_matches,
        "evidence": "not-evaluated-invalid-geometry",
        "policy_verdict": "geometry-rejected",
    }
    return audit, None, None, gate


def load_baselines(path: Path, *, condition: str) -> dict[str, dict[str, Any]]:
    rows = [row for row in load_jsonl(path) if row["condition"] == condition]
    baselines = {row["base_case_id"]: row for row in rows}
    if len(baselines) != len(rows):
        raise RuntimeError("duplicate baseline rows")
    return baselines


def correct_matches_baseline(
    audit: dict[str, Any],
    *,
    retention: float | None,
    regional_logit: float | None,
    baseline: dict[str, Any],
) -> bool:
    fields = ("lineage", "reference", "semantics", "evidence", "policy_verdict")
    if any(audit[field] != baseline[field] for field in fields):
        return False
    if retention is None or baseline["region_retention"] is None:
        if retention != baseline["region_retention"]:
            return False
    elif abs(retention - float(baseline["region_retention"])) > 1e-12:
        return False
    if regional_logit is None or baseline["logit_in_region"] is None:
        return regional_logit == baseline["logit_in_region"]
    return abs(regional_logit - float(baseline["logit_in_region"])) <= 1e-6


def run_matrix(
    cases: list[dict[str, Any]],
    *,
    conditions: list[str],
    config: dict[str, Any],
    output: Path,
    cache: PixelCache,
    baselines: dict[str, dict[str, Any]],
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
    shard = [case for case in cases if case["geometry_ordinal"] % num_shards == shard_id]
    total = len(shard) * len(conditions)
    done = len(completed)

    for case in shard:
        case_id = case["case_id"]
        baseline = baselines.get(case_id)
        if baseline is None:
            raise RuntimeError(f"missing frozen baseline: {case_id}")
        pixels = output / "pixels" / case["dataset"] / f"{case['dataset_index']:03d}.png"
        if not pixels.exists():
            pixels.parent.mkdir(parents=True, exist_ok=True)
            sanitize_pixels(Path(case["child_source"]), pixels)
        pixel_key, _ = decoded_rgb_key(pixels)
        if pixel_key != case["pixel_key"]:
            raise RuntimeError(f"sanitizing changed decoded pixels: {case_id}")
        cached = cache.get(pixel_key)
        if cached is None:
            raise RuntimeError(f"exact pixel-cache miss; GPU inference is forbidden: {case_id}")
        parent = parent_path(output, case["image"])
        if not parent.is_file():
            raise FileNotFoundError(f"prepared parent missing: {parent}")

        for condition in conditions:
            pair = (case_id, condition)
            if pair in completed:
                continue
            declared, missing_geometry, adaptive = condition_transform(
                condition,
                case["crop_transform"],
                region=case["region_assertion"],
                mask_logits=cached["mask_logits"],
                config=config,
            )
            run_id = f"{case_id}__{condition}"
            signed = output / "signed" / condition / case["dataset"] / f"{case['dataset_index']:03d}.png"
            request = {
                "run_id": run_id,
                "base_case_id": case_id,
                "dataset": case["dataset"],
                "dataset_index": case["dataset_index"],
                "condition": condition,
                "child_pixels": str(pixels.relative_to(output)),
                "parent": str(parent.relative_to(output)),
                "true_transform": case["crop_transform"],
                "declared_transform": declared,
                "crop_action_without_transform": missing_geometry,
            }
            request_path = output / "requests" / condition / f"{case_id}.json"
            if request_path.exists() and load_json(request_path) != request:
                raise RuntimeError(f"resume request mismatch: {request_path}")
            if not request_path.exists():
                atomic_json(request_path, request)
            if not signed.exists():
                sign_geometry_child(
                    pixels=pixels,
                    signed=signed,
                    parent=parent,
                    declared_transform=declared,
                    missing_geometry=missing_geometry,
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
            audit, retention, regional_logit, gate = audit_signed(
                inspection,
                cached=cached,
                required_matches=int(config["required_matches"]),
            )
            correct_match = None
            if condition == "correct":
                correct_match = correct_matches_baseline(
                    audit,
                    retention=retention,
                    regional_logit=regional_logit,
                    baseline=baseline,
                )
                if not correct_match:
                    raise RuntimeError(f"correct geometry does not reproduce frozen baseline: {case_id}")
            append_jsonl(
                rows_path,
                {
                    "run_id": run_id,
                    "base_case_id": case_id,
                    "dataset": case["dataset"],
                    "dataset_index": case["dataset_index"],
                    "image": case["image"],
                    "condition": condition,
                    "manifest_sha256": sha256_file(signed),
                    "manifest_valid": True,
                    "pixel_cache_key": pixel_key,
                    "pixel_cache_hit": True,
                    "gpu_inference_calls": 0,
                    "true_transform": case["crop_transform"],
                    "declared_transform": declared,
                    "adaptive_selection": adaptive,
                    "geometry_gate": gate,
                    "geometry_status": inspection["geometry"]["status"],
                    "geometry_errors": inspection["geometry"]["errors"],
                    "region_retention": retention,
                    "logit_in_region": regional_logit,
                    "baseline_evidence": baseline["evidence"],
                    "baseline_policy_verdict": baseline["policy_verdict"],
                    "correct_baseline_match": correct_match,
                    **audit,
                },
            )
            completed.add(pair)
            done += 1
            if done % 25 == 0 or done == total:
                print(f"shard {shard_id}: audited {done}/{total}", flush=True)


def condition_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    policy_changes = sum(row["policy_verdict"] != row["baseline_policy_verdict"] for row in rows)
    evidence_changes = sum(row["evidence"] != row["baseline_evidence"] for row in rows)
    new_silent = sum(
        row["policy_verdict"] == "silent-other"
        and row["baseline_policy_verdict"] != "silent-other"
        for row in rows
    )
    review_to_non_review = sum(
        str(row["baseline_policy_verdict"]).startswith("review-")
        and not str(row["policy_verdict"]).startswith("review-")
        for row in rows
    )
    review_to_silent = sum(
        str(row["baseline_policy_verdict"]).startswith("review-")
        and row["policy_verdict"] == "silent-other"
        for row in rows
    )
    logit_gains = [
        float(row["logit_in_region"]) - float(row["baseline_logit_in_region"])
        for row in rows
        if row.get("logit_in_region") is not None and row.get("baseline_logit_in_region") is not None
    ]
    return {
        "count": len(rows),
        "manifest_valid": sum(row["manifest_valid"] is True for row in rows),
        "pixel_cache_hits": sum(row["pixel_cache_hit"] is True for row in rows),
        "geometry_gates": dict(sorted(_counts(row["geometry_gate"] for row in rows).items())),
        "evidence_transitions": transition_counts(
            rows, baseline_field="baseline_evidence", observed_field="evidence"
        ),
        "policy_transitions": transition_counts(
            rows, baseline_field="baseline_policy_verdict", observed_field="policy_verdict"
        ),
        "evidence_changes": wilson_interval(evidence_changes, len(rows)),
        "policy_changes": wilson_interval(policy_changes, len(rows)),
        "new_silent_other": wilson_interval(new_silent, len(rows)),
        "review_to_non_review": wilson_interval(review_to_non_review, len(rows)),
        "review_to_silent_other": wilson_interval(review_to_silent, len(rows)),
        "logit_gain_count": len(logit_gains),
        "logit_gain_mean": statistics.fmean(logit_gains) if logit_gains else None,
        "logit_gain_min": min(logit_gains) if logit_gains else None,
        "logit_gain_max": max(logit_gains) if logit_gains else None,
    }


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return result


def summarize_full(
    output: Path,
    *,
    expected_cases: int,
    conditions: list[str],
    datasets: list[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output.glob("rows.part-*.jsonl")):
        rows.extend(load_jsonl(path))
    unique = {(row["base_case_id"], row["condition"]): row for row in rows}
    expected_rows = expected_cases * len(conditions)
    if len(unique) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} unique rows, observed {len(unique)}")
    ordered = [unique[key] for key in sorted(unique)]

    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in ordered:
        by_case[row["base_case_id"]][row["condition"]] = row
    for case_id, case_rows in by_case.items():
        correct = case_rows.get("correct")
        if correct is None:
            raise RuntimeError(f"correct condition missing: {case_id}")
        for row in case_rows.values():
            row["baseline_logit_in_region"] = correct["logit_in_region"]

    merged = output / "rows.jsonl"
    temporary = merged.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, merged)

    summary: dict[str, Any] = {
        "schema": "protocol-v2-geometry-lie-full-summary-1",
        "case_count": expected_cases,
        "condition_count": len(conditions),
        "row_count": len(ordered),
        "manifest_valid": sum(row["manifest_valid"] is True for row in ordered),
        "pixel_cache_hits": sum(row["pixel_cache_hit"] is True for row in ordered),
        "gpu_inference_calls": sum(int(row["gpu_inference_calls"]) for row in ordered),
        "correct_baseline_matches": sum(
            row["correct_baseline_match"] is True for row in ordered if row["condition"] == "correct"
        ),
        "conditions": {},
        "datasets": {},
    }
    for condition in conditions:
        condition_rows = [row for row in ordered if row["condition"] == condition]
        summary["conditions"][condition] = condition_summary(condition_rows)
    for dataset in datasets:
        dataset_rows = [row for row in ordered if row["dataset"] == dataset]
        summary["datasets"][dataset] = {
            "case_count": len({row["base_case_id"] for row in dataset_rows}),
            "row_count": len(dataset_rows),
            "conditions": {
                condition: condition_summary(
                    [row for row in dataset_rows if row["condition"] == condition]
                )
                for condition in conditions
            },
        }

    missing_rejected = sum(
        row["policy_verdict"] == "geometry-rejected"
        for row in ordered
        if row["condition"] in {"missing", "out-of-bounds"}
    )
    expected_rejections = expected_cases * 2
    checks = {
        "row_count": len(ordered) == expected_rows,
        "manifest_valid": summary["manifest_valid"] == expected_rows,
        "pixel_cache_hits": summary["pixel_cache_hits"] == expected_rows,
        "gpu_inference_calls": summary["gpu_inference_calls"] == 0,
        "correct_baseline_matches": summary["correct_baseline_matches"] == expected_cases,
        "missing_and_invalid_rejected": missing_rejected == expected_rejections,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"full geometry acceptance failed: {failed}")
    summary["acceptance"] = checks
    atomic_json(output / "summary.json", summary)
    return summary


def write_report(output: Path, summary: dict[str, Any], *, datasets: list[str]) -> None:
    lines = [
        "# Protocol v2 full declared-geometry stress test",
        "",
        f"Cases: {summary['case_count']}; rows: {summary['row_count']}; "
        f"valid manifests and exact cache hits: {summary['manifest_valid']}/{summary['row_count']}.",
        f"Correct declarations reproduce {summary['correct_baseline_matches']}/"
        f"{summary['case_count']} frozen manifest-matrix baselines. GPU inference calls: 0.",
        "",
        "The adaptive condition is defense-aware: it uses cached detector logits to select a false "
        "equal-size box. It is an upper-bound stress test, not the paper's primary attacker.",
        "",
    ]
    for dataset in datasets:
        adaptive = summary["datasets"][dataset]["conditions"]["adaptive-high-logit"]
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"Adaptive policy changes: {adaptive['policy_changes']['successes']}/"
                f"{adaptive['policy_changes']['total']}; new silent-other: "
                f"{adaptive['new_silent_other']['successes']}/"
                f"{adaptive['new_silent_other']['total']}; review-to-silent-other: "
                f"{adaptive['review_to_silent_other']['successes']}/"
                f"{adaptive['review_to_silent_other']['total']}.",
                "",
            ]
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/protocol_v2_geometry_lie_full.json"
    )
    parser.add_argument("--e2e-root", type=Path, default=ROOT)
    parser.add_argument("--pixel-cache", type=Path, required=True)
    parser.add_argument("--baseline-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cert", type=Path, default=ROOT / "certs/ec_chain.pem")
    parser.add_argument("--key", type=Path, default=ROOT / "certs/ec_key.pem")
    parser.add_argument("--c1-rho", type=float, default=0.4)
    parser.add_argument("--phase", choices=("parents", "matrix", "summarize", "all"), default="all")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    if args.limit is not None and args.limit_per_dataset is not None:
        raise ValueError("--limit and --limit-per-dataset are mutually exclusive")
    if args.limit_per_dataset is not None and args.limit_per_dataset < 1:
        raise ValueError("--limit-per-dataset must be positive")

    config = load_json(args.config.resolve(strict=True))
    datasets = list(config["datasets"])
    conditions = list(config["conditions"])
    if len(datasets) != int(config["expected_dataset_count"]):
        raise ValueError("dataset count does not match the frozen design")
    if len(conditions) != int(config["expected_condition_count"]):
        raise ValueError("condition count does not match the frozen design")
    e2e_root = args.e2e_root.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = select_cases(
        e2e_root,
        datasets=datasets,
        expected_per_dataset=int(config["expected_per_dataset"]),
        c1_rho=args.c1_rho,
    )
    if len(cases) != int(config["expected_case_count"]):
        raise RuntimeError("case count does not match the frozen design")
    if args.limit_per_dataset is not None:
        cases = [
            case
            for dataset in datasets
            for case in [item for item in cases if item["dataset"] == dataset][
                : args.limit_per_dataset
            ]
        ]
        for ordinal, case in enumerate(cases):
            case["geometry_ordinal"] = ordinal
    elif args.limit is not None:
        cases = cases[: args.limit]
    selection = public_selection(cases)
    selection_path = output / "selected_cases.json"
    if selection_path.exists() and load_json(selection_path) != selection:
        raise RuntimeError("selected cases changed during resume")
    if not selection_path.exists():
        atomic_json(selection_path, selection)

    baselines = load_baselines(
        args.baseline_rows.resolve(strict=True), condition=str(config["manifest_condition"])
    )
    missing_baselines = [case["case_id"] for case in cases if case["case_id"] not in baselines]
    if missing_baselines:
        raise RuntimeError(f"selected cases missing frozen baselines: {missing_baselines[:5]}")

    signer = None
    if args.phase in {"parents", "matrix", "all"}:
        from protocol_v2_sign_pilot import create_signer

        signer = create_signer(args.cert.resolve(strict=True), args.key.resolve(strict=True))
    if args.phase in {"parents", "all"}:
        assert signer is not None
        prepare_parents(
            cases,
            output=output,
            signer=signer,
            semantics=str(config["parent_semantics"]),
            resume=args.resume,
        )
        if args.phase == "parents":
            return
    if args.phase in {"matrix", "all"}:
        assert signer is not None
        run_matrix(
            cases,
            conditions=conditions,
            config=config,
            output=output,
            cache=PixelCache(args.pixel_cache.resolve(strict=True)),
            baselines=baselines,
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
            conditions=conditions,
            datasets=datasets,
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
        write_report(output, summary, datasets=datasets)


if __name__ == "__main__":
    main()
