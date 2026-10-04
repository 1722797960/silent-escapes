"""Run the frozen protocol-v2 manifest-semantics boundary matrix.

The experiment changes only signed manifest topology and disclosure semantics.
Decoded RGB pixels, decoded message bits, and two-dimensional detector logits
come from the completed protocol-v2 boundary-pilot cache.  Loading the detector
is forbidden in this runner.
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
from typing import Any

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


ROOT = Path(__file__).resolve().parents[1]
SIGN_HELPER = ROOT / "experiments" / "protocol_v2_sign_pilot.py"
CHAIN_RETAINED = {
    "honest-current-ai",
    "honest-ancestor-ai-child-omits",
    "issuer-claims-digital-creation",
    "issuer-claims-digital-capture",
}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


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
        str(c2pa_python),
        str(SIGN_HELPER),
        "--request",
        str(request_path),
        "--cert",
        str(cert),
        "--key",
        str(key),
        "--output-root",
        str(output),
    ]
    for root in input_roots:
        command.extend(["--input-root", str(root)])
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False, shell=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"signing failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def expected_policy(*, lineage: str, evidence: str, semantics: str) -> str:
    if lineage == "discontinuity":
        return "discontinuity"
    if evidence == "no-reference":
        return "no-reference"
    if evidence == "message-detected":
        return "consistent-synthetic" if semantics == "ai-disclosed-in-chain" else "integrity-clash"
    if evidence == "region-removed":
        return "review-evidence-removal"
    if evidence == "signal-suppressed":
        return "review-signal-suppression"
    return "silent-other"


def summarize(rows: list[dict[str, Any]], *, expected_rows: int) -> dict[str, Any]:
    if len(rows) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} rows, observed {len(rows)}")
    by_case: dict[str, list[dict[str, Any]]] = {}
    summary: dict[str, Any] = {
        "row_count": len(rows),
        "case_count": len({row["base_case_id"] for row in rows}),
        "condition_count": len({row["condition"] for row in rows}),
        "manifest_valid": sum(row["manifest_valid"] for row in rows),
        "pixel_cache_hits": sum(row["pixel_cache_hit"] for row in rows),
        "gpu_inference_calls": 0,
        "expected_state_matches": sum(row["expected_state_match"] for row in rows),
        "expected_policy_matches": sum(row["expected_policy_match"] for row in rows),
        "evidence_invariant_cases": 0,
        "policy_verdicts_by_condition": {},
        "evidence_states_by_condition": {},
    }
    for row in rows:
        by_case.setdefault(row["base_case_id"], []).append(row)
        for target, field in [
            ("policy_verdicts_by_condition", "policy_verdict"),
            ("evidence_states_by_condition", "evidence"),
        ]:
            condition = row["condition"]
            value = row[field]
            bucket = summary[target].setdefault(condition, {})
            bucket[value] = bucket.get(value, 0) + 1
    for case_rows in by_case.values():
        retained = [row for row in case_rows if row["condition"] in CHAIN_RETAINED]
        if len(retained) != len(CHAIN_RETAINED):
            raise RuntimeError("a case is missing a chain-retained condition")
        if len({row["evidence"] for row in retained}) == 1:
            summary["evidence_invariant_cases"] += 1
    return summary


def validate_summary(summary: dict[str, Any], *, expected_rows: int, expected_cases: int) -> None:
    checks = {
        "row_count": summary["row_count"] == expected_rows,
        "case_count": summary["case_count"] == expected_cases,
        "manifest_valid": summary["manifest_valid"] == expected_rows,
        "pixel_cache_hits": summary["pixel_cache_hits"] == expected_rows,
        "expected_state_matches": summary["expected_state_matches"] == expected_rows,
        "expected_policy_matches": summary["expected_policy_matches"] == expected_rows,
        "evidence_invariant_cases": summary["evidence_invariant_cases"] == expected_cases,
        "gpu_inference_calls": summary["gpu_inference_calls"] == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"manifest-matrix acceptance failed: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/protocol_v2_manifest_matrix.json"
    )
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--e2e-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/protocol_v2/manifest_matrix_pilot_v2",
    )
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
    selected = load_json(pilot / "selected_samples.json")
    base_rows = {row["case_id"]: row for row in load_jsonl(pilot / "audit_rows.jsonl")}
    if len(selected) != int(config["expected_case_count"]):
        raise ValueError("base pilot case count does not match the frozen design")
    conditions = list(config["conditions"])
    if len(conditions) != int(config["expected_condition_count"]):
        raise ValueError("condition count does not match the frozen design")
    cases = selected[: args.limit] if args.limit is not None else selected
    rows_path = output / "rows.jsonl"
    existing = load_jsonl(rows_path) if rows_path.exists() else []
    if existing and not args.resume:
        raise FileExistsError(f"refusing to append to {rows_path}; use --resume")
    completed = {(row["base_case_id"], row["condition"]) for row in existing}
    pixel_cache = PixelCache(pilot / "pixel_cache")
    manifest_cache = ManifestCache(output / "manifest_cache")

    for case in cases:
        case_id = case["case_id"]
        base_request = load_json(pilot / "requests" / f"{case_id}.json")
        pixels = pilot / "pixels" / f"{case_id}.png"
        pixel_key, _ = decoded_rgb_key(pixels)
        cached = pixel_cache.get(pixel_key)
        if cached is None:
            raise RuntimeError(f"pixel cache miss would require forbidden GPU inference: {case_id}")
        for condition in conditions:
            condition_id = condition["id"]
            pair = (case_id, condition_id)
            if pair in completed:
                continue
            run_id = f"{case_id}__{condition_id}"
            signed = output / "signed" / f"{run_id}.png"
            request_path = output / "requests" / f"{run_id}.json"
            request = {
                "case_id": run_id,
                "topology": condition["topology"],
                "manifest_semantics": condition["parent_semantics"],
                "child_disclosure": condition["child_disclosure"],
                "child_pixels": str(pixels),
                "child_output": str(signed),
                "parent_pixels": base_request["parent_pixels"],
                "parent_output": str(output / "parents" / f"{run_id}-parent.png"),
                "crop_transform": base_request.get("crop_transform"),
                "region_assertion": base_request["region_assertion"],
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
                required_matches=int(config["required_matches"]),
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
            manifest_sha = sha256_file(signed)
            cache_payload = {
                "run_id": run_id,
                "pixel_cache_key": pixel_key,
                "lineage": audit["lineage"],
                "reference": audit["reference"],
                "semantics": audit["semantics"],
                "evidence": audit["evidence"],
                "policy_verdict": audit["policy_verdict"],
            }
            previous = manifest_cache.get(manifest_sha)
            if previous is None:
                manifest_cache.put(manifest_sha256=manifest_sha, payload=cache_payload)
            elif any(previous.get(key) != value for key, value in cache_payload.items()):
                raise RuntimeError(f"manifest cache conflict: {run_id}")
            row = {
                "run_id": run_id,
                "base_case_id": case_id,
                "category": case["category"],
                "source": case["source"],
                "condition": condition_id,
                "parent_semantics_requested": condition["parent_semantics"],
                "child_disclosure_requested": condition["child_disclosure"],
                "signed_asset": str(signed.relative_to(output)),
                "manifest_sha256": manifest_sha,
                "manifest_valid": True,
                "pixel_cache_key": pixel_key,
                "pixel_cache_hit": True,
                "gpu_inference_calls": 0,
                "region_retention": retention,
                "logit_in_region": regional_logit,
                "expected_state_match": expected_state_match,
                "expected_policy_match": expected_policy_match,
                "base_evidence": base_rows[case_id]["evidence"],
                "base_policy_verdict": base_rows[case_id]["policy_verdict"],
                **audit,
            }
            append_jsonl(rows_path, row)
            completed.add(pair)
            print(f"audited {run_id} -> {audit['policy_verdict']}", flush=True)

    rows = load_jsonl(rows_path)
    expected_rows = len(cases) * len(conditions)
    summary = summarize(rows, expected_rows=expected_rows)
    if args.limit is None:
        validate_summary(
            summary,
            expected_rows=int(config["expected_row_count"]),
            expected_cases=int(config["expected_case_count"]),
        )
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": package_version("numpy"),
            "pillow": package_version("Pillow"),
            "torch": package_version("torch"),
            "base_pilot_environment": load_json(pilot / "environment.json"),
            "gpu_inference_allowed": False,
            "gpu_inference_calls": 0,
        },
    )
    report = [
        "# Protocol v2 manifest-semantics boundary matrix",
        "",
        f"Cases: {summary['case_count']}; conditions: {summary['condition_count']}; "
        f"rows: {summary['row_count']}; valid manifests: {summary['manifest_valid']}.",
        f"Pixel-cache hits: {summary['pixel_cache_hits']}; GPU inference calls: 0.",
        f"Expected lineage/semantics states: {summary['expected_state_matches']}/"
        f"{summary['row_count']}; expected policy verdicts: "
        f"{summary['expected_policy_matches']}/{summary['row_count']}.",
        f"Evidence is invariant across all chain-retained semantic conditions in "
        f"{summary['evidence_invariant_cases']}/{summary['case_count']} cases.",
        "",
        "This is a deliberate boundary pilot, not a statistical estimate. Intent cannot be "
        "inferred from a missing AI disclosure: issuer omission and an affirmative "
        "digitalCapture claim are therefore reported separately but share the same "
        "no-AI-disclosure policy state.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
