"""Re-evaluate declared-geometry results under an editor-trust policy.

This is a policy-only ablation.  It reads the immutable 2,400-row full
geometry stress result, performs no signing or detector inference, and emits a
cross-product of each row with trusted-editor and untrusted-editor policies.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

from protocol_v2_cache import sha256_file
from protocol_v2_common import geometry_trust_gate, policy_verdict
from protocol_v2_geometry_lie_full_v1 import transition_counts, wilson_interval
from protocol_v2_pilot_v1 import atomic_json, load_json, load_jsonl


ROOT = Path(__file__).resolve().parents[1]


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def evaluate_row(raw: dict[str, Any], *, editor_trust: str) -> dict[str, Any]:
    trust_gate = geometry_trust_gate(
        geometry_gate=raw["geometry_gate"],
        cropped_action_count=1,
        editor_trust=editor_trust,
    )
    if trust_gate == "review-untrusted":
        evidence = "not-evaluated-untrusted-geometry"
        verdict = policy_verdict(
            lineage=raw["lineage"], evidence=evidence, semantics=raw["semantics"]
        )
        pixel_evidence_consumed = False
    else:
        evidence = raw["evidence"]
        verdict = raw["policy_verdict"]
        pixel_evidence_consumed = trust_gate == "trusted"
    baseline_review = str(raw["baseline_policy_verdict"]).startswith("review-")
    return {
        "run_id": f"{raw['run_id']}__{editor_trust}",
        "base_case_id": raw["base_case_id"],
        "dataset": raw["dataset"],
        "dataset_index": raw["dataset_index"],
        "geometry_condition": raw["condition"],
        "editor_trust": editor_trust,
        "manifest_sha256": raw["manifest_sha256"],
        "manifest_valid": raw["manifest_valid"],
        "geometry_gate": raw["geometry_gate"],
        "geometry_trust_gate": trust_gate,
        "pixel_evidence_consumed": pixel_evidence_consumed,
        "raw_evidence": raw["evidence"],
        "raw_policy_verdict": raw["policy_verdict"],
        "baseline_evidence": raw["baseline_evidence"],
        "baseline_policy_verdict": raw["baseline_policy_verdict"],
        "evidence": evidence,
        "policy_verdict": verdict,
        "raw_review_evasion": baseline_review and raw["policy_verdict"] == "silent-other",
        "trust_gated_review_evasion": baseline_review and verdict == "silent-other",
        "gpu_inference_calls": 0,
        "manifest_signing_calls": 0,
    }


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return result


def summarize_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_evasions = sum(row["raw_review_evasion"] for row in rows)
    gated_evasions = sum(row["trust_gated_review_evasion"] for row in rows)
    return {
        "row_count": len(rows),
        "geometry_trust_gates": dict(
            sorted(_counts(row["geometry_trust_gate"] for row in rows).items())
        ),
        "evidence_states": dict(sorted(_counts(row["evidence"] for row in rows).items())),
        "policy_verdicts": dict(
            sorted(_counts(row["policy_verdict"] for row in rows).items())
        ),
        "pixel_evidence_consumed": sum(row["pixel_evidence_consumed"] for row in rows),
        "raw_review_evasions": wilson_interval(raw_evasions, len(rows)),
        "trust_gated_review_evasions": wilson_interval(gated_evasions, len(rows)),
        "raw_to_gated_policy_transitions": transition_counts(
            rows, baseline_field="raw_policy_verdict", observed_field="policy_verdict"
        ),
    }


def summarize(
    rows: list[dict[str, Any]],
    *,
    raw_rows: list[dict[str, Any]],
    trust_conditions: list[str],
    expected_valid: int,
    expected_rejected: int,
    expected_raw_adaptive_evasions: int,
) -> dict[str, Any]:
    datasets = sorted({row["dataset"] for row in rows})
    geometry_conditions = sorted({row["geometry_condition"] for row in rows})
    summary: dict[str, Any] = {
        "schema": "protocol-v2-geometry-trust-ablation-summary-1",
        "input_row_count": len(raw_rows),
        "output_row_count": len(rows),
        "gpu_inference_calls": sum(row["gpu_inference_calls"] for row in rows),
        "manifest_signing_calls": sum(row["manifest_signing_calls"] for row in rows),
        "trust_conditions": {},
        "datasets": {},
    }
    for trust in trust_conditions:
        trust_rows = [row for row in rows if row["editor_trust"] == trust]
        summary["trust_conditions"][trust] = {
            **summarize_subset(trust_rows),
            "geometry_conditions": {
                condition: summarize_subset(
                    [row for row in trust_rows if row["geometry_condition"] == condition]
                )
                for condition in geometry_conditions
            },
        }
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        summary["datasets"][dataset] = {
            trust: summarize_subset(
                [row for row in dataset_rows if row["editor_trust"] == trust]
            )
            for trust in trust_conditions
        }

    trusted = [row for row in rows if row["editor_trust"] == "trusted-editor"]
    untrusted = [row for row in rows if row["editor_trust"] == "untrusted-editor"]
    trusted_exact = sum(
        row["evidence"] == row["raw_evidence"]
        and row["policy_verdict"] == row["raw_policy_verdict"]
        for row in trusted
    )
    untrusted_valid = [row for row in untrusted if row["geometry_gate"] == "accepted"]
    untrusted_rejected = [
        row for row in untrusted if row["geometry_gate"] in {"reject-missing", "reject-invalid"}
    ]
    raw_adaptive_evasions = sum(
        row["condition"] == "adaptive-high-logit"
        and str(row["baseline_policy_verdict"]).startswith("review-")
        and row["policy_verdict"] == "silent-other"
        for row in raw_rows
    )
    checks = {
        "trusted_preserves_raw": trusted_exact == len(raw_rows),
        "untrusted_valid_is_fail_closed": (
            len(untrusted_valid) == expected_valid
            and all(row["policy_verdict"] == "review-untrusted-geometry" for row in untrusted_valid)
            and all(not row["pixel_evidence_consumed"] for row in untrusted_valid)
        ),
        "schema_rejection_precedes_trust": (
            len(untrusted_rejected) == expected_rejected
            and all(row["policy_verdict"] == "geometry-rejected" for row in untrusted_rejected)
        ),
        "raw_adaptive_evasions_preserved": (
            raw_adaptive_evasions == expected_raw_adaptive_evasions
        ),
        "untrusted_review_evasions_zero": not any(
            row["trust_gated_review_evasion"] for row in untrusted
        ),
        "gpu_inference_calls_zero": summary["gpu_inference_calls"] == 0,
        "manifest_signing_calls_zero": summary["manifest_signing_calls"] == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"geometry-trust ablation acceptance failed: {failed}")
    summary["acceptance"] = checks
    return summary


def write_rows(path: Path, rows: list[dict[str, Any]], *, resume: bool) -> None:
    serialized = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    if path.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {path}; pass --resume")
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError("existing policy-ablation rows differ from recomputation")
        return
    temporary = path.with_suffix(".tmp")
    temporary.write_text(serialized, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/protocol_v2_geometry_trust_ablation.json",
    )
    parser.add_argument("--input-rows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config.resolve(strict=True))
    input_path = args.input_rows.resolve(strict=True)
    raw_rows = load_jsonl(input_path)
    if len(raw_rows) != int(config["expected_input_rows"]):
        raise RuntimeError(
            f"expected {config['expected_input_rows']} input rows, observed {len(raw_rows)}"
        )
    raw_keys = {(row["base_case_id"], row["condition"]) for row in raw_rows}
    if len(raw_keys) != len(raw_rows):
        raise RuntimeError("input geometry rows are not unique")
    trust_conditions = list(config["editor_trust_conditions"])
    rows = [
        evaluate_row(raw, editor_trust=trust)
        for raw in raw_rows
        for trust in trust_conditions
    ]
    if len(rows) != int(config["expected_output_rows"]):
        raise RuntimeError("output row count differs from the frozen design")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "rows.jsonl", rows, resume=args.resume)
    summary = summarize(
        rows,
        raw_rows=raw_rows,
        trust_conditions=trust_conditions,
        expected_valid=int(config["expected_valid_geometry_rows_per_trust_condition"]),
        expected_rejected=int(config["expected_schema_rejected_rows_per_trust_condition"]),
        expected_raw_adaptive_evasions=int(config["expected_raw_adaptive_review_evasions"]),
    )
    summary["input_rows_sha256"] = sha256_file(input_path)
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": package_version("numpy"),
            "gpu_inference_allowed": False,
            "gpu_inference_calls": 0,
            "manifest_signing_allowed": False,
            "manifest_signing_calls": 0,
        },
    )
    trusted_adaptive = summary["trust_conditions"]["trusted-editor"]["geometry_conditions"][
        "adaptive-high-logit"
    ]
    untrusted_adaptive = summary["trust_conditions"]["untrusted-editor"][
        "geometry_conditions"
    ]["adaptive-high-logit"]
    report = [
        "# Protocol v2 geometry editor-trust policy ablation",
        "",
        f"Input rows: {summary['input_row_count']}; policy rows: {summary['output_row_count']}.",
        "GPU inference calls: 0; manifest signing calls: 0.",
        "",
        "Trusted editors preserve the raw geometry experiment, including "
        f"{trusted_adaptive['trust_gated_review_evasions']['successes']}/"
        f"{trusted_adaptive['trust_gated_review_evasions']['total']} adaptive review evasions.",
        "Untrusted valid crop geometry is not consumed: adaptive review evasions become "
        f"{untrusted_adaptive['trust_gated_review_evasions']['successes']}/"
        f"{untrusted_adaptive['trust_gated_review_evasions']['total']}, with valid rows routed to "
        "review-untrusted-geometry.",
        "",
        "This is a deployment-policy tradeoff, not cryptographic proof that a trusted editor's "
        "transform is truthful. A compromised trusted editor remains inside the measured raw "
        "attack boundary.",
    ]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
