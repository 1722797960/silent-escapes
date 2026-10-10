"""Re-sign and audit the complete SDXL centered-crop curve under protocol v2.

This runner is detector-free.  It consumes the exact 1,800-entry pixel cache,
places the 32-bit reference and regional assertion in a signed parent manifest,
reads crop geometry back from each signed child manifest, and evaluates both
trusted-editor and untrusted-editor deployment policies.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from protocol_v2_cache import PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_common import audit_protocol_v2, geometry_trust_gate, policy_verdict
from protocol_v2_geometry_lie_v1 import geometry_gate
from protocol_v2_main_curve_cache_v1 import centered_curve_rows
from protocol_v2_pilot_v1 import (
    atomic_json,
    crop_transform,
    load_json,
    load_jsonl,
    logit_mean,
    make_region_assertion,
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


def resolve_cases(
    e2e_root: Path,
    *,
    cache_provenance: Path,
    expected_images: int,
) -> list[dict[str, Any]]:
    c1 = e2e_root / "results/c1_v1/full"
    defense = e2e_root / "results/defense_e2e"
    attacks = centered_curve_rows(c1)
    legacy = {row["attack_id"]: row for row in load_jsonl(c1 / "audit_rows.jsonl")}
    cache_rows = {row["attack_id"]: row for row in load_jsonl(cache_provenance)}
    metadata = load_json(defense / "embed_meta.json")["images"]
    meta_by_image = {row["input_image"]: row for row in metadata}
    expected_cases = expected_images * 9
    if len(attacks) != expected_cases or len(cache_rows) != expected_cases:
        raise RuntimeError(
            f"expected {expected_cases} attacks/cache rows, observed "
            f"{len(attacks)}/{len(cache_rows)}"
        )

    cases: list[dict[str, Any]] = []
    for ordinal, attack in enumerate(attacks):
        attack_id = attack["attack_id"]
        old = legacy[attack_id]
        if old["output_relpath"] != attack["output_relpath"]:
            raise RuntimeError(f"legacy path mismatch: {attack_id}")
        image_name = attack["input_image"]
        meta = meta_by_image[image_name]
        child = (c1 / attack["output_relpath"]).resolve(strict=True)
        parent = (defense / "watermarked_unsigned" / meta["unsigned_image"]).resolve(strict=True)
        with Image.open(child) as image:
            output_size = [image.width, image.height]
        rho = float(attack["rho_requested"])
        transform = None
        if rho > 0:
            transform = crop_transform(
                attack["original_size"], attack["crop_box_px"], output_size
            )
        provenance = cache_rows[attack_id]
        case_id = f"sdxl_center_rho_{round(rho * 100):02d}__{int(attack['image_index']):04d}"
        cases.append(
            {
                "ordinal": ordinal,
                "case_id": case_id,
                "attack_id": attack_id,
                "image": image_name,
                "image_index": int(attack["image_index"]),
                "rho_requested": rho,
                "rho_actual": float(attack["rho_actual"]),
                "child_source": str(child),
                "child_source_relpath": str(child.relative_to(e2e_root)),
                "parent_source": str(parent),
                "parent_source_relpath": str(parent.relative_to(e2e_root)),
                "pixel_key": provenance["pixel_key"],
                "crop_transform": transform,
                "region_assertion": make_region_assertion(meta),
                "legacy": old,
            }
        )
    if len({case["case_id"] for case in cases}) != len(cases):
        raise RuntimeError("main-curve case IDs are not unique")
    if len({case["pixel_key"] for case in cases}) != len(cases):
        raise RuntimeError("main-curve decoded pixels are not unique")
    return cases


def parent_path(output: Path, image_index: int) -> Path:
    return output / "parents" / "capture" / f"{image_index:04d}.png"


def prepare_parents(
    cases: list[dict[str, Any]],
    *,
    output: Path,
    signer: Any,
    resume: bool,
) -> None:
    from protocol_v2_sign_pilot import parent_manifest, sign_file

    by_index: dict[int, dict[str, Any]] = {}
    for case in cases:
        prior = by_index.setdefault(case["image_index"], case)
        if prior["region_assertion"] != case["region_assertion"]:
            raise RuntimeError(f"region assertion changed for image {case['image_index']}")
    rows: list[dict[str, Any]] = []
    for image_index in sorted(by_index):
        case = by_index[image_index]
        destination = parent_path(output, image_index)
        if not destination.exists():
            sign_file(
                Path(case["parent_source"]),
                destination,
                parent_manifest(destination.name, "capture", case["region_assertion"]),
                signer,
            )
        elif not resume:
            raise FileExistsError(f"refusing to reuse {destination}; pass --resume")
        inspection = inspect_asset(destination)
        checks = {
            "manifest_valid": inspection["manifest_valid"],
            "reference_current": inspection["reference_source"] == "signed-current",
            "capture_semantics": inspection["chain_semantics"] == "no-ai-disclosure-in-chain",
        }
        if not all(checks.values()):
            raise RuntimeError(f"prepared parent failed validation: {image_index}: {checks}")
        rows.append(
            {
                "image_index": image_index,
                "image": case["image"],
                "signed_parent": str(destination.relative_to(output)),
                "sha256": sha256_file(destination),
            }
        )
        if len(rows) % 25 == 0 or len(rows) == len(by_index):
            print(f"prepared/verified parents: {len(rows)}/{len(by_index)}", flush=True)
    atomic_json(output / "parent_index.json", {"count": len(rows), "rows": rows})


def sign_child(
    case: dict[str, Any],
    *,
    pixels: Path,
    signed: Path,
    parent: Path,
    signer: Any,
) -> None:
    from protocol_v2_sign_pilot import child_actions, sign_file

    request = {
        "crop_transform": case["crop_transform"],
        "child_disclosure": "omitted",
    }
    manifest = {
        "claim_generator": "protocol_v2_main_curve_full/child",
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


def legacy_evidence(old: dict[str, Any]) -> str:
    if bool(old["detected"]):
        return "message-detected"
    if float(old["region_retention"]) < 0.05:
        return "region-removed"
    if float(old["logit_in_region"]) < 0.5:
        return "signal-suppressed"
    return "silent-other"


def trust_condition(
    raw: dict[str, Any],
    *,
    gate: str,
    cropped_action_count: int,
    editor_trust: str,
) -> dict[str, Any]:
    trust_gate = geometry_trust_gate(
        geometry_gate=gate,
        cropped_action_count=cropped_action_count,
        editor_trust=editor_trust,
    )
    if trust_gate == "review-untrusted":
        evidence = "not-evaluated-untrusted-geometry"
        verdict = policy_verdict(
            lineage=raw["lineage"], evidence=evidence, semantics=raw["semantics"]
        )
        consumed = False
    elif trust_gate == "schema-rejected":
        evidence = "not-evaluated-invalid-geometry"
        verdict = "geometry-rejected"
        consumed = False
    else:
        evidence = raw["evidence"]
        verdict = raw["policy_verdict"]
        consumed = True
    return {
        "geometry_trust_gate": trust_gate,
        "pixel_evidence_consumed": consumed,
        "evidence": evidence,
        "policy_verdict": verdict,
    }


def run_shard(
    cases: list[dict[str, Any]],
    *,
    output: Path,
    cache: PixelCache,
    signer: Any,
    required_matches: int,
    shard_id: int,
    num_shards: int,
    resume: bool,
) -> None:
    rows_path = output / f"rows.part-{shard_id:03d}.jsonl"
    if rows_path.exists() and not resume:
        raise FileExistsError(f"refusing to append to {rows_path}; pass --resume")
    existing = load_jsonl(rows_path) if rows_path.exists() else []
    completed = {row["case_id"] for row in existing}
    shard = [case for case in cases if case["ordinal"] % num_shards == shard_id]
    done = len(completed)

    for case in shard:
        if case["case_id"] in completed:
            continue
        pixels = output / "pixels" / f"{case['case_id']}.png"
        signed = output / "signed" / f"{case['case_id']}.png"
        parent = parent_path(output, case["image_index"])
        if not parent.is_file():
            raise FileNotFoundError(f"prepared parent missing: {parent}")
        sanitize_pixels(Path(case["child_source"]), pixels)
        pixel_key, _ = decoded_rgb_key(pixels)
        if pixel_key != case["pixel_key"]:
            raise RuntimeError(f"sanitizing changed decoded pixels: {case['case_id']}")
        cached = cache.get(pixel_key)
        if cached is None:
            raise RuntimeError(f"exact pixel-cache miss; GPU is forbidden: {case['case_id']}")

        request = {
            "case_id": case["case_id"],
            "attack_id": case["attack_id"],
            "child_pixels": str(pixels.relative_to(output)),
            "parent": str(parent.relative_to(output)),
            "crop_transform": case["crop_transform"],
            "manifest_condition": "issuer-claims-digital-capture",
        }
        request_path = output / "requests" / f"{case['case_id']}.json"
        if request_path.exists() and load_json(request_path) != request:
            raise RuntimeError(f"resume request mismatch: {request_path}")
        if not request_path.exists():
            atomic_json(request_path, request)
        if not signed.exists():
            sign_child(case, pixels=pixels, signed=signed, parent=parent, signer=signer)
        elif not resume:
            raise FileExistsError(f"refusing to reuse {signed}; pass --resume")

        inspection = inspect_asset(signed)
        signed_key, _ = decoded_rgb_key(signed)
        gate = geometry_gate(inspection)
        if gate != "accepted":
            raise RuntimeError(f"unexpected geometry gate for {case['case_id']}: {gate}")
        mapped, retention = mapped_mask_and_retention(
            inspection["region"], inspection["geometry"]
        )
        regional_logit = logit_mean(cached["mask_logits"], mapped)
        raw = audit_protocol_v2(
            lineage=inspection["lineage_state"],
            reference_state=inspection["reference_source"],
            semantics=inspection["chain_semantics"],
            reference_bits=inspection["region_assertion"]["reference_bits"],
            decoded_bits=cached["decoded_bits"],
            required_matches=required_matches,
            region_retention=retention,
            logit_in_region=regional_logit,
        )
        trusted = trust_condition(
            raw,
            gate=gate,
            cropped_action_count=inspection["cropped_action_count"],
            editor_trust="trusted-editor",
        )
        untrusted = trust_condition(
            raw,
            gate=gate,
            cropped_action_count=inspection["cropped_action_count"],
            editor_trust="untrusted-editor",
        )
        old = case["legacy"]
        legacy_match_count = int(round(float(old["bit_accuracy"]) * 32))
        expected_geometry = (
            inspection["cropped_action_count"] == (0 if case["rho_requested"] == 0 else 1)
            and inspection["geometry"]["status"]
            == ("missing" if case["rho_requested"] == 0 else "valid")
        )
        state_checks = {
            "manifest_valid": inspection["manifest_valid"],
            "lineage": raw["lineage"] == "ancestor",
            "reference": raw["reference"] == "signed-ancestor",
            "semantics": raw["semantics"] == "no-ai-disclosure-in-chain",
            "geometry": expected_geometry,
            "legacy_match_count": raw["match_count"] == legacy_match_count,
            "legacy_evidence": raw["evidence"] == legacy_evidence(old),
            "legacy_retention": abs(retention - float(old["region_retention"])) <= 1e-8,
            "legacy_suppression_class": (regional_logit < 0.5)
            == (float(old["logit_in_region"]) < 0.5),
        }
        if not all(state_checks.values()):
            raise RuntimeError(
                f"protocol-v2 main-curve acceptance failed for {case['case_id']}: "
                f"{state_checks}"
            )
        append_jsonl(
            rows_path,
            {
                "case_id": case["case_id"],
                "attack_id": case["attack_id"],
                "image": case["image"],
                "image_index": case["image_index"],
                "rho_requested": case["rho_requested"],
                "rho_actual": case["rho_actual"],
                "manifest_sha256": sha256_file(signed),
                "manifest_valid": inspection["manifest_valid"],
                "cropped_action_count": inspection["cropped_action_count"],
                "geometry_status": inspection["geometry"]["status"],
                "geometry_gate": gate,
                "pixel_cache_key": pixel_key,
                "pixel_cache_hit": True,
                "gpu_inference_calls": 0,
                "region_retention": retention,
                "logit_in_region": regional_logit,
                "raw_evidence": raw["evidence"],
                "raw_policy_verdict": raw["policy_verdict"],
                "trusted_geometry_gate": trusted["geometry_trust_gate"],
                "trusted_pixel_evidence_consumed": trusted["pixel_evidence_consumed"],
                "trusted_evidence": trusted["evidence"],
                "trusted_policy_verdict": trusted["policy_verdict"],
                "untrusted_geometry_gate": untrusted["geometry_trust_gate"],
                "untrusted_pixel_evidence_consumed": untrusted["pixel_evidence_consumed"],
                "untrusted_evidence": untrusted["evidence"],
                "untrusted_policy_verdict": untrusted["policy_verdict"],
                "legacy_match_count": legacy_match_count,
                "legacy_evidence": legacy_evidence(old),
                "state_checks": state_checks,
                **{key: raw[key] for key in (
                    "lineage", "reference", "semantics", "match_count",
                    "bit_length", "required_matches",
                )},
            },
        )
        completed.add(case["case_id"])
        done += 1
        if done % 25 == 0 or done == len(shard):
            print(f"shard {shard_id}: signed/audited {done}/{len(shard)}", flush=True)


def _counts(values: Any) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def summarize(
    output: Path,
    *,
    expected_cases: int,
    full_acceptance: dict[str, int] | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(output.glob("rows.part-*.jsonl")):
        rows.extend(load_jsonl(path))
    unique = {row["case_id"]: row for row in rows}
    if len(rows) != expected_cases or len(unique) != expected_cases:
        raise RuntimeError(
            f"expected {expected_cases} unique rows, observed {len(rows)}/{len(unique)}"
        )
    ordered = [unique[key] for key in sorted(unique)]
    merged = output / "rows.jsonl"
    temporary = merged.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, merged)

    escapes = [row for row in ordered if row["raw_evidence"] != "message-detected"]
    reviewed = [
        row
        for row in escapes
        if row["trusted_policy_verdict"]
        in {"review-evidence-removal", "review-signal-suppression"}
    ]
    silent = [row for row in escapes if row["trusted_policy_verdict"] == "silent-other"]
    per_rho: dict[str, Any] = {}
    for rho in sorted({float(row["rho_requested"]) for row in ordered}):
        subset = [row for row in ordered if float(row["rho_requested"]) == rho]
        rho_escapes = [row for row in subset if row["raw_evidence"] != "message-detected"]
        per_rho[f"{rho:.1f}"] = {
            "assets": len(subset),
            "escapes": len(rho_escapes),
            "raw_evidence": _counts(row["raw_evidence"] for row in subset),
            "trusted_policy_verdicts": _counts(
                row["trusted_policy_verdict"] for row in subset
            ),
            "untrusted_policy_verdicts": _counts(
                row["untrusted_policy_verdict"] for row in subset
            ),
        }
    checks = {
        "manifest_valid": sum(row["manifest_valid"] for row in ordered) == expected_cases,
        "pixel_cache_hits": sum(row["pixel_cache_hit"] for row in ordered) == expected_cases,
        "gpu_inference_zero": sum(row["gpu_inference_calls"] for row in ordered) == 0,
        "state_checks": all(all(row["state_checks"].values()) for row in ordered),
    }
    if full_acceptance is not None:
        checks.update(
            {
                "expected_escapes": len(escapes)
                == int(full_acceptance["expected_escapes"]),
                "expected_reviewed_escapes": len(reviewed)
                == int(full_acceptance["expected_reviewed_escapes"]),
                "expected_silent_other": len(silent)
                == int(full_acceptance["expected_silent_other"]),
            }
        )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"main-curve full acceptance failed: {failed}")
    summary = {
        "schema": "protocol-v2-main-curve-full-summary-1",
        "case_count": expected_cases,
        "manifest_valid": sum(row["manifest_valid"] for row in ordered),
        "pixel_cache_hits": sum(row["pixel_cache_hit"] for row in ordered),
        "gpu_inference_calls": 0,
        "escapes": len(escapes),
        "reviewed_escapes_trusted_editor": len(reviewed),
        "silent_other_trusted_editor": len(silent),
        "escape_review_rate_trusted_editor": len(reviewed) / len(escapes) if escapes else None,
        "raw_evidence": _counts(row["raw_evidence"] for row in ordered),
        "trusted_policy_verdicts": _counts(row["trusted_policy_verdict"] for row in ordered),
        "untrusted_policy_verdicts": _counts(row["untrusted_policy_verdict"] for row in ordered),
        "per_rho": per_rho,
        "acceptance": checks,
    }
    atomic_json(output / "summary.json", summary)
    return summary


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/protocol_v2_main_curve_full.json"
    )
    parser.add_argument("--e2e-root", type=Path, default=ROOT)
    parser.add_argument("--cache-provenance", type=Path, required=True)
    parser.add_argument("--pixel-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cert", type=Path, default=ROOT / "certs/ec_chain.pem")
    parser.add_argument("--key", type=Path, default=ROOT / "certs/ec_key.pem")
    parser.add_argument(
        "--phase", choices=("parents", "matrix", "summarize", "all"), default="all"
    )
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")

    config = load_json(args.config.resolve(strict=True))
    e2e_root = args.e2e_root.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = resolve_cases(
        e2e_root,
        cache_provenance=args.cache_provenance.resolve(strict=True),
        expected_images=int(config["expected_images"]),
    )
    if args.limit is not None:
        cases = cases[: args.limit]
    public_selection = [
        {
            key: value
            for key, value in case.items()
            if key not in {"region_assertion", "legacy", "child_source", "parent_source"}
        }
        for case in cases
    ]
    selection_path = output / "selected_cases.json"
    if selection_path.exists() and load_json(selection_path) != public_selection:
        raise RuntimeError("main-curve selected cases changed during resume")
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
        run_shard(
            cases,
            output=output,
            cache=PixelCache(args.pixel_cache.resolve(strict=True)),
            signer=signer,
            required_matches=int(config["required_matches"]),
            shard_id=args.shard_id,
            num_shards=args.num_shards,
            resume=args.resume,
        )
        if args.phase == "matrix":
            return
    if args.phase in {"summarize", "all"}:
        summary = summarize(
            output,
            expected_cases=len(cases),
            full_acceptance=config["full_acceptance"] if args.limit is None else None,
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
        rate = summary["escape_review_rate_trusted_editor"]
        report = [
            "# Protocol v2 SDXL centered-crop main curve",
            "",
            f"Cases and valid manifests: {summary['case_count']}/{summary['manifest_valid']}.",
            f"Escapes: {summary['escapes']}; trusted-editor evidence reviews: "
            f"{summary['reviewed_escapes_trusted_editor']} ({100 * rate:.1f}%).",
            f"Trusted-editor silent-other: {summary['silent_other_trusted_editor']}.",
            "The untrusted-editor policy does not consume declared crop geometry; valid crops are "
            "routed to review-untrusted-geometry.",
            "GPU inference calls during manifest evaluation: 0.",
        ]
        (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
