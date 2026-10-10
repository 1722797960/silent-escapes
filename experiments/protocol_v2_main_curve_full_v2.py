"""Re-sign and audit the paper's per-side SDXL crop curve under protocol v2."""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from protocol_v2_cache import PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_common import audit_protocol_v2
from protocol_v2_geometry_lie_v1 import geometry_gate
from protocol_v2_main_curve_cache_v2 import EXPECTED_CROPS
from protocol_v2_main_curve_full_v1 import (
    package_version,
    prepare_parents,
    append_jsonl,
    legacy_evidence,
    parent_path,
    sign_child,
    trust_condition,
    summarize as summarize_v1,
)
from protocol_v2_pilot_v1 import (
    atomic_json, crop_transform, load_json, load_jsonl, make_region_assertion,
    logit_mean, mapped_mask_and_retention, sanitize_pixels,
)
from read_actions_v2 import inspect_asset


ROOT = Path(__file__).resolve().parents[1]


def resolve_cases(
    e2e_root: Path,
    *,
    cache_provenance: Path,
    expected_images: int,
    selection_limit: int | None = None,
) -> list[dict[str, Any]]:
    defense = e2e_root / "results/defense_e2e"
    metadata = load_json(defense / "embed_meta.json")["images"]
    meta_by_image = {row["input_image"]: row for row in metadata}
    index_by_image = {row["input_image"]: int(row["index"]) for row in metadata}
    legacy_rows = load_json(defense / "audit_retention.json")["rows"]
    legacy_by_key = {
        (row["image"], round(float(row["crop_frac"]), 2)): row for row in legacy_rows
    }
    cache_rows = {row["attack_id"]: row for row in load_jsonl(cache_provenance)}
    expected_cases = expected_images * len(EXPECTED_CROPS)
    if len(metadata) != expected_images or len(legacy_rows) != expected_cases:
        raise RuntimeError(
            f"expected {expected_images} images/{expected_cases} audit rows, observed "
            f"{len(metadata)}/{len(legacy_rows)}"
        )
    selected_count = min(selection_limit, expected_cases) if selection_limit is not None else expected_cases
    if len(cache_rows) < selected_count:
        raise RuntimeError(f"expected at least {selected_count} cache rows, observed {len(cache_rows)}")

    cases: list[dict[str, Any]] = []
    ordinal = 0
    for image_name in sorted(index_by_image, key=index_by_image.get):
        if ordinal >= selected_count:
            break
        image_index = index_by_image[image_name]
        meta = meta_by_image[image_name]
        parent = (defense / "watermarked_unsigned" / meta["unsigned_image"]).resolve(strict=True)
        with Image.open(parent) as image:
            source_size = [image.width, image.height]
        if source_size[0] != source_size[1]:
            raise RuntimeError(f"legacy crop code assumes square inputs: {image_name} {source_size}")

        for crop in EXPECTED_CROPS:
            if ordinal >= selected_count:
                break
            attack_id = f"sdxl_per_side_c{round(crop * 100):02d}__{image_index:04d}"
            old = dict(legacy_by_key[(image_name, crop)])
            old["detected"] = float(old["bit_accuracy"]) >= 0.75
            child = (
                defense
                / "attack_resign"
                / f"crop{round(crop * 100):02d}"
                / meta["signed_image"]
            ).resolve(strict=True)
            with Image.open(child) as image:
                output_size = [image.width, image.height]
            margin = int(round(source_size[0] * crop))
            box = [margin, margin, source_size[0] - margin, source_size[1] - margin]
            transform = None if crop == 0 else crop_transform(source_size, box, output_size)
            provenance = cache_rows[attack_id]
            cases.append(
                {
                    "ordinal": ordinal,
                    "case_id": attack_id,
                    "attack_id": attack_id,
                    "image": image_name,
                    "image_index": image_index,
                    # Shared runner field; here it is c, not removed-area rho.
                    "rho_requested": crop,
                    "rho_actual": margin / source_size[0],
                    "crop_frac": crop,
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
            ordinal += 1
    if len(cases) != selected_count or len({case["case_id"] for case in cases}) != selected_count:
        raise RuntimeError("per-side main-curve selection is incomplete or non-unique")
    if len({case["pixel_key"] for case in cases}) != selected_count:
        raise RuntimeError("per-side main curve requires unique decoded pixels")
    return cases


def signed_transform_matches(case: dict[str, Any], inspection: dict[str, Any]) -> bool:
    expected = case["crop_transform"]
    observed = inspection["geometry"]
    if expected is None:
        return observed["status"] == "missing"
    if observed["status"] != "valid":
        return False
    actual = observed["transform"]
    return all(
        actual[key] == [int(value) for value in expected[key].split(",")]
        for key in ("source_size_px", "crop_box_px", "output_size_px")
    ) and actual["resize_filter"] == expected["resize_filter"]


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
            "signed_pixels_unchanged": signed_key == pixel_key,
            "signed_reference_unchanged": inspection["region_assertion"]["reference_bits"]
            == case["region_assertion"]["reference_bits"],
            "signed_transform_unchanged": signed_transform_matches(case, inspection),
            "finite_bounded_evidence": math.isfinite(retention) and 0 <= retention <= 1
            and math.isfinite(regional_logit) and 0 <= regional_logit <= 1,
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
                "crop_frac": case["crop_frac"],
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
                "legacy_region_retention": float(old["region_retention"]),
                "legacy_logit_in_region": float(old["logit_in_region"]),
                "evidence_changed_from_legacy": raw["evidence"] != legacy_evidence(old),
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
    summary = summarize_v1(
        output,
        expected_cases=expected_cases,
        full_acceptance=None,
    )
    rows = load_jsonl(output / "rows.jsonl")
    checks = summary["acceptance"]
    if full_acceptance is not None:
        checks["expected_escapes"] = summary["escapes"] == int(full_acceptance["expected_escapes"])
    checks["escape_partition"] = (
        summary["reviewed_escapes_trusted_editor"] + summary["silent_other_trusted_editor"]
        == summary["escapes"]
    )
    if not all(checks.values()):
        raise RuntimeError(f"v2 acceptance failed: {checks}")
    transitions = Counter((row["legacy_evidence"], row["raw_evidence"]) for row in rows)
    summary["legacy_evidence_transitions"] = [
        {"legacy": old, "v2": new, "count": count}
        for (old, new), count in sorted(transitions.items())
    ]
    summary["evidence_changed_from_legacy"] = sum(row["evidence_changed_from_legacy"] for row in rows)
    summary["legacy_reviewed_escapes"] = sum(
        row["legacy_evidence"] in {"region-removed", "signal-suppressed"} for row in rows
    )
    per_crop: dict[str, Any] = {}
    for crop in sorted({round(float(row["rho_requested"]), 2) for row in rows}):
        subset = [row for row in rows if round(float(row["rho_requested"]), 2) == crop]
        escapes = [row for row in subset if row["raw_evidence"] != "message-detected"]
        per_crop[f"{crop:.2f}"] = {
            "assets": len(subset),
            "escapes": len(escapes),
            "raw_evidence": _counts(row["raw_evidence"] for row in subset),
            "trusted_policy_verdicts": _counts(row["trusted_policy_verdict"] for row in subset),
            "untrusted_policy_verdicts": _counts(row["untrusted_policy_verdict"] for row in subset),
        }
    summary["schema"] = "protocol-v2-per-side-main-curve-full-summary-2"
    summary["crop_parameter"] = "fraction removed from each side"
    summary["per_crop"] = per_crop
    summary.pop("per_rho", None)
    atomic_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/protocol_v2_main_curve_full_v2.json"
    )
    parser.add_argument("--e2e-root", type=Path, default=ROOT)
    parser.add_argument("--cache-provenance", type=Path, required=True)
    parser.add_argument("--pixel-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cert", type=Path, default=ROOT / "certs/ec_chain.pem")
    parser.add_argument("--key", type=Path, default=ROOT / "certs/ec_key.pem")
    parser.add_argument("--phase", choices=("parents", "matrix", "summarize", "all"), default="all")
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
        selection_limit=args.limit,
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
        raise RuntimeError("per-side main-curve selection changed during resume")
    if not selection_path.exists():
        atomic_json(selection_path, public_selection)

    signer = None
    if args.phase in {"parents", "matrix", "all"}:
        from protocol_v2_sign_pilot import create_signer
        signer = create_signer(args.cert.resolve(strict=True), args.key.resolve(strict=True))
    if args.phase in {"parents", "all"}:
        prepare_parents(cases, output=output, signer=signer, resume=args.resume)
        if args.phase == "parents":
            return
    if args.phase in {"matrix", "all"}:
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
        (output / "report.md").write_text(
            "\n".join(
                [
                    "# Protocol v2 SDXL per-side centered-crop main curve",
                    "",
                    f"Cases and valid manifests: {summary['case_count']}/{summary['manifest_valid']}.",
                    f"Escapes: {summary['escapes']}; trusted-editor evidence reviews: "
                    f"{summary['reviewed_escapes_trusted_editor']} ({100 * rate:.1f}%).",
                    f"Trusted-editor silent-other: {summary['silent_other_trusted_editor']}.",
                    "The untrusted-editor policy does not consume declared crop geometry; valid crops "
                    "are routed to review-untrusted-geometry.",
                    "GPU inference calls during manifest evaluation: 0.",
                ]
            ) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
