"""Read protocol-v2 actions and the signed regional reference from one asset.

This helper intentionally accepts only the asset path. It never reads
embed_meta.json or attack_manifest.jsonl, so it can be used to verify that the
SDK spike is self-contained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from protocol_v2_common import (
    action_items,
    chain_semantics,
    lineage_state,
    parent_manifest_labels,
    parent_validation_succeeded,
    resolve_region_reference,
)


REGION_LABEL = "com.example.region_assertion"
TRANSFORM_KEY = "com.example.crop_transform"


def _csv_ints(value: Any, count: int, field: str) -> list[int]:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a CSV string")
    pieces = value.split(",")
    if len(pieces) != count:
        raise ValueError(f"{field} must contain {count} integers")
    try:
        numbers = [int(piece.strip()) for piece in pieces]
    except ValueError as exc:
        raise ValueError(f"{field} contains a non-integer") from exc
    return numbers


def parse_crop_transform(parameters: Any) -> dict[str, Any]:
    """Parse and validate the namespaced crop transform.

    A geometrically false but syntactically valid declaration remains valid
    here: a verifier without the pre-edit pixels cannot infer the real crop.
    The spike runner compares it with controlled ground truth separately.
    """
    if not isinstance(parameters, dict) or TRANSFORM_KEY not in parameters:
        return {"status": "missing", "transform": None, "errors": []}
    raw = parameters[TRANSFORM_KEY]
    if not isinstance(raw, dict):
        return {
            "status": "invalid",
            "transform": None,
            "errors": [f"{TRANSFORM_KEY} must be an object"],
        }
    errors: list[str] = []
    try:
        source_w, source_h = _csv_ints(raw.get("source_size_px"), 2, "source_size_px")
        x0, y0, x1, y1 = _csv_ints(raw.get("crop_box_px"), 4, "crop_box_px")
        output_w, output_h = _csv_ints(raw.get("output_size_px"), 2, "output_size_px")
    except ValueError as exc:
        return {"status": "invalid", "transform": None, "errors": [str(exc)]}
    if raw.get("schema_version") != "1":
        errors.append("unsupported crop-transform schema_version")
    if raw.get("coordinate_system") != "source-pixel-half-open":
        errors.append("unsupported coordinate_system")
    if min(source_w, source_h, output_w, output_h) <= 0:
        errors.append("source and output dimensions must be positive")
    if not (0 <= x0 < x1 <= source_w and 0 <= y0 < y1 <= source_h):
        errors.append("crop_box_px is outside source bounds or has zero area")
    resize_filter = raw.get("resize_filter")
    if resize_filter not in {"NONE", "LANCZOS"}:
        errors.append("resize_filter must be NONE or LANCZOS")
    parsed = {
        "schema_version": raw.get("schema_version"),
        "coordinate_system": raw.get("coordinate_system"),
        "source_size_px": [source_w, source_h],
        "crop_box_px": [x0, y0, x1, y1],
        "output_size_px": [output_w, output_h],
        "resize_filter": resize_filter,
    }
    return {
        "status": "valid" if not errors else "invalid",
        "transform": parsed,
        "errors": errors,
    }


def validate_region_assertion(region: Any) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(region, dict):
        return {
            "status": "missing",
            "reference_bits": None,
            "required_matches": None,
            "payload_hash_ok": None,
            "errors": [],
        }
    reference = region.get("reference_bits")
    if not isinstance(reference, str) or len(reference) != 32 or set(reference) - {"0", "1"}:
        errors.append("reference_bits must contain exactly 32 ASCII binary digits")
        reference = None
    try:
        reference_length = int(region.get("reference_length", ""))
    except (TypeError, ValueError):
        reference_length = -1
    try:
        required_matches = int(region.get("required_matches", ""))
    except (TypeError, ValueError):
        required_matches = -1
    if reference_length != 32:
        errors.append("reference_length must be 32")
    if required_matches != 24:
        errors.append("required_matches must be 24")
    if region.get("schema_version") != "2":
        errors.append("region assertion schema_version must be 2")
    expected_hash = (
        hashlib.sha256(reference.encode("ascii")).hexdigest()
        if reference is not None else None
    )
    payload_hash_ok = (
        region.get("payload_hash") == expected_hash
        if expected_hash is not None else False
    )
    if payload_hash_ok is False:
        errors.append("payload_hash is inconsistent with reference_bits")
    return {
        "status": "valid" if not errors else "invalid",
        "reference_bits": reference,
        "reference_length": reference_length,
        "required_matches": required_matches,
        "payload_hash_ok": payload_hash_ok,
        "errors": errors,
    }


def decoded_rgb_sha256(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        prefix = f"{rgb.width}x{rgb.height}:RGB:".encode("ascii")
        return hashlib.sha256(prefix + rgb.tobytes()).hexdigest()


def inspect_asset(path: Path) -> dict[str, Any]:
    import c2pa

    reader = c2pa.Reader(str(path))
    detailed = json.loads(reader.detailed_json())
    validation_state = reader.get_validation_state()
    manifest_valid = validation_state == "Valid"
    active_label = detailed.get("active_manifest")
    active = detailed.get("manifests", {}).get(active_label, {})
    actions = action_items(active)
    crop_actions = [item for item in actions if item.get("action") == "c2pa.cropped"]
    opened_actions = [item for item in actions if item.get("action") == "c2pa.opened"]
    if len(crop_actions) == 1:
        geometry = parse_crop_transform(crop_actions[0].get("parameters"))
    elif len(crop_actions) == 0:
        geometry = {"status": "missing", "transform": None, "errors": ["c2pa.cropped missing"]}
    else:
        geometry = {"status": "invalid", "transform": None, "errors": ["multiple c2pa.cropped actions"]}
    resolved = resolve_region_reference(detailed, manifest_valid=manifest_valid)
    region_check = validate_region_assertion(resolved.get("region"))
    lineage = lineage_state(
        detailed,
        manifest_valid=manifest_valid,
        reference_state=resolved["state"],
    )
    return {
        "asset": str(path),
        "validation_state": validation_state,
        "manifest_valid": manifest_valid,
        "active_manifest": active_label,
        "manifest_count": len(detailed.get("manifests", {})),
        "parent_of_count": len(parent_manifest_labels(active)),
        "parent_manifest_validated": parent_validation_succeeded(detailed),
        "opened_action_count": len(opened_actions),
        "cropped_action_count": len(crop_actions),
        "actions": actions,
        "geometry": geometry,
        "lineage_state": lineage,
        "reference_source": resolved["state"],
        "reference_manifest": resolved["manifest"],
        "reference_depth": resolved["depth"],
        "reference_errors": resolved["errors"],
        "region": resolved.get("region"),
        "region_assertion": region_check,
        "chain_semantics": chain_semantics(detailed, manifest_valid=manifest_valid),
        "decoded_rgb_sha256": decoded_rgb_sha256(path),
        "validation_results": detailed.get("validation_results", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_asset(args.asset), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
