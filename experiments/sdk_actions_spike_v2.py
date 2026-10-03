"""C2PA protocol-v2 compatibility spike.

Creates a tiny signed parent plus four child assets and checks whether
c2pa-python preserves the action parameters and signed reference. No model or
GPU is required. Outputs are isolated under results/protocol_v2/sdk_spike.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import c2pa
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from PIL import Image

from read_actions_v2 import decoded_rgb_sha256, inspect_asset


ROOT = Path(__file__).resolve().parents[1]
REGION_LABEL = "com.example.region_assertion"
PARENT_LABEL = "parent-source"
TRANSFORM_KEY = "com.example.crop_transform"
REFERENCE_BITS = "01011010010110100101101001011010"
ACTUAL_BOX = [8, 4, 56, 44]
SOURCE_SIZE = [64, 48]
OUTPUT_SIZE = [48, 40]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_signer(cert_path: Path, key_path: Path) -> c2pa.Signer:
    key = serialization.load_pem_private_key(
        key_path.read_bytes(), password=None, backend=default_backend()
    )
    return c2pa.Signer.from_callback(
        callback=lambda data: key.sign(data, ec.ECDSA(hashes.SHA256())),
        alg=c2pa.C2paSigningAlg.ES256,
        certs=cert_path.read_text(encoding="utf-8"),
        tsa_url=None,
    )


def region_assertion() -> dict[str, str]:
    # A deterministic all-ones 64x64 mask is enough to test serialization.
    bitmap = base64.b64encode(bytes([255]) * (64 * 64 // 8)).decode("ascii")
    return {
        "schema_version": "2",
        "alg": "wam_mit_regional",
        "reference_bits": REFERENCE_BITS,
        "reference_length": "32",
        "required_matches": "24",
        "payload_hash": hashlib.sha256(REFERENCE_BITS.encode("ascii")).hexdigest(),
        "bbox_csv": "0.125000,0.083333,0.875000,0.916667",
        "bitmap64": bitmap,
    }


def transform(box: list[int] | None) -> dict[str, str] | None:
    if box is None:
        return None
    return {
        "schema_version": "1",
        "coordinate_system": "source-pixel-half-open",
        "source_size_px": ",".join(map(str, SOURCE_SIZE)),
        "crop_box_px": ",".join(map(str, box)),
        "output_size_px": ",".join(map(str, OUTPUT_SIZE)),
        "resize_filter": "NONE" if box == ACTUAL_BOX else "LANCZOS",
    }


def generate_source(path: Path) -> None:
    image = Image.new("RGB", tuple(SOURCE_SIZE))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = ((x * 3 + y) % 256, (x + y * 5) % 256, (x * 7 + y * 11) % 256)
    image.save(path)


def crop_source(source: Path, output: Path) -> None:
    with Image.open(source) as image:
        image.convert("RGB").crop(tuple(ACTUAL_BOX)).save(output)


def sign_parent(source: Path, output: Path, signer: c2pa.Signer) -> None:
    manifest = {
        "claim_generator": "sdk_actions_spike_v2/parent",
        "format": "image/png",
        "title": output.name,
        "assertions": [
            {
                "label": "c2pa.actions.v2",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
                            "softwareAgent": "ProtocolV2Source/1.0",
                        }
                    ]
                },
            },
            {"label": REGION_LABEL, "data": region_assertion()},
        ],
    }
    with c2pa.Builder(manifest) as builder:
        builder.sign_file(source_path=str(source), dest_path=str(output), signer=signer)


def sign_child(
    parent: Path,
    pixels: Path,
    output: Path,
    signer: c2pa.Signer,
    declared_box: list[int] | None,
    propagate_region: bool,
) -> None:
    parameters: dict[str, Any] = {"ingredientIds": [PARENT_LABEL]}
    encoded_transform = transform(declared_box)
    if encoded_transform is not None:
        parameters[TRANSFORM_KEY] = encoded_transform
    actions = [
        {
            "action": "c2pa.opened",
            "softwareAgent": "ProtocolV2Editor/1.0",
            "parameters": {"ingredientIds": [PARENT_LABEL]},
        },
        {
            "action": "c2pa.cropped",
            "softwareAgent": "ProtocolV2Editor/1.0",
            "parameters": parameters,
        },
    ]
    assertions: list[dict[str, Any]] = [
        {"label": "c2pa.actions.v2", "data": {"actions": actions}}
    ]
    if propagate_region:
        assertions.append({"label": REGION_LABEL, "data": region_assertion()})
    manifest = {
        "claim_generator": "sdk_actions_spike_v2/child",
        "format": "image/png",
        "title": output.name,
        "assertions": assertions,
    }
    with c2pa.Builder(manifest) as builder:
        with parent.open("rb") as parent_stream:
            builder.add_ingredient(
                {
                    "title": parent.name,
                    "relationship": "parentOf",
                    "label": PARENT_LABEL,
                },
                "image/png",
                parent_stream,
            )
        builder.sign_file(source_path=str(pixels), dest_path=str(output), signer=signer)


def subprocess_inspect(asset: Path) -> dict[str, Any]:
    helper = Path(__file__).with_name("read_actions_v2.py")
    completed = subprocess.run(
        [sys.executable, str(helper), str(asset)],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    return json.loads(completed.stdout)


def compare_readers(direct: dict[str, Any], separate: dict[str, Any]) -> bool:
    keys = [
        "validation_state",
        "manifest_count",
        "parent_of_count",
        "opened_action_count",
        "cropped_action_count",
        "geometry",
        "reference_source",
        "region_assertion",
        "decoded_rgb_sha256",
    ]
    return all(direct.get(key) == separate.get(key) for key in keys)


def geometry_matches_ground_truth(result: dict[str, Any]) -> bool | None:
    geometry = result["geometry"]
    if geometry["status"] == "missing":
        return None
    if geometry["status"] != "valid":
        return False
    declared = geometry["transform"]
    return (
        declared["source_size_px"] == SOURCE_SIZE
        and declared["crop_box_px"] == ACTUAL_BOX
        and declared["output_size_px"] == OUTPUT_SIZE
    )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def package_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "protocol_v2" / "sdk_spike",
    )
    parser.add_argument("--cert", type=Path, default=ROOT / "certs" / "ec_chain.pem")
    parser.add_argument("--key", type=Path, default=ROOT / "certs" / "ec_key.pem")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    if rows_path.exists():
        raise SystemExit(f"refusing to overwrite existing spike results: {rows_path}")
    for required in (args.cert, args.key):
        if not required.exists():
            raise SystemExit(f"missing required file: {required}")

    source = output_dir / "source.png"
    parent = output_dir / "parent_signed.png"
    cropped_pixels = output_dir / "cropped_pixels.png"
    generate_source(source)
    crop_source(source, cropped_pixels)
    signer = create_signer(args.cert, args.key)
    sign_parent(source, parent, signer)

    conditions = [
        ("parent_assertion_v2", parent, None, None, "signed-current"),
        ("child_correct", output_dir / "child_correct.png", ACTUAL_BOX, False, "signed-ancestor"),
        ("child_missing_geometry", output_dir / "child_missing.png", None, False, "signed-ancestor"),
        ("child_identity_lie", output_dir / "child_identity_lie.png", [0, 0, 64, 48], False, "signed-ancestor"),
        ("child_propagated", output_dir / "child_propagated.png", ACTUAL_BOX, True, "signed-current"),
    ]
    rows: list[dict[str, Any]] = []
    for name, asset, declared_box, propagate, expected_reference in conditions:
        if name != "parent_assertion_v2":
            sign_child(parent, cropped_pixels, asset, signer, declared_box, bool(propagate))
        direct = inspect_asset(asset)
        separate = subprocess_inspect(asset)
        input_pixels = source if name == "parent_assertion_v2" else cropped_pixels
        row = {
            "condition": name,
            "asset": str(asset),
            "manifest_valid": direct["manifest_valid"],
            "reference_source": direct["reference_source"],
            "expected_reference_source": expected_reference,
            "reference_source_ok": direct["reference_source"] == expected_reference,
            "reference_roundtrip_ok": direct["region_assertion"]["status"] == "valid",
            "geometry_status": direct["geometry"]["status"],
            "geometry": direct["geometry"],
            "geometry_matches_controlled_ground_truth": (
                None if name == "parent_assertion_v2" else geometry_matches_ground_truth(direct)
            ),
            "parent_of_count": direct["parent_of_count"],
            "opened_action_count": direct["opened_action_count"],
            "cropped_action_count": direct["cropped_action_count"],
            "decoded_pixels_unchanged_by_signing": (
                decoded_rgb_sha256(input_pixels) == direct["decoded_rgb_sha256"]
            ),
            "separate_reader_process_agrees": compare_readers(direct, separate),
        }
        rows.append(row)
        append_jsonl(rows_path, row)

    by_name = {row["condition"]: row for row in rows}
    passed = all(row["manifest_valid"] for row in rows)
    passed &= all(row["reference_source_ok"] for row in rows)
    passed &= all(row["reference_roundtrip_ok"] for row in rows)
    passed &= all(row["decoded_pixels_unchanged_by_signing"] for row in rows)
    passed &= all(row["separate_reader_process_agrees"] for row in rows)
    passed &= by_name["child_correct"]["geometry_status"] == "valid"
    passed &= by_name["child_correct"]["geometry_matches_controlled_ground_truth"] is True
    passed &= by_name["child_missing_geometry"]["geometry_status"] == "missing"
    passed &= by_name["child_identity_lie"]["geometry_status"] == "valid"
    passed &= by_name["child_identity_lie"]["geometry_matches_controlled_ground_truth"] is False
    passed &= by_name["child_propagated"]["geometry_matches_controlled_ground_truth"] is True

    summary = {
        "passed": bool(passed),
        "condition_count": len(rows),
        "valid_manifest_count": sum(row["manifest_valid"] for row in rows),
        "reference_roundtrip_count": sum(row["reference_roundtrip_ok"] for row in rows),
        "separate_reader_agreement_count": sum(row["separate_reader_process_agrees"] for row in rows),
        "important_boundary": (
            "The controlled harness detects the identity lie by comparing with known generation "
            "geometry. The asset-only verifier cannot prove that a syntactically valid declaration is truthful."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "c2pa_python": package_version("c2pa-python", "c2pa"),
        "pillow": package_version("Pillow"),
        "cryptography": package_version("cryptography"),
        "cert_sha256": sha256_file(args.cert),
        "git_metadata_available": (ROOT / ".git").exists(),
        "gpu_used": False,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# Protocol v2 SDK spike",
        "",
        f"Overall: **{'PASS' if passed else 'FAIL'}**",
        "",
        "| Condition | Manifest | Reference | Geometry | Ground truth | Pixels | Reader agreement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['manifest_valid']} | "
            f"{row['reference_source']} | {row['geometry_status']} | "
            f"{row['geometry_matches_controlled_ground_truth']} | "
            f"{row['decoded_pixels_unchanged_by_signing']} | "
            f"{row['separate_reader_process_agrees']} |"
        )
    lines.extend([
        "",
        "The identity-lie row is deliberately syntactically valid. Only the controlled",
        "harness knows the real crop, so this row documents the trust boundary rather than",
        "claiming that C2PA alone proves edit geometry.",
    ])
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
