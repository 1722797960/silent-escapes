"""Sign one protocol-v2 pilot case under a controlled manifest topology.

Run this helper with the c2pa-python environment. The request JSON is produced
by protocol_v2_pilot_v1.py and contains no private key material.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import c2pa
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

from protocol_v2_common import confined_path


REGION_LABEL = "com.example.region_assertion"
PARENT_LABEL = "parent-source"
TRANSFORM_KEY = "com.example.crop_transform"
TOPOLOGIES = {"ancestor-reference", "current-reference", "cut-chain", "no-region-claim"}


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


def validate_region(region: dict[str, Any]) -> None:
    bits = region.get("reference_bits")
    if not isinstance(bits, str) or len(bits) != 32 or set(bits) - {"0", "1"}:
        raise ValueError("region reference_bits must be exactly 32 binary digits")
    expected = hashlib.sha256(bits.encode("ascii")).hexdigest()
    if region.get("payload_hash") != expected:
        raise ValueError("region payload_hash is inconsistent with reference_bits")
    if region.get("schema_version") != "2":
        raise ValueError("region schema_version must be 2")


def created_action(semantics: str, agent: str) -> dict[str, Any]:
    if semantics == "ai":
        source_type = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
    elif semantics == "none":
        source_type = "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"
    else:
        raise ValueError(f"unknown manifest semantics: {semantics}")
    return {
        "action": "c2pa.created",
        "digitalSourceType": source_type,
        "softwareAgent": agent,
    }


def parent_manifest(title: str, semantics: str, region: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_generator": "protocol_v2_pilot/parent",
        "format": "image/png",
        "title": title,
        "assertions": [
            {
                "label": "c2pa.actions.v2",
                "data": {"actions": [created_action(semantics, "ProtocolV2Source/1.0")]},
            },
            {"label": REGION_LABEL, "data": region},
        ],
    }


def child_actions(request: dict[str, Any], *, with_parent: bool) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if with_parent:
        actions.append({
            "action": "c2pa.opened",
            "softwareAgent": "ProtocolV2Editor/1.0",
            "parameters": {"ingredientIds": [PARENT_LABEL]},
        })
    transform = request.get("crop_transform")
    if transform is not None:
        parameters: dict[str, Any] = {TRANSFORM_KEY: transform}
        if with_parent:
            parameters["ingredientIds"] = [PARENT_LABEL]
        actions.append({
            "action": "c2pa.cropped",
            "softwareAgent": "ProtocolV2Editor/1.0",
            "parameters": parameters,
        })
    else:
        actions.append({
            "action": "c2pa.edited",
            "softwareAgent": "ProtocolV2Editor/1.0",
        })
    return actions


def sign_file(source: Path, output: Path, manifest: dict[str, Any], signer: c2pa.Signer,
              parent: Path | None = None) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with c2pa.Builder(manifest) as builder:
        if parent is not None:
            with parent.open("rb") as parent_stream:
                builder.add_ingredient(
                    {"title": parent.name, "relationship": "parentOf", "label": PARENT_LABEL},
                    "image/png",
                    parent_stream,
                )
        builder.sign_file(source_path=str(source), dest_path=str(output), signer=signer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--cert", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    output_root = args.output_root.resolve(strict=True)
    input_roots = [output_root, *(root.resolve(strict=True) for root in args.input_root)]
    request_path = confined_path(
        args.request, [output_root], field="request", must_exist=True, require_file=True
    )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    topology = request["topology"]
    if topology not in TOPOLOGIES:
        raise ValueError(f"unsupported topology: {topology}")
    child_pixels = confined_path(
        request["child_pixels"], input_roots,
        field="child_pixels", must_exist=True, require_file=True,
    )
    child_output = confined_path(
        request["child_output"], [output_root], field="child_output", must_exist=False
    )
    signer = create_signer(args.cert, args.key)

    if topology == "no-region-claim":
        manifest = {
            "claim_generator": "protocol_v2_pilot/no-region",
            "format": "image/png",
            "title": child_output.name,
            "assertions": [{
                "label": "c2pa.actions.v2",
                "data": {"actions": [created_action(request["manifest_semantics"], "ProtocolV2Source/1.0")]},
            }],
        }
        sign_file(child_pixels, child_output, manifest, signer)
    elif topology == "cut-chain":
        # A fresh claim must start with c2pa.created or c2pa.opened. It still
        # represents a continuity break because the subsequent edit/crop has
        # no parentOf ingredient.
        actions = [
            created_action(request["manifest_semantics"], "ProtocolV2FreshRoot/1.0"),
            *child_actions(request, with_parent=False),
        ]
        manifest = {
            "claim_generator": "protocol_v2_pilot/cut-chain",
            "format": "image/png",
            "title": child_output.name,
            "assertions": [{
                "label": "c2pa.actions.v2",
                "data": {"actions": actions},
            }],
        }
        sign_file(child_pixels, child_output, manifest, signer)
    else:
        region = request["region_assertion"]
        validate_region(region)
        parent_pixels = confined_path(
            request["parent_pixels"], input_roots,
            field="parent_pixels", must_exist=True, require_file=True,
        )
        parent_output = confined_path(
            request["parent_output"], [output_root], field="parent_output", must_exist=False
        )
        if not parent_output.exists():
            sign_file(
                parent_pixels,
                parent_output,
                parent_manifest(parent_output.name, request["manifest_semantics"], region),
                signer,
            )
        assertions: list[dict[str, Any]] = [{
            "label": "c2pa.actions.v2",
            "data": {"actions": child_actions(request, with_parent=True)},
        }]
        if topology == "current-reference":
            assertions.append({"label": REGION_LABEL, "data": region})
        manifest = {
            "claim_generator": "protocol_v2_pilot/child",
            "format": "image/png",
            "title": child_output.name,
            "assertions": assertions,
        }
        sign_file(child_pixels, child_output, manifest, signer, parent_output)
    print(json.dumps({"case_id": request["case_id"], "output": str(child_output)}))


if __name__ == "__main__":
    main()
