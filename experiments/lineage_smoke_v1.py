"""Build and inspect one C2PA parentOf edit chain.

This is a compatibility smoke test for c2pa-python 0.37.10.  It creates a new
manifest for an already edited pixel asset, links the signed pre-edit asset as
its parentOf ingredient, and verifies that the ancestor's region assertion is
available through the resulting manifest store.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import c2pa
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec


REGION_LABEL = "com.example.region_assertion"
PARENT_LABEL = "parent-source"


def create_signer(cert_path: Path, key_path: Path) -> c2pa.Signer:
    certs_pem = cert_path.read_text(encoding="utf-8")
    key = serialization.load_pem_private_key(
        key_path.read_bytes(), password=None, backend=default_backend()
    )
    return c2pa.Signer.from_callback(
        callback=lambda data: key.sign(data, ec.ECDSA(hashes.SHA256())),
        alg=c2pa.C2paSigningAlg.ES256,
        certs=certs_pem,
        tsa_url=None,
    )


def build_child_manifest(parent: Path, edited_pixels: Path, output: Path,
                         cert: Path, key: Path) -> None:
    manifest = {
        "claim_generator": "lineage_smoke_v1",
        "format": "image/png",
        "title": output.name,
        "assertions": [
            {
                "label": "c2pa.actions.v2",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.opened",
                            "softwareAgent": "LineageAwareEditor/1.0",
                            "parameters": {"ingredientIds": [PARENT_LABEL]},
                        },
                        {
                            "action": "c2pa.edited",
                            "softwareAgent": "LineageAwareEditor/1.0",
                        },
                    ]
                },
            }
        ],
    }
    signer = create_signer(cert, key)
    output.parent.mkdir(parents=True, exist_ok=True)
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
        builder.sign_file(
            source_path=str(edited_pixels),
            dest_path=str(output),
            signer=signer,
        )


def inspect_chain(path: Path) -> dict:
    reader = c2pa.Reader(str(path))
    detailed = json.loads(reader.detailed_json())
    active_label = detailed["active_manifest"]
    manifests = detailed["manifests"]
    active = manifests[active_label]
    ancestor_hits = []
    for label, manifest in manifests.items():
        if label == active_label:
            continue
        region = manifest.get("assertion_store", {}).get(REGION_LABEL)
        if region is not None:
            ancestor_hits.append({"manifest": label, "region": region})
    return {
        "validation_state": reader.get_validation_state(),
        "active_manifest": active_label,
        "manifest_count": len(manifests),
        "active_assertions": sorted(active.get("assertion_store", {}).keys()),
        "active_ingredients": active.get("ingredients", []),
        "ancestor_region_assertion_count": len(ancestor_hits),
        "ancestor_region_assertions": ancestor_hits,
        "validation_results": detailed.get("validation_results", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True)
    parser.add_argument("--edited-pixels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    build_child_manifest(
        Path(args.parent), Path(args.edited_pixels), output,
        Path(args.cert), Path(args.key),
    )
    result = inspect_chain(output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["validation_state"] != "Valid":
        raise SystemExit("child manifest did not validate")
    if result["manifest_count"] < 2:
        raise SystemExit("parent manifest was not embedded as an ingredient")
    if result["ancestor_region_assertion_count"] != 1:
        raise SystemExit("expected exactly one ancestor region assertion")


if __name__ == "__main__":
    main()
