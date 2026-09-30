"""Re-sign an edited PNG under one of three C2PA lineage conditions."""
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
MODES = {"propagated", "ancestor-only", "cut-chain"}


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


def manifest_for(mode: str, title: str, region: dict | None) -> dict:
    if mode in {"propagated", "ancestor-only"}:
        actions = [
            {
                "action": "c2pa.opened",
                "softwareAgent": "LineageAwareEditor/1.0",
                "parameters": {"ingredientIds": [PARENT_LABEL]},
            },
            {"action": "c2pa.edited", "softwareAgent": "LineageAwareEditor/1.0"},
        ]
    else:
        actions = [
            {
                "action": "c2pa.created",
                "digitalSourceType":
                    "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture",
                "softwareAgent": "PhotoEditor/2.0",
            },
            {"action": "c2pa.edited", "softwareAgent": "PhotoEditor/2.0"},
        ]
    assertions = [{"label": "c2pa.actions.v2", "data": {"actions": actions}}]
    if mode == "propagated":
        if region is None:
            raise ValueError("propagated mode requires --region-json")
        assertions.append({"label": REGION_LABEL, "data": region})
    return {
        "claim_generator": "lineage_ablation_v1",
        "format": "image/png",
        "title": title,
        "assertions": assertions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--parent")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--region-json")
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {output}")
    if args.mode != "cut-chain" and not args.parent:
        raise SystemExit(f"{args.mode} mode requires --parent")
    region = None
    if args.region_json:
        region = json.loads(Path(args.region_json).read_text(encoding="utf-8"))

    output.parent.mkdir(parents=True, exist_ok=True)
    signer = create_signer(Path(args.cert), Path(args.key))
    with c2pa.Builder(manifest_for(args.mode, output.name, region)) as builder:
        if args.mode != "cut-chain":
            with Path(args.parent).open("rb") as parent_stream:
                builder.add_ingredient(
                    {
                        "title": Path(args.parent).name,
                        "relationship": "parentOf",
                        "label": PARENT_LABEL,
                    },
                    "image/png",
                    parent_stream,
                )
        builder.sign_file(
            source_path=args.input,
            dest_path=str(output),
            signer=signer,
        )


if __name__ == "__main__":
    main()
