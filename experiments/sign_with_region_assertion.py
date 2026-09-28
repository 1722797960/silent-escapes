"""Signing helper (runs under the integrity-clash venv with c2pa installed).

Reads a region-assertion JSON from --region-json (a file path, or "-" for
stdin) and signs --input with:
  - the misleading human-edited manifest template, plus
  - a com.example.region_assertion custom assertion (bbox/bitmap64/hash/alg)
"""
import argparse
import json
import sys
from pathlib import Path

import c2pa
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend


def create_signer(cert_path: Path, key_path: Path) -> c2pa.Signer:
    certs_pem = cert_path.read_bytes().decode("utf-8")
    key = serialization_load(key_path)
    return c2pa.Signer.from_callback(
        callback=lambda data: key.sign(data, ec.ECDSA(hashes.SHA256())),
        alg=c2pa.C2paSigningAlg.ES256,
        certs=certs_pem,
        tsa_url=None,
    )


def serialization_load(key_path: Path):
    from cryptography.hazmat.primitives import serialization
    return serialization.load_pem_private_key(key_path.read_bytes(),
                                              password=None,
                                              backend=default_backend())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--manifest-template", required=True)
    ap.add_argument("--cert", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--region-json", required=True,
                    help="path to region assertion JSON, or '-' for stdin")
    args = ap.parse_args()

    if args.region_json == "-":
        region = json.loads(sys.stdin.read())
    else:
        region = json.loads(Path(args.region_json).read_text())

    base = json.loads(Path(args.manifest_template).read_text())
    base["format"] = "image/png"
    base.setdefault("title", Path(args.input).name)
    base["assertions"].append({
        "label": "com.example.region_assertion",
        "data": region,
    })

    signer = create_signer(Path(args.cert), Path(args.key))
    with c2pa.Builder(base) as builder:
        builder.sign_file(source_path=args.input,
                          dest_path=args.output,
                          signer=signer)


if __name__ == "__main__":
    main()
