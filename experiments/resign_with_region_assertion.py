"""Re-sign an attacked (cropped) watermarked image with the region assertion.

Models a *compliant* editing tool: after the adversary's geometric edit, the
tool re-signs the asset. Following the same manifest semantics as the original
integrity-clash pipeline (c2pa.created digitalCapture + c2pa.edited by a photo
editor — the "misleading human-edited" claim), and carries the
com.example.region_assertion (bbox_csv + 64x64 bitmap + payload hash) into the
new manifest. We verified with c2pa 0.37.10 that the re-signed file validates
(only failure: self-signed cert untrusted — identical to the original paper)
and the assertion survives.

Run under the integrity-clash venv (c2pa installed).
"""
import argparse
import json
import sys
from pathlib import Path

import c2pa
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend

REGION_LABEL = "com.example.region_assertion"


def create_signer(cert_path: Path, key_path: Path) -> c2pa.Signer:
    certs_pem = cert_path.read_bytes().decode("utf-8")
    key = serialization.load_pem_private_key(key_path.read_bytes(),
                                             password=None,
                                             backend=default_backend())
    return c2pa.Signer.from_callback(
        callback=lambda data: key.sign(data, ec.ECDSA(hashes.SHA256())),
        alg=c2pa.C2paSigningAlg.ES256,
        certs=certs_pem,
        tsa_url=None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="attacked (cropped) PNG")
    ap.add_argument("--output", required=True)
    ap.add_argument("--region-json", required=True,
                    help="region assertion JSON file, or '-' for stdin")
    ap.add_argument("--cert", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    region = (json.loads(sys.stdin.read()) if args.region_json == "-"
              else json.loads(Path(args.region_json).read_text()))

    manifest = {
        "claim_generator": "compliant_editor_e2e",
        "format": "image/png",
        "title": args.title or Path(args.input).name,
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType":
                                "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture",
                            "softwareAgent": "PhotoEditor/2.0",
                        },
                        {
                            "action": "c2pa.edited",
                            "softwareAgent": "PhotoEditor/2.0",
                        },
                    ]
                },
            },
            {"label": REGION_LABEL, "data": region},
        ],
    }

    signer = create_signer(Path(args.cert), Path(args.key))
    with c2pa.Builder(manifest) as builder:
        builder.sign_file(source_path=args.input,
                          dest_path=args.output,
                          signer=signer)


if __name__ == "__main__":
    main()
