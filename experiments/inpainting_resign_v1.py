"""Re-sign LaMa outputs while carrying the original signed region assertion."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from inpainting_common_v1 import sha256_file


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=root / "results/inpainting_e2e_v1")
    parser.add_argument("--c2pa-python", type=Path, required=True)
    parser.add_argument("--cert", type=Path, default=root / "certs/ec_chain.pem")
    parser.add_argument("--key", type=Path, default=root / "certs/ec_key.pem")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    meta_path = args.experiment_dir / "inpaint_meta.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    signed_dir = args.experiment_dir / "inpainted_signed"
    signed_dir.mkdir(parents=True, exist_ok=True)
    helper = root / "experiments/resign_with_region_assertion.py"

    signed_entries = []
    for ordinal, entry in enumerate(payload["images"], start=1):
        source = args.experiment_dir / "inpainted_unsigned" / entry["unsigned_inpainted"]
        output_name = source.stem + "_signed_ra.png"
        destination = signed_dir / output_name
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {destination}")
        region = {
            "bbox_csv": entry["bbox_csv"],
            "bitmap64": entry["bitmap64"],
            "payload_hash": entry["payload_hash"],
            "alg": "wam_mit_regional",
        }
        subprocess.run(
            [
                str(args.c2pa_python),
                str(helper),
                "--input",
                str(source),
                "--output",
                str(destination),
                "--region-json",
                "-",
                "--cert",
                str(args.cert),
                "--key",
                str(args.key),
            ],
            input=json.dumps(region),
            text=True,
            check=True,
            capture_output=True,
        )
        enriched = dict(entry)
        enriched["signed_inpainted"] = output_name
        enriched["signed_sha256"] = sha256_file(destination)
        signed_entries.append(enriched)
        print(f"[{ordinal:02d}/{len(payload['images']):02d}] signed {output_name}", flush=True)

    output_payload = dict(payload)
    output_payload["signing"] = {
        "helper": str(helper.resolve()),
        "helper_sha256": sha256_file(helper),
        "certificate_sha256": sha256_file(args.cert),
    }
    output_payload["images"] = signed_entries
    output_path = args.experiment_dir / "signed_meta.json"
    output_path.write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
