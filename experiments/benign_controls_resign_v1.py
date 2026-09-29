"""Sign benign edits with honest AI provenance and the applicable region claim."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from inpainting_common_v1 import sha256_file


def run_checked(command: list[str], *, stdin: str | None = None) -> None:
    completed = subprocess.run(
        command,
        input=stdin,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {command}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=root / "results/benign_controls_v1_pilot",
    )
    parser.add_argument("--c2pa-python", type=Path, required=True)
    parser.add_argument("--cert", type=Path, default=root / "certs/ec_chain.pem")
    parser.add_argument("--key", type=Path, default=root / "certs/ec_key.pem")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    generate_meta = args.experiment_dir / "generate_meta.json"
    payload = json.loads(generate_meta.read_text(encoding="utf-8"))
    output_meta = args.experiment_dir / "signed_meta.json"
    if output_meta.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_meta}")

    region_helper = root / "experiments/sign_with_region_assertion.py"
    directory_helper = root / "scripts/sign_images.py"
    ai_edited_template = root / "manifests/manifest_ai_edited.json"
    ai_template = root / "manifests/manifest_ai.json"
    signed_rows = []

    # Region-bearing assets are signed individually because every assertion
    # carries a different mask and payload hash.
    watermarked = [row for row in payload["rows"] if row["watermarked"]]
    for ordinal, row in enumerate(watermarked, start=1):
        source = args.experiment_dir / "unsigned" / row["condition"] / row["unsigned_output"]
        signed_dir = args.experiment_dir / "signed" / row["condition"]
        signed_dir.mkdir(parents=True, exist_ok=True)
        output_name = source.stem + "_signed_ai_ra.png"
        destination = signed_dir / output_name
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {destination}")
        region = {
            "bbox_csv": row["bbox_csv"],
            "bitmap64": row["bitmap64"],
            "payload_hash": row["payload_hash"],
            "alg": "wam_mit_regional",
        }
        run_checked(
            [
                str(args.c2pa_python),
                str(region_helper),
                "--input",
                str(source),
                "--output",
                str(destination),
                "--manifest-template",
                str(ai_edited_template),
                "--region-json",
                "-",
                "--cert",
                str(args.cert),
                "--key",
                str(args.key),
            ],
            stdin=json.dumps(region),
        )
        enriched = dict(row)
        enriched["signed_output"] = output_name
        enriched["signed_sha256"] = sha256_file(destination)
        signed_rows.append(enriched)
        print(f"[{ordinal:02d}/{len(watermarked):02d}] signed {row['condition']}", flush=True)

    # Clean negatives carry an honest AI manifest but make no regional WAM
    # claim, so absence of the custom assertion is expected rather than an error.
    clean_rows = [row for row in payload["rows"] if not row["watermarked"]]
    clean_input = args.experiment_dir / "unsigned" / "clean_negative"
    clean_output = args.experiment_dir / "signed" / "clean_negative"
    clean_output.mkdir(parents=True, exist_ok=True)
    expected_clean = [
        clean_output / f"{Path(row['unsigned_output']).stem}_signed_manifest_ai.png"
        for row in clean_rows
    ]
    if not args.overwrite and any(path.exists() for path in expected_clean):
        raise FileExistsError("Refusing to overwrite signed clean-negative assets")
    run_checked(
        [
            str(args.c2pa_python),
            str(directory_helper),
            str(clean_input),
            str(clean_output),
            "--manifest-template",
            str(ai_template),
            "--cert",
            str(args.cert),
            "--key",
            str(args.key),
        ]
    )
    for row, destination in zip(clean_rows, expected_clean):
        if not destination.is_file():
            raise FileNotFoundError(f"signer did not create {destination}")
        enriched = dict(row)
        enriched["signed_output"] = destination.name
        enriched["signed_sha256"] = sha256_file(destination)
        signed_rows.append(enriched)

    lookup = {(row["condition"], row["input_image"]): row for row in signed_rows}
    ordered_rows = [lookup[(row["condition"], row["input_image"])] for row in payload["rows"]]
    output_payload = dict(payload)
    output_payload["signing"] = {
        "region_helper_sha256": sha256_file(region_helper),
        "directory_helper_sha256": sha256_file(directory_helper),
        "ai_edited_template_sha256": sha256_file(ai_edited_template),
        "ai_template_sha256": sha256_file(ai_template),
        "certificate_sha256": sha256_file(args.cert),
    }
    output_payload["rows"] = ordered_rows
    output_meta.write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_meta} ({len(ordered_rows)} rows)")


if __name__ == "__main__":
    main()
