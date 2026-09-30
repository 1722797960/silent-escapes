"""Run the three-condition C2PA lineage ablation on existing crop-40 assets.

The three signed outputs for each image have identical decoded pixels.  WAM
metrics are therefore reused from the already completed end-to-end crop audit;
this experiment changes only the manifest topology.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODES = ("propagated", "ancestor-only", "cut-chain")
EXPECTED_STATUS = {
    "propagated": "CURRENT_ASSERTION",
    "ancestor-only": "ANCESTOR_ASSERTION",
    "cut-chain": "PROVENANCE_DISCONTINUITY",
}


@dataclass(frozen=True)
class RunConfig:
    e2e: Path
    output: Path
    audit_path: Path
    c2pa_python: Path
    cert: Path
    key: Path
    resign_helper: Path
    read_helper: Path
    num_images: int
    resume: bool


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def decoded_pixel_hash(path: Path) -> str:
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(pixels.tobytes()).hexdigest()


def summarize(rows: list[dict]) -> dict:
    summary = {"experiment": "lineage_ablation_v1", "conditions": {}}
    for mode in MODES:
        selected = [row for row in rows if row["condition"] == mode]
        summary["conditions"][mode] = {
            "n": len(selected),
            "manifest_valid": sum(row["manifest_valid"] for row in selected),
            "pixel_identical": sum(row["pixel_identical"] for row in selected),
            "parent_manifest_validated": sum(
                row["parent_manifest_validated"] for row in selected
            ),
            "payload_hash_ok": sum(row["payload_hash_ok"] is True for row in selected),
            "baseline_silent_escape": sum(
                row["baseline_audit"] == "SILENT_ESCAPE" for row in selected
            ),
            "lineage_status": dict(Counter(row["lineage_status"] for row in selected)),
            "audit_v3": dict(Counter(row["audit_v3"] for row in selected)),
        }
    return summary


def write_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# C2PA lineage ablation v1",
        "",
        "All conditions reuse identical crop-40 pixels and saved WAM metrics; only the manifest topology changes.",
        "",
        "| Condition | N | Valid manifest | Pixel-identical | Parent validated | Region source / policy status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for mode in MODES:
        item = summary["conditions"][mode]
        statuses = ", ".join(
            f"{key}: {value}" for key, value in item["lineage_status"].items()
        )
        lines.append(
            f"| {mode} | {item['n']} | {item['manifest_valid']} | "
            f"{item['pixel_identical']} | {item['parent_manifest_validated']} | {statuses} |"
        )
    lines.extend(
        [
            "",
            "`PROVENANCE_DISCONTINUITY` is a strict consumer-policy result for an "
            "edited manifest without a `parentOf` link. It is not included in the "
            "paper's region-evidence flag rate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-dir", default=str(ROOT / "results" / "defense_e2e"))
    parser.add_argument(
        "--output-dir", default=str(ROOT / "results" / "lineage_ablation_v1")
    )
    parser.add_argument(
        "--audit-json",
        help="corrected saved detector audit; defaults to audit_retention.json when present",
    )
    parser.add_argument(
        "--c2pa-python", default=str(ROOT / "venvs" / "c2pa" / "bin" / "python")
    )
    parser.add_argument("--cert", default=str(ROOT / "certs" / "ec_chain.pem"))
    parser.add_argument("--key", default=str(ROOT / "certs" / "ec_key.pem"))
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> RunConfig:
    e2e = Path(args.e2e_dir).resolve()
    default_audit = e2e / "audit_retention.json"
    audit_path = Path(args.audit_json).resolve() if args.audit_json else default_audit
    if not audit_path.exists() and not args.audit_json:
        audit_path = e2e / "audit.json"
    c2pa_python = Path(args.c2pa_python).expanduser()
    if not c2pa_python.is_absolute():
        c2pa_python = Path.cwd() / c2pa_python
    return RunConfig(
        e2e=e2e,
        output=Path(args.output_dir).resolve(),
        audit_path=audit_path,
        # Preserve the virtual-environment launcher instead of resolving its
        # symlink to a base interpreter that lacks the venv's site-packages.
        c2pa_python=c2pa_python.absolute(),
        cert=Path(args.cert).resolve(),
        key=Path(args.key).resolve(),
        resign_helper=(ROOT / "experiments" / "resign_with_lineage_v1.py").resolve(),
        read_helper=(ROOT / "experiments" / "read_region_lineage_v1.py").resolve(),
        num_images=args.num_images,
        resume=args.resume,
    )


def validate_inputs(config: RunConfig, rows_path: Path) -> None:
    if rows_path.exists() and not config.resume:
        raise SystemExit(f"{rows_path} already exists; use --resume or a new directory")
    required = [
        config.e2e / "embed_meta.json",
        config.audit_path,
        config.c2pa_python,
        config.cert,
        config.key,
        config.resign_helper,
        config.read_helper,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required paths:\n" + "\n".join(missing))


def load_inputs(config: RunConfig) -> tuple[list[dict], dict[str, dict]]:
    metadata_path = config.e2e / "embed_meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["images"]
    audit_rows = json.loads(config.audit_path.read_text(encoding="utf-8"))["rows"]
    crop40 = {
        row["image"]: row
        for row in audit_rows
        if abs(float(row["crop_frac"]) - 0.40) < 1e-9
    }
    return metadata[: config.num_images], crop40


def write_edited_pixels(attacked: Path, destination: Path) -> str:
    Image.open(attacked).convert("RGB").save(destination, format="PNG")
    return decoded_pixel_hash(destination)


def sign_condition(
    config: RunConfig,
    mode: str,
    pixels: Path,
    signed: Path,
    parent: Path,
    region_json: Path,
) -> dict:
    command = [
        str(config.c2pa_python),
        str(config.resign_helper),
        "--mode",
        mode,
        "--input",
        str(pixels),
        "--output",
        str(signed),
        "--cert",
        str(config.cert),
        "--key",
        str(config.key),
    ]
    if mode != "cut-chain":
        command.extend(["--parent", str(parent)])
    if mode == "propagated":
        command.extend(["--region-json", str(region_json)])
    if signed.exists() and config.resume:
        command.append("--overwrite")
    subprocess.run(command, check=True, capture_output=True, text=True)
    inspection = subprocess.run(
        [str(config.c2pa_python), str(config.read_helper), str(signed)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(inspection.stdout)


def validate_condition(
    mode: str,
    input_image: str,
    signed: Path,
    inspected: dict,
    reference_pixel_hash: str,
    expected_payload_hash: str,
) -> tuple[bool, bool | None]:
    status = inspected["strict_continuity_status"]
    if status != EXPECTED_STATUS[mode]:
        raise RuntimeError(f"{mode} yielded {status} for {input_image}")
    pixel_identical = decoded_pixel_hash(signed) == reference_pixel_hash
    if not pixel_identical:
        raise RuntimeError(f"decoded pixels changed in {signed}")
    payload_hash_ok = None
    if mode != "cut-chain":
        resolved_region = inspected.get("region") or {}
        payload_hash_ok = resolved_region.get("payload_hash") == expected_payload_hash
        if not payload_hash_ok:
            raise RuntimeError(f"payload hash mismatch in {signed}")
    if not inspected["manifest_valid"]:
        raise RuntimeError(f"manifest validation failed for {signed}")
    if mode != "cut-chain" and not inspected["parent_manifest_validated"]:
        raise RuntimeError(f"parent validation missing for {signed}")
    return pixel_identical, payload_hash_ok


def build_row(
    mode: str,
    input_image: str,
    signed: Path,
    output: Path,
    inspected: dict,
    old: dict,
    reference_pixel_hash: str,
    pixel_identical: bool,
    payload_hash_ok: bool | None,
) -> dict:
    cut_chain = mode == "cut-chain"
    return {
        "image": input_image,
        "condition": mode,
        "signed_output": str(signed.relative_to(output)),
        "manifest_valid": bool(inspected["manifest_valid"]),
        "manifest_count": inspected["manifest_count"],
        "active_parent_count": inspected["active_parent_count"],
        "parent_manifest_validated": bool(inspected["parent_manifest_validated"]),
        "region_assertion_source": inspected["region_assertion_source"],
        "lineage_depth": inspected["lineage_depth"],
        "lineage_status": inspected["strict_continuity_status"],
        "payload_hash_ok": payload_hash_ok,
        "pixel_identical": pixel_identical,
        "decoded_pixel_sha256": reference_pixel_hash,
        "detector_metrics_reused": True,
        "bit_accuracy": old["bit_accuracy"],
        "region_retention": old.get("region_retention", old["region_in_frame"]),
        "logit_in_region": old["logit_in_region"],
        "baseline_audit": old["audit_v1_deployed"],
        "region_audit": None if cut_chain else old["audit_v2_region_aware"],
        "audit_v3": (
            "PROVENANCE_DISCONTINUITY" if cut_chain else old["audit_v2_region_aware"]
        ),
    }


def process_asset(
    config: RunConfig,
    entry: dict,
    crop40: dict[str, dict],
    completed: set[tuple[str, str]],
    rows_path: Path,
    temp_pixels: Path,
    temp_region: Path,
) -> None:
    input_image = entry["input_image"]
    signed_name = entry["signed_image"]
    if input_image not in crop40:
        raise KeyError(f"missing crop-40 audit row for {input_image}")
    parent = config.e2e / "watermarked_signed_ra" / signed_name
    attacked = config.e2e / "attack_resign" / "crop40" / signed_name
    reference_pixel_hash = write_edited_pixels(attacked, temp_pixels)
    region = {
        "bbox_csv": entry["bbox_csv"],
        "bitmap64": entry["bitmap64"],
        "payload_hash": entry["payload_hash"],
        "alg": "wam_mit_regional",
    }
    temp_region.write_text(json.dumps(region), encoding="utf-8")
    for mode in MODES:
        if (input_image, mode) in completed:
            continue
        signed = config.output / "signed" / mode / signed_name
        inspected = sign_condition(
            config, mode, temp_pixels, signed, parent, temp_region
        )
        pixel_identical, payload_hash_ok = validate_condition(
            mode,
            input_image,
            signed,
            inspected,
            reference_pixel_hash,
            entry["payload_hash"],
        )
        row = build_row(
            mode,
            input_image,
            signed,
            config.output,
            inspected,
            crop40[input_image],
            reference_pixel_hash,
            pixel_identical,
            payload_hash_ok,
        )
        append_jsonl(rows_path, row)
        completed.add((input_image, mode))


def main() -> None:
    config = make_config(parse_args())
    rows_path = config.output / "rows.jsonl"
    validate_inputs(config, rows_path)
    config.output.mkdir(parents=True, exist_ok=True)
    metadata, crop40 = load_inputs(config)
    completed = {(row["image"], row["condition"]) for row in load_jsonl(rows_path)}
    temp_pixels = config.output / "_edited_pixels.png"
    temp_region = config.output / "_region.json"

    for index, entry in enumerate(metadata, start=1):
        process_asset(
            config, entry, crop40, completed, rows_path, temp_pixels, temp_region
        )
        print(f"[{index:03d}/{len(metadata):03d}] {entry['signed_image']}", flush=True)

    temp_pixels.unlink(missing_ok=True)
    temp_region.unlink(missing_ok=True)
    rows = load_jsonl(rows_path)
    summary = summarize(rows)
    (config.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_markdown(config.output / "summary.md", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
