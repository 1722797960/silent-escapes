#!/usr/bin/env python3
"""Prepare and summarize honest centered-crop controls.

The source pixels are the existing centered crop-and-resize assets from the
end-to-end experiment.  This script removes their old C2PA metadata, re-signs
them with an honest AI-created-plus-edited manifest, and preserves the signed
regional watermark assertion.  The standard defense auditor is run separately
between the ``prepare`` and ``summarize`` stages.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

from PIL import Image

from inpainting_common_v1 import sha256_file, wilson_interval


FALSE_FLAG_VERDICTS = {"TAMPERED_REGION", "WATERMARK_SUPPRESSED"}


def parse_crop_fracs(value: str) -> list[float]:
    fractions = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not fractions or any(fraction <= 0 or fraction >= 0.5 for fraction in fractions):
        raise ValueError("crop fractions must be comma-separated values in (0, 0.5)")
    return fractions


def crop_name(fraction: float) -> str:
    return f"crop{int(round(fraction * 100)):02d}"


def decoded_rgb_sha256(path: Path) -> str:
    image = Image.open(path).convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{image.width}x{image.height}:RGB:".encode("ascii"))
    digest.update(image.tobytes())
    return digest.hexdigest()


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


def prepare(args: argparse.Namespace, root: Path) -> None:
    started = time.perf_counter()
    source_meta_path = args.source_e2e / "embed_meta.json"
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    entries = source_meta["images"]
    fractions = parse_crop_fracs(args.crop_fracs)
    output_meta_path = args.output_dir / "prepare_meta.json"
    if output_meta_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_meta_path}")

    helper = root / "experiments/sign_with_region_assertion.py"
    template = root / "manifests/manifest_ai_edited.json"
    rows: list[dict] = []
    for fraction in fractions:
        condition = crop_name(fraction)
        for ordinal, entry in enumerate(entries, start=1):
            source = args.source_e2e / "attack_resign" / condition / entry["signed_image"]
            if not source.is_file():
                raise FileNotFoundError(source)
            unsigned_dir = args.output_dir / "unsigned" / condition
            signed_dir = args.output_dir / "attack_resign" / condition
            unsigned_dir.mkdir(parents=True, exist_ok=True)
            signed_dir.mkdir(parents=True, exist_ok=True)
            unsigned = unsigned_dir / entry["signed_image"]
            signed = signed_dir / entry["signed_image"]
            if not args.overwrite and (unsigned.exists() or signed.exists()):
                raise FileExistsError(f"Refusing to overwrite {unsigned} or {signed}")

            image = Image.open(source).convert("RGB")
            image.save(unsigned, format="PNG")
            source_pixel_hash = decoded_rgb_sha256(source)
            unsigned_pixel_hash = decoded_rgb_sha256(unsigned)
            if source_pixel_hash != unsigned_pixel_hash:
                raise AssertionError(f"decoded pixels changed while stripping C2PA: {source}")

            region = {
                "bbox_csv": entry["bbox_csv"],
                "bitmap64": entry["bitmap64"],
                "payload_hash": entry["payload_hash"],
                "alg": "wam_mit_regional",
            }
            run_checked(
                [
                    str(args.c2pa_python),
                    str(helper),
                    "--input",
                    str(unsigned),
                    "--output",
                    str(signed),
                    "--manifest-template",
                    str(template),
                    "--region-json",
                    "-",
                    "--cert",
                    str(args.cert),
                    "--key",
                    str(args.key),
                ],
                stdin=json.dumps(region),
            )
            signed_pixel_hash = decoded_rgb_sha256(signed)
            if signed_pixel_hash != unsigned_pixel_hash:
                raise AssertionError(f"decoded pixels changed during C2PA signing: {signed}")
            rows.append(
                {
                    "image": entry["input_image"],
                    "signed_image": entry["signed_image"],
                    "crop_frac_per_side": fraction,
                    "condition": condition,
                    "source_attack_sha256": sha256_file(source),
                    "unsigned_sha256": sha256_file(unsigned),
                    "honest_signed_sha256": sha256_file(signed),
                    "decoded_rgb_sha256": signed_pixel_hash,
                }
            )
            print(
                f"[{condition} {ordinal:03d}/{len(entries):03d}] "
                f"honest re-sign complete",
                flush=True,
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "embed_meta.json").write_text(
        json.dumps(source_meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    payload = {
        "experiment": "honest centered-crop controls",
        "version": "benign_centered_crops_v1",
        "crop_fraction_semantics": "fraction removed from each side before resize-back",
        "crop_fracs": fractions,
        "manifest": {
            "template": "manifests/manifest_ai_edited.json",
            "claims": ["c2pa.created/trainedAlgorithmicMedia", "c2pa.edited"],
            "template_sha256": sha256_file(template),
        },
        "source_embed_meta_sha256": sha256_file(source_meta_path),
        "signing_helper_sha256": sha256_file(helper),
        "certificate_sha256": sha256_file(args.cert),
        "pixel_invariant": "decoded RGB pixels match the existing centered-crop assets",
        "prepare_seconds": round(time.perf_counter() - started, 3),
        "rows": rows,
    }
    output_meta_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_meta_path} ({len(rows)} rows)")


def honest_verdict(row: dict, *, threshold: float, floor: float, logit_min: float) -> str:
    if not row["manifest_valid"]:
        return "PIPELINE_INVALID_MANIFEST"
    if not row["region_assertion_present"] or not row["payload_hash_ok"]:
        return "PIPELINE_INVALID_ASSERTION"
    if float(row["bit_accuracy"]) >= threshold:
        return "CONSISTENT_SYNTHETIC"
    if float(row["region_retention"]) < floor:
        return "TAMPERED_REGION"
    if float(row["logit_in_region"]) < logit_min:
        return "WATERMARK_SUPPRESSED"
    return "BENIGN_NO_FLAG"


def rate_record(count: int, total: int) -> dict:
    low, high = wilson_interval(count, total)
    return {
        "count": count,
        "total": total,
        "rate": count / total if total else 0.0,
        "wilson95": [low, high],
    }


def summarize(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    audit_path = args.output_dir / "audit_retention.json"
    if not audit_path.is_file():
        audit_path = args.output_dir / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    baseline = json.loads(args.reference_audit.read_text(encoding="utf-8"))
    threshold = float(audit.get("threshold", 0.75))
    floor = float(audit.get("region_min_retention", 0.05))
    logit_min = float(audit.get("logit_mean_min", 0.5))

    baseline_failed: set[str] = set()
    for row in baseline["rows"]:
        if float(row["crop_frac"]) != 0.0:
            continue
        bit_accuracy = float(row["bit_accuracy"])
        logit = float(row["logit_in_region"])
        if bit_accuracy < threshold and logit < logit_min:
            baseline_failed.add(row["image"])

    enriched_rows = []
    for row in audit["rows"]:
        enriched = dict(row)
        verdict = honest_verdict(row, threshold=threshold, floor=floor, logit_min=logit_min)
        enriched["honest_manifest_verdict"] = verdict
        enriched["benign_region_false_flag"] = verdict in FALSE_FLAG_VERDICTS
        enriched["baseline_qualified"] = row["image"] not in baseline_failed
        enriched_rows.append(enriched)

    conditions = []
    for fraction in sorted({float(row["crop_frac"]) for row in enriched_rows}):
        rows = [row for row in enriched_rows if float(row["crop_frac"]) == fraction]
        qualified = [row for row in rows if row["baseline_qualified"]]
        strict_flags = sum(row["benign_region_false_flag"] for row in rows)
        qualified_flags = sum(row["benign_region_false_flag"] for row in qualified)
        conditions.append(
            {
                "crop_frac_per_side": fraction,
                "n": len(rows),
                "manifest_valid": sum(bool(row["manifest_valid"]) for row in rows),
                "region_assertion_present": sum(
                    bool(row["region_assertion_present"]) for row in rows
                ),
                "payload_hash_ok": sum(bool(row["payload_hash_ok"]) for row in rows),
                "strict_false_flag": rate_record(strict_flags, len(rows)),
                "baseline_qualified_false_flag": rate_record(
                    qualified_flags, len(qualified)
                ),
                "verdict_counts": dict(
                    Counter(row["honest_manifest_verdict"] for row in rows)
                ),
            }
        )

    summary = {
        "experiment": "honest centered-crop controls",
        "version": "benign_centered_crops_v1",
        "flag_semantics": (
            "A flag means evidence bound to the signed region was removed or "
            "suppressed; it does not establish malicious intent."
        ),
        "manifest": "honest AI-created plus edited",
        "reference_operating_point": {
            "bit_threshold": threshold,
            "region_min_retention": floor,
            "logit_mean_min": logit_min,
        },
        "baseline_failed_images": sorted(baseline_failed),
        "conditions": conditions,
        "summarize_seconds": round(time.perf_counter() - started, 3),
    }
    (args.output_dir / "audit_honest.json").write_text(
        json.dumps({"summary": summary, "rows": enriched_rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "audit_honest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(enriched_rows[0]))
        writer.writeheader()
        writer.writerows(enriched_rows)
    print(json.dumps(summary, indent=2))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "summarize"))
    parser.add_argument(
        "--source-e2e", type=Path, default=root / "results/defense_e2e"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results/benign_centered_crops_v1"
    )
    parser.add_argument("--crop-fracs", default="0.10,0.20")
    parser.add_argument("--c2pa-python", type=Path)
    parser.add_argument("--cert", type=Path, default=root / "certs/ec_chain.pem")
    parser.add_argument("--key", type=Path, default=root / "certs/ec_key.pem")
    parser.add_argument(
        "--reference-audit", type=Path, default=root / "results/defense_e2e/audit.json"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare":
        if args.c2pa_python is None:
            parser.error("--c2pa-python is required for prepare")
        prepare(args, root)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
