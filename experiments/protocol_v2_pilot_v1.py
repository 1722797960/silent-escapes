"""Prepare, detect, and audit the 20-case protocol-v2 boundary pilot.

Preparation may read historical experiment metadata to migrate the old 32-bit
reference into a signed v2 assertion. The verifier subprocess receives only the
new signed asset. Detector outputs are cached by decoded RGB pixels, while
manifest and policy results are cached separately.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from protocol_v2_cache import ManifestCache, PixelCache, decoded_rgb_key, sha256_file
from protocol_v2_common import audit_protocol_v2, matching_bit_count


ROOT = Path(__file__).resolve().parents[1]
SIGN_HELPER = ROOT / "experiments" / "protocol_v2_sign_pilot.py"
READ_HELPER = ROOT / "experiments" / "read_actions_v2.py"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def find_one(rows: list[dict[str, Any]], selector: dict[str, Any]) -> dict[str, Any]:
    matches = [
        row for row in rows
        if all(row.get(key) == value for key, value in selector.items() if key != "derived_policy")
    ]
    if len(matches) != 1:
        raise ValueError(f"selector expected one row, observed {len(matches)}: {selector}")
    return matches[0]


def make_region_assertion(meta: dict[str, Any]) -> dict[str, str]:
    reference = "".join(str(int(bit)) for bit in meta["message_bits"])
    if len(reference) != 32:
        raise ValueError(f"expected 32 reference bits for {meta['input_image']}")
    return {
        "schema_version": "2",
        "alg": "wam_mit_regional",
        "reference_bits": reference,
        "reference_length": "32",
        "required_matches": "24",
        "payload_hash": hashlib.sha256(reference.encode("ascii")).hexdigest(),
        "bbox_csv": meta["bbox_csv"],
        "bitmap64": meta["bitmap64"],
    }


def crop_transform(source_size: list[int], box: list[int], output_size: list[int]) -> dict[str, str]:
    return {
        "schema_version": "1",
        "coordinate_system": "source-pixel-half-open",
        "source_size_px": ",".join(map(str, source_size)),
        "crop_box_px": ",".join(map(str, box)),
        "output_size_px": ",".join(map(str, output_size)),
        "resize_filter": "LANCZOS",
    }


def safe_crop_box(side: str, width: int, height: int, fraction: float = 0.01) -> list[int]:
    if side in {"left", "right"}:
        amount = max(1, round(width * fraction))
        return [amount, 0, width, height] if side == "left" else [0, 0, width - amount, height]
    amount = max(1, round(height * fraction))
    return [0, amount, width, height] if side == "top" else [0, 0, width, height - amount]


def resolve_cases(config: dict[str, Any], *, e2e_root: Path, c1_dir: Path) -> list[dict[str, Any]]:
    defense = e2e_root / "results" / "defense_e2e"
    benign = e2e_root / "results" / "benign_controls_v1_full"
    inpainting = e2e_root / "results" / "inpainting_e2e_v1_full"
    lineage = e2e_root / "results" / "lineage_ablation_v1"
    meta_rows = load_json(defense / "embed_meta.json")["images"]
    meta_by_image = {row["input_image"]: row for row in meta_rows}
    c1_rows = load_jsonl(c1_dir / "audit_rows.jsonl")
    benign_audit = load_json(benign / "audit.json")["rows"]
    benign_signed = load_json(benign / "signed_meta.json")["rows"]
    inpaint_audit = load_json(inpainting / "audit.json")["rows"]
    inpaint_signed_payload = load_json(inpainting / "signed_meta.json")
    inpaint_signed = inpaint_signed_payload.get("images", inpaint_signed_payload.get("rows", []))
    lineage_rows = load_jsonl(lineage / "rows.jsonl")
    resolved = []
    for ordinal, case in enumerate(config["cases"]):
        selector = case["selector"]
        source = case["source"]
        if source == "c1":
            old = find_one(c1_rows, selector)
            image = old["input_image"]
            child = c1_dir / old["output_relpath"]
            with Image.open(child) as child_image:
                output_size = list(child_image.size)
            transform = crop_transform(old["original_size"], old["crop_box_px"], output_size)
        elif source == "benign":
            old = find_one(benign_audit, selector)
            signed_meta = find_one(benign_signed, {
                "condition": selector["condition"],
                "input_image": selector["image"],
            })
            image = old["image"]
            child = benign / "unsigned" / old["condition"] / signed_meta["unsigned_output"]
            transform = None
            if old["condition"] == "safe_crop_1pct":
                width, height = int(signed_meta["width"]), int(signed_meta["height"])
                transform = crop_transform(
                    [width, height], safe_crop_box(old["crop_side"], width, height), [width, height]
                )
        elif source == "inpainting":
            old = find_one(inpaint_audit, selector)
            image = old["image"]
            signed_meta = find_one(
                inpaint_signed,
                {"input_image": image} if any("input_image" in row for row in inpaint_signed) else {"image": image},
            )
            unsigned_name = signed_meta.get("unsigned_inpainted") or signed_meta.get("inpainted_unsigned")
            if not unsigned_name:
                raise KeyError(f"inpainting metadata lacks unsigned output for {image}")
            child = inpainting / "inpainted_unsigned" / unsigned_name
            transform = None
        elif source == "lineage":
            old = find_one(lineage_rows, selector)
            image = old["image"]
            child = lineage / old["signed_output"]
            transform = crop_transform([1024, 1024], [410, 410, 614, 614], [1024, 1024])
        else:
            raise ValueError(f"unknown pilot source: {source}")
        meta = meta_by_image[image]
        parent = defense / "watermarked_unsigned" / meta["unsigned_image"]
        for required in (child, parent):
            if not required.is_file():
                raise FileNotFoundError(required)
        resolved.append({
            **case,
            "ordinal": ordinal,
            "image": image,
            "child_source": str(child),
            "parent_source": str(parent),
            "crop_transform": transform,
            "historical": old,
            "region_assertion": make_region_assertion(meta),
            "mask_coverage": meta.get("mask_coverage"),
        })
    if len(resolved) != 20 or len({case["case_id"] for case in resolved}) != 20:
        raise ValueError("pilot must resolve to exactly 20 unique case IDs")
    return resolved


def sanitize_pixels(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination, format="PNG")


def sign_case(
    case: dict[str, Any], *, output: Path, c2pa_python: Path, cert: Path, key: Path,
    input_roots: list[Path], resume: bool,
) -> Path:
    case_id = case["case_id"]
    child_pixels = output / "pixels" / f"{case_id}.png"
    child_signed = output / "signed" / f"{case_id}.png"
    parent_signed = output / "parents" / f"{case_id}-parent.png"
    request_path = output / "requests" / f"{case_id}.json"
    sanitize_pixels(Path(case["child_source"]), child_pixels)
    request = {
        "case_id": case_id,
        "topology": case["topology"],
        "manifest_semantics": case["manifest_semantics"],
        "child_pixels": str(child_pixels),
        "child_output": str(child_signed),
        "parent_pixels": case["parent_source"],
        "parent_output": str(parent_signed),
        "crop_transform": case["crop_transform"],
        "region_assertion": case["region_assertion"],
    }
    request_path.parent.mkdir(parents=True, exist_ok=True)
    if request_path.exists():
        if load_json(request_path) != request:
            raise RuntimeError(f"resume request mismatch: {request_path}")
    else:
        atomic_json(request_path, request)
    if child_signed.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite {child_signed}; use --resume")
        return child_signed
    completed = subprocess.run(
        [
            str(c2pa_python), str(SIGN_HELPER), "--request", str(request_path),
            "--cert", str(cert), "--key", str(key),
            "--output-root", str(output),
            *[value for root in input_roots for value in ("--input-root", str(root))],
        ],
        capture_output=True, text=True, check=False, shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"signing failed for {case_id}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    source_key, _ = decoded_rgb_key(child_pixels)
    signed_key, _ = decoded_rgb_key(child_signed)
    if source_key != signed_key:
        raise RuntimeError(f"signing changed decoded pixels for {case_id}")
    return child_signed


def inspect_signed(asset: Path, c2pa_python: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(c2pa_python), str(READ_HELPER), str(asset)],
        capture_output=True, text=True, check=False, shell=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"C2PA inspection failed for {asset}:\n{completed.stderr}")
    return json.loads(completed.stdout)


def unpack_mask(bitmap64: str) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(base64.b64decode(bitmap64), dtype=np.uint8))
    if bits.size != 64 * 64:
        raise ValueError(f"expected 4096 signed mask bits, observed {bits.size}")
    return bits.reshape(64, 64).astype(np.float32)


def mapped_mask_and_retention(region: dict[str, Any], geometry: dict[str, Any]) -> tuple[np.ndarray, float]:
    mask = unpack_mask(region["bitmap64"])
    if geometry.get("status") == "missing":
        return mask, 1.0
    if geometry.get("status") != "valid":
        raise ValueError(f"cannot map invalid geometry: {geometry}")
    transform = geometry["transform"]
    width, height = transform["source_size_px"]
    x0, y0, x1, y1 = transform["crop_box_px"]
    gh, gw = mask.shape
    gx0 = max(0, min(gw, int(math.floor(x0 * gw / width))))
    gy0 = max(0, min(gh, int(math.floor(y0 * gh / height))))
    gx1 = max(gx0 + 1, min(gw, int(math.ceil(x1 * gw / width))))
    gy1 = max(gy0 + 1, min(gh, int(math.ceil(y1 * gh / height))))
    kept = mask[gy0:gy1, gx0:gx1]
    retention = float(kept.sum() / max(float(mask.sum()), 1e-12))
    mapped = np.array(
        Image.fromarray((kept * 255).astype(np.uint8)).resize((64, 64), Image.Resampling.NEAREST),
        dtype=np.float32,
    ) / 255.0
    return (mapped > 0.5).astype(np.float32), retention


class Detector:
    def __init__(self, *, wam_repo: Path, params: Path, checkpoint: Path, device: str) -> None:
        import torch

        self.torch = torch
        self.wam_repo = wam_repo
        previous = Path.cwd()
        sys.path.insert(0, str(wam_repo))
        os.chdir(wam_repo)
        try:
            from notebooks.inference_utils import load_model_from_checkpoint
            self.model = load_model_from_checkpoint(str(params.resolve()), str(checkpoint.resolve()))
        finally:
            os.chdir(previous)
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        from watermark_anything.data.metrics import msg_predict_inference
        from watermark_anything.data.transforms import default_transform
        self.msg_predict_inference = msg_predict_inference
        self.default_transform = default_transform

    def run(self, asset: Path) -> tuple[str, np.ndarray]:
        image = Image.open(asset).convert("RGB")
        tensor = self.default_transform(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            predictions = self.model.detect(tensor)["preds"]
        logits = self.torch.sigmoid(predictions[:, 0])
        decoded = self.msg_predict_inference(predictions[:, 1:], logits).cpu().to(self.torch.int32)[0]
        bits = "".join(str(int(value)) for value in decoded.tolist())
        return bits, logits[0].detach().cpu().numpy()


def logit_mean(mask_logits: np.ndarray, mapped_mask: np.ndarray) -> float:
    import torch
    import torch.nn.functional as functional

    logits = torch.from_numpy(mask_logits.astype(np.float32))[None, None]
    resized = functional.interpolate(logits, size=(64, 64), mode="bilinear", align_corners=False)[0, 0]
    mask = torch.from_numpy(mapped_mask.astype(np.float32))
    return float((resized * mask).sum().item() / max(mask.sum().item(), 1e-12))


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_summary(output: Path, expected_cases: int) -> None:
    rows = []
    for path in sorted(output.glob("audit_rows.part-*.jsonl")):
        rows.extend(load_jsonl(path))
    unique = {row["case_id"]: row for row in rows}
    if len(unique) != expected_cases:
        return
    ordered = [unique[key] for key in sorted(unique)]
    merged = output / "audit_rows.jsonl"
    temporary = merged.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, merged)
    summary = {
        "case_count": len(ordered),
        "manifest_valid": sum(row["manifest_valid"] for row in ordered),
        "pixel_cache_hits": sum(row["pixel_cache_hit"] for row in ordered),
        "old_bit_count_matches": sum(row["old_bit_count_matches"] is True for row in ordered),
        "reference_states": {},
        "lineage_states": {},
        "evidence_states": {},
        "policy_verdicts": {},
    }
    for row in ordered:
        for source, field in [
            ("reference_states", "reference"), ("lineage_states", "lineage"),
            ("evidence_states", "evidence"), ("policy_verdicts", "policy_verdict"),
        ]:
            value = row[field]
            summary[source][value] = summary[source].get(value, 0) + 1
    atomic_json(output / "summary.json", summary)
    lines = [
        "# Protocol v2 boundary pilot",
        "",
        f"Cases: {len(ordered)}; valid manifests: {summary['manifest_valid']}; "
        f"historical integer-count matches: {summary['old_bit_count_matches']}.",
        "",
        "This is a deliberate boundary pilot, not a statistical estimate.",
        "The verifier reads references only from the signed current or ancestor manifest.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/protocol_v2_pilot.json")
    parser.add_argument("--e2e-root", type=Path, default=ROOT)
    parser.add_argument("--c1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/protocol_v2/pilot_v1")
    parser.add_argument("--c2pa-python", type=Path, required=True)
    parser.add_argument("--cert", type=Path, default=ROOT / "certs/ec_chain.pem")
    parser.add_argument("--key", type=Path, default=ROOT / "certs/ec_key.pem")
    parser.add_argument("--wam-repo", type=Path, required=True)
    parser.add_argument("--params-json", type=Path, default=ROOT / "ckpts/params.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "ckpts/wam_mit.pth")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--phase", choices=("prepare", "audit", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_json(args.config)
    cases = resolve_cases(config, e2e_root=args.e2e_root.resolve(), c1_dir=args.c1_dir.resolve())
    public_selection = [
        {key: value for key, value in case.items() if key not in {"region_assertion", "historical"}}
        for case in cases
    ]
    selection_path = output / "selected_samples.json"
    if selection_path.exists() and load_json(selection_path) != public_selection:
        raise RuntimeError("selected sample set changed during resume")
    if not selection_path.exists():
        atomic_json(selection_path, public_selection)
    shard = [case for case in cases if case["ordinal"] % args.num_shards == args.shard_id]

    if args.phase in {"prepare", "all"}:
        for case in shard:
            sign_case(
                case, output=output, c2pa_python=args.c2pa_python,
                cert=args.cert, key=args.key,
                input_roots=[args.e2e_root.resolve(), args.c1_dir.resolve()],
                resume=args.resume,
            )
            print(f"prepared {case['case_id']}", flush=True)
    if args.phase == "prepare":
        return

    rows_path = output / f"audit_rows.part-{args.shard_id:03d}.jsonl"
    if rows_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {rows_path}; use --resume")
    completed = {row["case_id"] for row in load_jsonl(rows_path)} if rows_path.exists() else set()
    pixel_cache = PixelCache(output / "pixel_cache")
    manifest_cache = ManifestCache(output / "manifest_cache")
    detector: Detector | None = None
    model_hash = sha256_file(args.params_json)
    checkpoint_hash = sha256_file(args.checkpoint)
    thresholds = config["thresholds"]
    for case in shard:
        if case["case_id"] in completed:
            continue
        signed = output / "signed" / f"{case['case_id']}.png"
        if not signed.is_file():
            raise FileNotFoundError(f"prepared signed asset missing: {signed}")
        inspection = inspect_signed(signed, args.c2pa_python)
        if not inspection["manifest_valid"]:
            raise RuntimeError(f"invalid manifest: {case['case_id']}")
        pixel_key, image_info = decoded_rgb_key(signed)
        cached = pixel_cache.get(pixel_key)
        cache_hit = cached is not None
        if cached is None:
            if detector is None:
                detector = Detector(
                    wam_repo=args.wam_repo, params=args.params_json,
                    checkpoint=args.checkpoint, device=args.device,
                )
            decoded_bits, mask_logits = detector.run(signed)
            pixel_cache.put(
                key=pixel_key, decoded_bits=decoded_bits, mask_logits=mask_logits,
                image=image_info, model_sha256=model_hash,
                checkpoint_sha256=checkpoint_hash,
                preprocessing_version="wam-default-transform-v1",
            )
            cached = pixel_cache.get(pixel_key)
        assert cached is not None
        reference_state = inspection["reference_source"]
        region = inspection.get("region")
        if reference_state in {"signed-current", "signed-ancestor"}:
            if inspection["region_assertion"]["status"] != "valid":
                raise RuntimeError(f"invalid signed region assertion: {case['case_id']}")
            reference_bits = inspection["region_assertion"]["reference_bits"]
            mapped_mask, retention = mapped_mask_and_retention(region, inspection["geometry"])
            regional_logit = logit_mean(cached["mask_logits"], mapped_mask)
        else:
            reference_bits = None
            retention = None
            regional_logit = None
        audit = audit_protocol_v2(
            lineage=inspection["lineage_state"], reference_state=reference_state,
            semantics=inspection["chain_semantics"], reference_bits=reference_bits,
            decoded_bits=cached["decoded_bits"] if reference_bits is not None else None,
            required_matches=int(thresholds["required_matches"]),
            region_retention=retention, logit_in_region=regional_logit,
        )
        oracle_reference = case["region_assertion"]["reference_bits"]
        oracle_count = matching_bit_count(oracle_reference, cached["decoded_bits"])
        old_accuracy = case.get("expected_old_bit_accuracy")
        old_count_matches = (
            None if old_accuracy is None else oracle_count == round(float(old_accuracy) * 32)
        )
        if old_count_matches is False:
            raise RuntimeError(
                f"decoded-bit mismatch for {case['case_id']}: old={old_accuracy}, new={oracle_count}/32"
            )
        manifest_sha = sha256_file(signed)
        manifest_payload = {
            "case_id": case["case_id"], "asset": str(signed),
            "lineage": audit["lineage"], "reference": audit["reference"],
            "semantics": audit["semantics"], "evidence": audit["evidence"],
            "policy_verdict": audit["policy_verdict"], "pixel_cache_key": pixel_key,
        }
        existing_manifest = manifest_cache.get(manifest_sha)
        if existing_manifest is None:
            manifest_cache.put(manifest_sha256=manifest_sha, payload=manifest_payload)
        elif any(existing_manifest.get(key) != value for key, value in manifest_payload.items()):
            raise RuntimeError(f"manifest cache conflict: {case['case_id']}")
        row = {
            "case_id": case["case_id"], "category": case["category"],
            "source": case["source"], "image": case["image"],
            "signed_asset": str(signed.relative_to(output)),
            "manifest_sha256": manifest_sha, "manifest_valid": True,
            "pixel_cache_key": pixel_key, "pixel_cache_hit": cache_hit,
            "decoded_bits": cached["decoded_bits"], "oracle_match_count": oracle_count,
            "oracle_usage": "evaluation-only",
            "old_bit_count_matches": old_count_matches,
            "region_retention": retention,
            "logit_in_region": regional_logit,
            **audit,
        }
        append_jsonl(rows_path, row)
        completed.add(case["case_id"])
        print(f"audited {case['case_id']} -> {audit['policy_verdict']}", flush=True)

    atomic_json(output / "environment.json", {
        "python": sys.version, "platform": platform.platform(),
        "torch": package_version("torch"), "numpy": package_version("numpy"),
        "pillow": package_version("Pillow"), "checkpoint_sha256": checkpoint_hash,
        "params_sha256": model_hash, "device_requested": args.device,
        "shard_id": args.shard_id, "num_shards": args.num_shards,
    })
    write_summary(output, len(cases))


if __name__ == "__main__":
    main()
