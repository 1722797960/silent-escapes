"""Audit C1 attacked assets with corrected region-retention semantics."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
WAM_REPO = ROOT / "third_party" / "watermark-anything"
DEFAULT_E2E = ROOT / "results" / "defense_e2e"
DEFAULT_C1 = ROOT / "results" / "c1_v1"
DEFAULT_C2PA_PY = ROOT / "venvs" / "c2pa" / "bin" / "python"
READ_HELPER = ROOT / "experiments" / "read_region_assertion.py"


def unpack_mask(bitmap64: str) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(base64.b64decode(bitmap64), dtype=np.uint8))
    if bits.size != 64 * 64:
        raise ValueError(f"expected 4096 mask bits, got {bits.size}")
    return bits.reshape(64, 64).astype(np.float32)


def transform_mask(mask: np.ndarray, box: list[int], width: int,
                   height: int) -> tuple[np.ndarray, float]:
    """Crop an original-coordinate mask and resize it into attacked-image coords."""
    x0, y0, x1, y1 = box
    gh, gw = mask.shape
    gx0 = max(0, min(gw, int(math.floor(x0 * gw / width))))
    gy0 = max(0, min(gh, int(math.floor(y0 * gh / height))))
    gx1 = max(gx0 + 1, min(gw, int(math.ceil(x1 * gw / width))))
    gy1 = max(gy0 + 1, min(gh, int(math.ceil(y1 * gh / height))))
    kept = mask[gy0:gy1, gx0:gx1]
    retention = float(kept.sum() / max(mask.sum(), 1e-6))
    mapped = np.array(
        Image.fromarray((kept * 255).astype(np.uint8)).resize(
            (gw, gh), Image.Resampling.NEAREST), dtype=np.float32) / 255.0
    mapped = (mapped > 0.5).astype(np.float32)
    return mapped, retention


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1-dir", default=str(DEFAULT_C1))
    parser.add_argument("--e2e-dir", default=str(DEFAULT_E2E))
    parser.add_argument("--ckpt", default=str(ROOT / "ckpts" / "wam_mit.pth"))
    parser.add_argument("--params-json", default=str(ROOT / "ckpts" / "params.json"))
    parser.add_argument("--c2pa-python", default=os.environ.get("C2PA_PY", str(DEFAULT_C2PA_PY)))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    c1_dir = Path(args.c1_dir).resolve()
    e2e_dir = Path(args.e2e_dir).resolve()
    records_path = c1_dir / "attack_manifest.jsonl"
    output_path = c1_dir / "audit_rows.jsonl"
    config = json.loads((c1_dir / "config_effective.json").read_text(encoding="utf-8"))
    threshold = float(config["watermark_threshold"])
    retention_floor = float(config["region_retention_floor"])
    suppression_threshold = float(config["suppression_threshold"])

    if output_path.exists() and not args.resume:
        raise SystemExit(f"{output_path} already exists; use --resume or a new output directory")
    completed = set()
    if args.resume and output_path.exists():
        completed = {row["attack_id"] for row in load_jsonl(output_path)}

    # Preserve the venv entry-point path. Resolving its python symlink would
    # bypass pyvenv.cfg and run the base interpreter without c2pa installed.
    c2pa_python = Path(args.c2pa_python).expanduser()
    if not c2pa_python.is_absolute():
        c2pa_python = Path.cwd() / c2pa_python
    required = [records_path, e2e_dir / "embed_meta.json", Path(args.ckpt),
                Path(args.params_json), c2pa_python, READ_HELPER]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required paths:\n" + "\n".join(missing))

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sys.path.insert(0, str(WAM_REPO))
    previous = os.getcwd()
    abs_params = str(Path(args.params_json).resolve())
    abs_ckpt = str(Path(args.ckpt).resolve())
    os.chdir(WAM_REPO)
    try:
        from notebooks.inference_utils import load_model_from_checkpoint
        model = load_model_from_checkpoint(abs_params, abs_ckpt)
    finally:
        os.chdir(previous)
    model.to(device).eval()
    from watermark_anything.data.metrics import msg_predict_inference
    from watermark_anything.data.transforms import default_transform

    meta_rows = json.loads((e2e_dir / "embed_meta.json").read_text(encoding="utf-8"))["images"]
    meta = {entry["input_image"]: entry for entry in meta_rows}
    attacks = load_jsonl(records_path)
    total = len(attacks)
    done = 0

    for attack in attacks:
        if attack["attack_id"] in completed:
            done += 1
            continue
        entry = meta[attack["input_image"]]
        attacked_path = c1_dir / attack["output_relpath"]
        if not attacked_path.exists():
            raise FileNotFoundError(attacked_path)

        reader = subprocess.run(
            [str(c2pa_python), str(READ_HELPER), str(attacked_path)],
            capture_output=True, text=True)
        if reader.returncode != 0:
            raise RuntimeError(
                f"C2PA reader failed for {attack['attack_id']}:\n{reader.stderr}")
        assertion = json.loads(reader.stdout)
        manifest_valid = bool(assertion["manifest_valid"])
        assertion_present = bool(assertion["region_assertion_present"])
        region = assertion.get("region") or {}
        msg_string = "".join(str(bit) for bit in entry["message_bits"])
        expected_hash = hashlib.sha256(msg_string.encode("utf-8")).hexdigest()
        payload_hash_ok = region.get("payload_hash") == expected_hash
        if not (manifest_valid and assertion_present and payload_hash_ok):
            raise RuntimeError(
                f"strict validation failed for {attack['attack_id']}: "
                f"manifest={manifest_valid}, assertion={assertion_present}, hash={payload_hash_ok}")

        original_width, original_height = attack["original_size"]
        original_mask = unpack_mask(entry["bitmap64"])
        mapped_mask, region_retention = transform_mask(
            original_mask, attack["crop_box_px"], original_width, original_height)
        region_occupancy = float(mapped_mask.mean())
        geometry_retention = float(attack["mask_retention_geometry"])
        if abs(region_retention - geometry_retention) > 1e-6:
            raise RuntimeError(
                f"geometry mismatch for {attack['attack_id']}: "
                f"audit={region_retention}, generation={geometry_retention}")

        image = Image.open(attacked_path).convert("RGB")
        tensor = default_transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model.detect(tensor)["preds"]
        mask_logits = torch.sigmoid(preds[:, 0])
        predicted = msg_predict_inference(preds[:, 1:], mask_logits).cpu().float()[0]
        message = torch.tensor(entry["message_bits"], dtype=torch.float32)
        bit_accuracy = float((predicted == message).float().mean().item())

        mapped_tensor = torch.from_numpy(mapped_mask).to(mask_logits.device)
        logits_64 = F.interpolate(mask_logits.unsqueeze(1), size=mapped_mask.shape,
                                  mode="bilinear", align_corners=False)[0, 0]
        logit_in_region = float(
            (logits_64 * mapped_tensor).sum().item() /
            max(mapped_tensor.sum().item(), 1e-6))

        deployed = "INTEGRITY_CLASH" if bit_accuracy >= threshold else "SILENT_ESCAPE"
        if bit_accuracy >= threshold:
            region_aware = "INTEGRITY_CLASH"
        elif region_retention < retention_floor:
            region_aware = "TAMPERED_REGION"
        elif logit_in_region < suppression_threshold:
            region_aware = "WATERMARK_SUPPRESSED"
        else:
            region_aware = "SILENT_OTHER"

        row = {
            **attack,
            "manifest_valid": manifest_valid,
            "region_assertion_present": assertion_present,
            "payload_hash_ok": payload_hash_ok,
            "bit_accuracy": round(bit_accuracy, 6),
            "detected": bit_accuracy >= threshold,
            "region_retention": round(region_retention, 8),
            "region_occupancy": round(region_occupancy, 8),
            "mask_removed_fraction": round(1.0 - region_retention, 8),
            "logit_in_region": round(logit_in_region, 6),
            "audit_v1_deployed": deployed,
            "audit_v2_region_aware": region_aware,
        }
        append_jsonl(output_path, row)
        completed.add(attack["attack_id"])
        done += 1
        if done % 25 == 0 or done == total:
            print(f"audited {done}/{total}", flush=True)

    print(f"complete: {done}/{total}; rows={output_path}")


if __name__ == "__main__":
    main()
