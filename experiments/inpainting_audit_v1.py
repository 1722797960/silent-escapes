"""Audit exact-mask LaMa outputs with the corrected region-aware logic."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from inpainting_common_v1 import classify_inpainting, sha256_file


BIT_THRESHOLD = 0.75
REGION_MIN_RETENTION = 0.05
LOGIT_MEAN_MIN = 0.5


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, default=root / "results/inpainting_e2e_v1")
    parser.add_argument("--reference-audit", type=Path, default=root / "results/defense_e2e/audit_retention.json")
    parser.add_argument("--c2pa-python", type=Path, required=True)
    parser.add_argument("--wam-repo", type=Path, default=root / "third_party/watermark-anything")
    parser.add_argument("--ckpt", type=Path, default=root / "ckpts/wam_mit.pth")
    parser.add_argument("--params-json", type=Path, default=root / "ckpts/params.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    signed_meta_path = args.experiment_dir / "signed_meta.json"
    payload = json.loads(signed_meta_path.read_text(encoding="utf-8"))
    reference_payload = json.loads(args.reference_audit.read_text(encoding="utf-8"))
    reference = {
        row["image"]: row
        for row in reference_payload["rows"]
        if float(row["crop_frac"]) == 0.0
    }

    wam_repo = args.wam_repo.resolve()
    if not (wam_repo / "notebooks/inference_utils.py").is_file():
        raise FileNotFoundError(f"invalid WAM repository: {wam_repo}")
    sys.path.insert(0, str(wam_repo))
    previous = Path.cwd()
    os.chdir(wam_repo)
    try:
        from notebooks.inference_utils import load_model_from_checkpoint

        model = load_model_from_checkpoint(str(args.params_json.resolve()), str(args.ckpt.resolve()))
    finally:
        os.chdir(previous)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model.to(device).eval()
    from watermark_anything.data.metrics import msg_predict_inference
    from watermark_anything.data.transforms import default_transform

    read_helper = root / "experiments/read_region_assertion.py"
    rows = []
    for ordinal, entry in enumerate(payload["images"], start=1):
        if entry["width"] <= 0 or entry["height"] <= 0 or entry["outside_diff_max"] != 0:
            raise AssertionError(f"invalid geometry invariant: {entry['input_image']}")
        signed_path = args.experiment_dir / "inpainted_signed" / entry["signed_inpainted"]
        assertion = json.loads(
            subprocess.run(
                [str(args.c2pa_python), str(read_helper), str(signed_path)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
        region = assertion.get("region") or {}
        message_string = "".join(str(bit) for bit in entry["message_bits"])
        payload_hash_ok = region.get("payload_hash") == hashlib.sha256(
            message_string.encode("utf-8")
        ).hexdigest()

        packed = np.frombuffer(base64.b64decode(entry["bitmap64"]), dtype=np.uint8)
        grid = np.unpackbits(packed).reshape(64, 64).astype(np.float32)
        region_retention = 1.0
        image = Image.open(signed_path).convert("RGB")
        if image.size != (entry["width"], entry["height"]):
            raise AssertionError(f"signed image size changed: {signed_path}")
        tensor = default_transform(image).unsqueeze(0).to(device)
        message = torch.tensor(entry["message_bits"], dtype=torch.float32)
        with torch.inference_mode():
            predictions = model.detect(tensor)["preds"]
        mask_logits = torch.sigmoid(predictions[:, 0])
        predicted_message = msg_predict_inference(
            predictions[:, 1:], mask_logits
        ).cpu().float()[0]
        bit_accuracy = float((predicted_message == message).float().mean().item())
        resized_logits = F.interpolate(
            mask_logits.unsqueeze(1), size=grid.shape, mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        logit_in_region = float((resized_logits * grid).sum() / max(grid.sum(), 1e-6))
        baseline, region_aware = classify_inpainting(
            bit_accuracy,
            bool(assertion["region_assertion_present"]),
            payload_hash_ok,
            region_retention,
            logit_in_region,
            BIT_THRESHOLD,
            REGION_MIN_RETENTION,
            LOGIT_MEAN_MIN,
        )
        before = reference.get(entry["input_image"], {})
        rows.append(
            {
                "image": entry["input_image"],
                "signed_inpainted": entry["signed_inpainted"],
                "mask_coverage": entry["mask_coverage"],
                "manifest_valid": bool(assertion["manifest_valid"]),
                "region_assertion_present": bool(assertion["region_assertion_present"]),
                "payload_hash_ok": payload_hash_ok,
                "geometry_unchanged": True,
                "outside_diff_max": entry["outside_diff_max"],
                "region_retention": region_retention,
                "bit_accuracy_before": before.get("bit_accuracy"),
                "bit_accuracy_after": round(bit_accuracy, 4),
                "logit_in_region_before": before.get("logit_in_region"),
                "logit_in_region_after": round(logit_in_region, 4),
                "baseline_audit": baseline,
                "region_aware_audit": region_aware,
            }
        )
        print(
            f"[{ordinal:02d}/{len(payload['images']):02d}] acc={bit_accuracy:.4f} "
            f"logit={logit_in_region:.4f} {baseline}/{region_aware}",
            flush=True,
        )

    result = {
        "experiment": "exact-mask LaMa inpainting audit",
        "version": "inpainting_e2e_v1",
        "bit_threshold": BIT_THRESHOLD,
        "region_min_retention": REGION_MIN_RETENTION,
        "logit_mean_min": LOGIT_MEAN_MIN,
        "region_metric": "unchanged geometry => retained_mask_pixels/original_mask_pixels = 1",
        "signed_metadata_sha256": sha256_file(signed_meta_path),
        "reference_audit_sha256": sha256_file(args.reference_audit),
        "wam_checkpoint_sha256": sha256_file(args.ckpt),
        "rows": rows,
    }
    output_path = args.experiment_dir / "audit.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
