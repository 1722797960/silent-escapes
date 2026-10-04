"""Detect WAM after a common removed-area crop grid for one control condition."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image

from granularity_control_common_v1 import (
    append_jsonl,
    crop_and_resize,
    load_config,
    load_jsonl,
    validate_conditions,
)


ROOT = Path(__file__).resolve().parents[1]
WAM_REPO = ROOT / "third_party" / "watermark-anything"
DEFAULT_CONFIG = ROOT / "configs" / "granularity_control_v1.json"
DEFAULT_EXPERIMENT = ROOT / "results" / "granularity_control_v1" / "pilot"


def import_subject_cache(experiment: Path, config: dict, embed_rows: list[dict], c1_path: Path) -> None:
    output_path = experiment / "detection_subject_region.jsonl"
    if output_path.exists():
        raise SystemExit(f"{output_path} already exists")
    allowed = {row["input_image"] for row in embed_rows}
    rhos = {round(float(value), 8) for value in config["removed_area_fractions"]}
    source = [row for row in load_jsonl(c1_path)
              if row.get("method") == "center"
              and row.get("input_image") in allowed
              and round(float(row["rho_requested"]), 8) in rhos]
    expected = len(allowed) * len(rhos)
    if len(source) != expected:
        raise RuntimeError(f"C1 cache has {len(source)} matching rows; expected {expected}")
    index = {row["input_image"]: int(row["image_index"]) for row in embed_rows}
    source_cache = str(config.get("c1_cache", "results/c1_v1/full/audit_rows.jsonl")).replace("\\", "/")
    for cached in sorted(source, key=lambda row: (index[row["input_image"]], float(row["rho_requested"]))):
        matches = int(round(float(cached["bit_accuracy"]) * int(config["message_bits"])))
        row = {
            "row_id": f"{index[cached['input_image']]:04d}|subject_region|{float(cached['rho_requested']):.3f}",
            "image_index": index[cached["input_image"]],
            "input_image": cached["input_image"],
            "condition": "subject_region",
            "rho_requested": float(cached["rho_requested"]),
            "rho_actual": float(cached["rho_actual"]),
            "crop_box_px": cached["crop_box_px"],
            "matched_bits": matches,
            "total_bits": int(config["message_bits"]),
            "bit_accuracy": matches / int(config["message_bits"]),
            "detected": matches >= int(config["matched_bits_threshold"]),
            "source_cache": source_cache,
        }
        append_jsonl(output_path, row)
    print(f"imported {expected} subject-region rows from {c1_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--ckpt", default=str(ROOT / "ckpts" / "wam_mit.pth"))
    parser.add_argument("--params-json", default=str(ROOT / "ckpts" / "params.json"))
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "originals_500"))
    parser.add_argument("--c1-cache", default=str(ROOT / "results" / "c1_v1" / "full" / "audit_rows.jsonl"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--import-c1", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    conditions = validate_conditions(config["conditions"])
    if args.condition not in (*conditions, "clean_reference"):
        raise SystemExit(f"unsupported condition {args.condition!r}; expected {conditions}")
    experiment = Path(args.experiment_dir).resolve()
    embed_rows = load_jsonl(experiment / "embed_rows.jsonl")
    if not embed_rows:
        raise SystemExit("embed_rows.jsonl is empty or missing")
    if args.condition == "subject_region" and args.import_c1:
        import_subject_cache(experiment, config, embed_rows, Path(args.c1_cache).resolve())
        return

    output_path = experiment / f"detection_{args.condition}.jsonl"
    if output_path.exists() and not args.resume:
        raise SystemExit(f"{output_path} exists; use --resume")
    completed = {row["row_id"] for row in load_jsonl(output_path)} if args.resume else set()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, str(WAM_REPO))
    previous = os.getcwd()
    os.chdir(WAM_REPO)
    try:
        from notebooks.inference_utils import load_model_from_checkpoint
        model = load_model_from_checkpoint(str(Path(args.params_json).resolve()), str(Path(args.ckpt).resolve()))
    finally:
        os.chdir(previous)
    model.to(device).eval()
    from watermark_anything.data.metrics import msg_predict_inference
    from watermark_anything.data.transforms import default_transform

    rhos = [0.0] if args.condition == "clean_reference" else [float(v) for v in config["removed_area_fractions"]]
    input_dir = Path(args.input_dir).resolve()
    total = len(embed_rows) * len(rhos)
    done = 0
    for embed in embed_rows:
        if args.condition == "clean_reference":
            source_path = input_dir / embed["input_image"]
        else:
            source_path = experiment / embed["conditions"][args.condition]["image_relpath"]
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        base = Image.open(source_path).convert("RGB")
        message = torch.tensor(embed["message_bits"], dtype=torch.float32)
        for rho in rhos:
            row_id = f"{int(embed['image_index']):04d}|{args.condition}|{rho:.3f}"
            if row_id in completed:
                done += 1
                continue
            attacked, box, actual = crop_and_resize(base, rho)
            tensor = default_transform(attacked).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model.detect(tensor)["preds"]
            mask_probs = torch.sigmoid(preds[:, 0])
            predicted = msg_predict_inference(preds[:, 1:], mask_probs).cpu().float()[0]
            matches = int((predicted == message).sum().item())
            row = {
                "row_id": row_id,
                "image_index": int(embed["image_index"]),
                "input_image": embed["input_image"],
                "condition": args.condition,
                "rho_requested": rho,
                "rho_actual": actual,
                "crop_box_px": list(box),
                "matched_bits": matches,
                "total_bits": int(config["message_bits"]),
                "bit_accuracy": matches / int(config["message_bits"]),
                "detected": matches >= int(config["matched_bits_threshold"]),
                "source_cache": None,
            }
            append_jsonl(output_path, row)
            completed.add(row_id)
            done += 1
            if done % 25 == 0 or done == total:
                print(f"{args.condition}: {done}/{total}", flush=True)
    print(f"complete: {done}/{total}; rows={output_path}")


if __name__ == "__main__":
    main()
