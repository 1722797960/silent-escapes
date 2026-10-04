"""Create paired subject, random-region, and full-frame WAM controls."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from granularity_control_common_v1 import (
    append_jsonl,
    decoded_rgb_sha256,
    energy_match_fullframe,
    load_config,
    load_jsonl,
    mse,
    psnr_from_mse,
    quantize_rgb01,
    random_compact_mask_same_area,
    stable_seed,
    translate_mask_same_area,
    validate_conditions,
)


ROOT = Path(__file__).resolve().parents[1]
WAM_REPO = ROOT / "third_party" / "watermark-anything"
DEFAULT_CONFIG = ROOT / "configs" / "granularity_control_v1.json"
DEFAULT_OUTPUT = ROOT / "results" / "granularity_control_v1" / "pilot"


def save_rgb01(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Arrays passed here have already been quantized by quantize_rgb01.
    Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8), "RGB").save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--input-dir", default=str(ROOT / "data" / "originals_500"))
    parser.add_argument("--mask-dir", default=str(ROOT / "data" / "masks_200"))
    parser.add_argument("--e2e-dir", default=str(ROOT / "results" / "defense_e2e"))
    parser.add_argument("--ckpt", default=str(ROOT / "ckpts" / "wam_mit.pth"))
    parser.add_argument("--params-json", default=str(ROOT / "ckpts" / "params.json"))
    parser.add_argument("--num-images", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    conditions = validate_conditions(config["conditions"])
    random_condition = next(value for value in conditions if value.startswith("random_"))
    count = int(args.num_images or config["pilot_num_images"])
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "embed_rows.jsonl"
    if rows_path.exists() and not args.resume:
        raise SystemExit(f"{rows_path} exists; use --resume or a new output directory")
    completed = {int(row["image_index"]) for row in load_jsonl(rows_path)} if args.resume else set()

    e2e = Path(args.e2e_dir).resolve()
    input_dir = Path(args.input_dir).resolve()
    mask_dir = Path(args.mask_dir).resolve()
    required = [e2e / "embed_meta.json", input_dir, mask_dir, Path(args.ckpt), Path(args.params_json)]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required paths:\n" + "\n".join(missing))

    meta = json.loads((e2e / "embed_meta.json").read_text(encoding="utf-8"))["images"][:count]
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
    to_tensor = T.ToTensor()

    try:
        output_record = output.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        output_record = str(output)

    effective = {
        **config,
        "effective_num_images": len(meta),
        "output_dir": output_record,
        "device": device,
        "python": sys.version,
        "platform": platform.platform(),
        "started_unix": time.time(),
    }
    (output / "config_effective.json").write_text(
        json.dumps(effective, indent=2, ensure_ascii=False), encoding="utf-8")

    tolerance = float(config["energy_match_relative_tolerance"])
    for position, entry in enumerate(meta):
        image_index = int(entry["index"])
        if image_index in completed:
            continue
        original_path = input_dir / entry["input_image"]
        mask_path = mask_dir / f"{Path(entry['input_image']).stem}_mask.png"
        subject_existing = e2e / "watermarked_unsigned" / entry["unsigned_image"]
        if not all(path.exists() for path in (original_path, mask_path, subject_existing)):
            raise FileNotFoundError(f"missing paired input for {entry['input_image']}")

        original_pil = Image.open(original_path).convert("RGB")
        original_u8 = np.asarray(original_pil, dtype=np.uint8)
        original = original_u8.astype(np.float64) / 255.0
        height, width = original_u8.shape[:2]
        mask_raw = Image.open(mask_path).convert("L").resize((width, height), Image.Resampling.NEAREST)
        subject_mask = np.asarray(mask_raw, dtype=np.uint8) > 127
        seed_namespace = (
            "random-mask" if random_condition == "random_translated_region"
            else random_condition)
        random_seed = stable_seed(
            int(config["random_seed"]), entry["input_image"], seed_namespace)
        if random_condition == "random_compact_equal_area":
            random_mask, translation = random_compact_mask_same_area(
                subject_mask.shape, int(subject_mask.sum()), random_seed)
        elif random_condition == "random_translated_region":
            random_mask, translation = translate_mask_same_area(subject_mask, random_seed)
        else:
            raise ValueError(f"unsupported random control: {random_condition}")

        image_tensor = to_tensor(original_pil).unsqueeze(0).to(device)
        message = torch.tensor(entry["message_bits"], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            candidate_tensor = model.embed(image_tensor, message)["imgs_w"].clamp(0, 1)
        candidate = candidate_tensor[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
        subject = quantize_rgb01(np.where(subject_mask[..., None], candidate, original))
        random_region = quantize_rgb01(np.where(random_mask[..., None], candidate, original))
        native = quantize_rgb01(candidate)
        subject_mse = mse(subject, original)
        energy, alpha, energy_mse = energy_match_fullframe(original, candidate, subject_mse)
        relative_error = abs(energy_mse - subject_mse) / max(subject_mse, 1e-15)
        if relative_error > tolerance:
            raise RuntimeError(
                f"energy match failed for {entry['input_image']}: rel_error={relative_error:.6g}")

        stem = f"{image_index:04d}"
        paths = {
            "subject_region": output / "images" / "subject_region" / f"{stem}.png",
            random_condition: output / "images" / random_condition / f"{stem}.png",
            "fullframe_native": output / "images" / "fullframe_native" / f"{stem}.png",
            "fullframe_energy_matched": output / "images" / "fullframe_energy_matched" / f"{stem}.png",
        }
        arrays = {
            "subject_region": subject,
            random_condition: random_region,
            "fullframe_native": native,
            "fullframe_energy_matched": energy,
        }
        for condition in conditions:
            save_rgb01(paths[condition], arrays[condition])
        random_mask_path = output / "random_masks" / f"{stem}.png"
        random_mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(random_mask.astype(np.uint8) * 255, "L").save(random_mask_path)

        intersection = int(np.logical_and(subject_mask, random_mask).sum())
        union = int(np.logical_or(subject_mask, random_mask).sum())
        subject_centroid = np.argwhere(subject_mask).mean(axis=0)
        random_centroid = np.argwhere(random_mask).mean(axis=0)
        centroid_distance = float(np.linalg.norm(subject_centroid - random_centroid))
        centroid_distance_normalized = centroid_distance / float(np.hypot(height, width))

        regenerated_hash = decoded_rgb_sha256(paths["subject_region"])
        existing_hash = decoded_rgb_sha256(subject_existing)
        if regenerated_hash != existing_hash:
            raise RuntimeError(
                f"subject-region RGB hash mismatch for {entry['input_image']}: "
                f"existing={existing_hash}, regenerated={regenerated_hash}")

        condition_metrics = {}
        for condition in conditions:
            value = mse(arrays[condition], original)
            changed = subject_mask if condition == "subject_region" else (
                random_mask if condition == random_condition else np.ones_like(subject_mask))
            marked_diff = (arrays[condition] - original)[changed]
            marked_mse = float(np.mean(marked_diff * marked_diff)) if marked_diff.size else 0.0
            condition_metrics[condition] = {
                "mse": value,
                "psnr_db": psnr_from_mse(value),
                "marked_pixel_mse": marked_mse,
                "image_relpath": str(paths[condition].relative_to(output)).replace("\\", "/"),
            }

        row = {
            "image_index": image_index,
            "input_image": entry["input_image"],
            "message_bits": [int(bit) for bit in entry["message_bits"]],
            "subject_mask_pixels": int(subject_mask.sum()),
            "mask_coverage": float(subject_mask.mean()),
            "random_mask_relpath": str(random_mask_path.relative_to(output)).replace("\\", "/"),
            "random_translation": translation,
            "subject_random_iou": intersection / max(union, 1),
            "subject_random_centroid_distance_normalized": centroid_distance_normalized,
            "subject_existing_rgb_sha256": existing_hash,
            "subject_regenerated_rgb_sha256": regenerated_hash,
            "subject_hash_match": True,
            "energy_match_alpha": alpha,
            "energy_match_relative_error": relative_error,
            "conditions": condition_metrics,
        }
        append_jsonl(rows_path, row)
        completed.add(image_index)
        print(f"embedded {position + 1}/{len(meta)} {entry['input_image']}", flush=True)

    print(f"complete: {len(completed)}/{len(meta)}; rows={rows_path}")


if __name__ == "__main__":
    main()
