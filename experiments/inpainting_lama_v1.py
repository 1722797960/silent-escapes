"""Run LaMa on the exact signed SAM region while preserving outside pixels.

The primary experiment deliberately uses no mask dilation. LaMa predicts the
masked content, then the script composites the prediction into the original
uint8 image so every pixel outside the signed mask remains byte-identical.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from inpainting_common_v1 import select_stratified_indices, sha256_file


def load_lama(lama_source: Path, model_dir: Path, device: torch.device):
    sys.path.insert(0, str(lama_source.resolve()))
    from saicinpainting.training.trainers import load_checkpoint

    config = OmegaConf.load(model_dir / "config.yaml")
    config.training_model.predict_only = True
    config.visualizer.kind = "noop"
    model = load_checkpoint(
        config,
        model_dir / "models" / "best.ckpt",
        strict=False,
        map_location="cpu",
    )
    model.freeze()
    model.to(device)
    return model


def run_lama(model, image_u8: np.ndarray, mask: np.ndarray, device: torch.device) -> np.ndarray:
    image = torch.from_numpy(image_u8.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    batch = {"image": image.to(device), "mask": mask_tensor.to(device)}
    with torch.inference_mode():
        prediction = model(batch)["inpainted"][0].detach().clamp(0, 1)
    predicted_u8 = prediction.mul(255.0).round().byte().cpu().numpy().transpose(1, 2, 0)
    # Enforce the experiment's causal boundary: only signed-mask pixels change.
    return np.where(mask[..., None], predicted_u8, image_u8)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-dir", type=Path, default=root / "results/defense_e2e")
    parser.add_argument("--mask-dir", type=Path, default=root / "data/masks_200")
    parser.add_argument("--output-dir", type=Path, default=root / "results/inpainting_e2e_v1")
    parser.add_argument("--lama-source", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--pilot-count", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    meta_path = args.e2e_dir / "embed_meta.json"
    entries = json.loads(meta_path.read_text(encoding="utf-8"))["images"]
    selected = select_stratified_indices(
        [float(entry["mask_coverage"]) for entry in entries], args.pilot_count
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    unsigned_dir = args.output_dir / "inpainted_unsigned"
    unsigned_dir.mkdir(parents=True, exist_ok=True)
    meta_out = args.output_dir / "inpaint_meta.json"
    if meta_out.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {meta_out}; pass --overwrite explicitly")

    model = load_lama(args.lama_source, args.model_dir, device)
    completed = []
    for ordinal, index in enumerate(selected, start=1):
        entry = entries[index]
        source_path = args.e2e_dir / "watermarked_signed_ra" / entry["signed_image"]
        mask_name = Path(entry["input_image"]).stem + "_mask.png"
        mask_path = args.mask_dir / mask_name
        # PIL may expose a read-only NumPy view.  LaMa does not mutate it, but
        # own the buffer explicitly so torch.from_numpy never relies on that
        # implementation detail.
        image_u8 = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.uint8).copy()
        mask_u8 = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)
        if mask_u8.shape != image_u8.shape[:2]:
            raise ValueError(f"size mismatch for {entry['input_image']}: {image_u8.shape} vs {mask_u8.shape}")
        mask = mask_u8 > 127
        if not mask.any():
            raise ValueError(f"empty mask: {mask_path}")

        result_u8 = run_lama(model, image_u8, mask, device)
        outside = ~mask
        outside_diff_max = int(
            np.abs(result_u8[outside].astype(np.int16) - image_u8[outside].astype(np.int16)).max(initial=0)
        )
        if outside_diff_max != 0:
            raise AssertionError(f"outside-mask pixels changed for {entry['input_image']}")

        output_name = Path(entry["signed_image"]).stem.replace("_signed_ra", "") + "_inpaint.png"
        output_path = unsigned_dir / output_name
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output_path}")
        Image.fromarray(result_u8, mode="RGB").save(output_path, format="PNG")
        reopened = np.asarray(Image.open(output_path).convert("RGB"), dtype=np.uint8)
        if reopened.shape != image_u8.shape or not np.array_equal(reopened[outside], image_u8[outside]):
            raise AssertionError(f"saved output violated geometry/outside-mask invariant: {output_path}")

        completed.append(
            {
                "selection_ordinal": ordinal,
                "source_index": index,
                "input_image": entry["input_image"],
                "signed_source": entry["signed_image"],
                "mask_image": mask_name,
                "unsigned_inpainted": output_name,
                "width": int(image_u8.shape[1]),
                "height": int(image_u8.shape[0]),
                "mask_coverage": float(mask.mean()),
                "outside_diff_max": outside_diff_max,
                "source_sha256": sha256_file(source_path),
                "mask_sha256": sha256_file(mask_path),
                "output_sha256": sha256_file(output_path),
                "message_bits": entry["message_bits"],
                "payload_hash": entry["payload_hash"],
                "bbox_csv": entry["bbox_csv"],
                "bitmap64": entry["bitmap64"],
            }
        )
        print(f"[{ordinal:02d}/{len(selected):02d}] {entry['input_image']} coverage={mask.mean():.4f}", flush=True)

    payload = {
        "experiment": "exact-mask LaMa inpainting",
        "version": "inpainting_e2e_v1",
        "mask_dilation_pixels": 0,
        "selection": "deterministic mask-coverage quantiles",
        "device": str(device),
        "lama_source": str(args.lama_source.resolve()),
        "lama_commit": os.environ.get("LAMA_COMMIT", "unknown"),
        "checkpoint_path": str((args.model_dir / "models/best.ckpt").resolve()),
        "checkpoint_sha256": sha256_file(args.model_dir / "models/best.ckpt"),
        "source_metadata_sha256": sha256_file(meta_path),
        "images": completed,
    }
    meta_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {meta_out}")


if __name__ == "__main__":
    main()
