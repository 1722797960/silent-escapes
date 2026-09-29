"""Generate deterministic benign edits and paired unwatermarked controls."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benign_controls_common_v1 import decode_bitmap64, select_max_retention_crop
from inpainting_common_v1 import select_stratified_indices, sha256_file


WATERMARKED_CONDITIONS = (
    "identity_resave",
    "jpeg_q90",
    "resize_90",
    "safe_crop_1pct",
)
CLEAN_CONDITION = "clean_negative"


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=0)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").copy()


def resize_roundtrip(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    reduced = image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.Resampling.LANCZOS,
    )
    return reduced.resize((width, height), Image.Resampling.LANCZOS)


def one_sided_crop(image: Image.Image, side: str, fraction: float) -> Image.Image:
    width, height = image.size
    if side in ("left", "right"):
        pixels = max(1, round(width * fraction))
        box = (pixels, 0, width, height) if side == "left" else (0, 0, width - pixels, height)
    else:
        pixels = max(1, round(height * fraction))
        box = (0, pixels, width, height) if side == "top" else (0, 0, width, height - pixels)
    return image.crop(box).resize((width, height), Image.Resampling.LANCZOS)


def save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    reopened = Image.open(path).convert("RGB")
    if reopened.size != image.size:
        raise AssertionError(f"saved geometry changed: {path}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-dir", type=Path, default=root / "results/defense_e2e")
    parser.add_argument("--originals-dir", type=Path, default=root / "data/originals_500")
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results/benign_controls_v1_pilot"
    )
    parser.add_argument("--pilot-count", type=int, default=10)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--resize-scale", type=float, default=0.90)
    parser.add_argument("--safe-crop-fraction", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    meta_path = args.e2e_dir / "embed_meta.json"
    entries = json.loads(meta_path.read_text(encoding="utf-8"))["images"]
    selected = select_stratified_indices(
        [float(entry["mask_coverage"]) for entry in entries], args.pilot_count
    )
    output_meta = args.output_dir / "generate_meta.json"
    if output_meta.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_meta}")

    rows = []
    for ordinal, source_index in enumerate(selected, start=1):
        entry = entries[source_index]
        source_path = args.e2e_dir / "watermarked_signed_ra" / entry["signed_image"]
        clean_path = args.originals_dir / entry["input_image"]
        source = Image.open(source_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")
        if source.size != clean.size:
            raise ValueError(f"paired image size mismatch: {entry['input_image']}")

        grid = decode_bitmap64(entry["bitmap64"])
        crop_side, crop_retention = select_max_retention_crop(
            grid, args.safe_crop_fraction
        )
        variants = {
            "identity_resave": (source.copy(), 1.0, None),
            "jpeg_q90": (jpeg_roundtrip(source, args.jpeg_quality), 1.0, None),
            "resize_90": (resize_roundtrip(source, args.resize_scale), 1.0, None),
            "safe_crop_1pct": (
                one_sided_crop(source, crop_side, args.safe_crop_fraction),
                crop_retention,
                crop_side,
            ),
            CLEAN_CONDITION: (clean.copy(), None, None),
        }

        for condition, (image, retention, side) in variants.items():
            stem = Path(entry["input_image"]).stem
            output_name = f"{stem}_benign_{condition}.png"
            destination = args.output_dir / "unsigned" / condition / output_name
            if destination.exists() and not args.overwrite:
                raise FileExistsError(f"Refusing to overwrite {destination}")
            save_png(image, destination)
            rows.append(
                {
                    "selection_ordinal": ordinal,
                    "source_index": source_index,
                    "input_image": entry["input_image"],
                    "signed_source": entry["signed_image"],
                    "condition": condition,
                    "watermarked": condition != CLEAN_CONDITION,
                    "expected_region_assertion": condition != CLEAN_CONDITION,
                    "unsigned_output": output_name,
                    "width": image.width,
                    "height": image.height,
                    "message_bits": entry["message_bits"],
                    "payload_hash": entry["payload_hash"],
                    "bbox_csv": entry["bbox_csv"],
                    "bitmap64": entry["bitmap64"],
                    "mask_coverage": entry["mask_coverage"],
                    "region_retention": retention,
                    "crop_side": side,
                    "output_sha256": sha256_file(destination),
                }
            )
        print(
            f"[{ordinal:02d}/{len(selected):02d}] {entry['input_image']} "
            f"safe-crop={crop_side} retention={crop_retention:.4f}",
            flush=True,
        )

    payload = {
        "experiment": "benign edit controls",
        "version": "benign_controls_v1",
        "selection": "deterministic mask-coverage quantiles",
        "pilot_count": args.pilot_count,
        "conditions": list(WATERMARKED_CONDITIONS) + [CLEAN_CONDITION],
        "jpeg_quality": args.jpeg_quality,
        "jpeg_subsampling": 0,
        "resize_scale": args.resize_scale,
        "safe_crop_fraction": args.safe_crop_fraction,
        "safe_crop_policy": "one of four sides with maximum signed-mask retention",
        "source_metadata_sha256": sha256_file(meta_path),
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_meta.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {output_meta} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
