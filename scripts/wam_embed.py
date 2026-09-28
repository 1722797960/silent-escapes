"""Embed WAM (Watermark Anything) watermarks into images.

Reads PNGs from an input directory, embeds a random 32-bit message per image
using the WAM model (full-image watermark, mask=1), writes watermarked images
to an output directory, and saves per-image embedded message bits to a
metadata JSON file (same contract as integrity-clash watermark_embed.py).

This script lives in integrity-clash-wam/ and only READS the WAM repo code;
it never modifies the source repository.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch
import torchvision.transforms as T
from PIL import Image

# Read-only import of the vendored WAM code (no files modified there).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAM_REPO = os.path.join(_REPO_ROOT, "third_party", "watermark-anything")
if WAM_REPO not in sys.path:
    sys.path.insert(0, WAM_REPO)
from notebooks.inference_utils import load_model_from_checkpoint  # noqa: E402


def get_device(requested: str) -> str:
    if requested and requested.lower() != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed WAM watermarks (full-image) into images."
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--metadata-path", type=str, default=None)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--params-json", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata_path = (
        args.metadata_path
        if args.metadata_path is not None
        else os.path.join(args.output_dir, "watermark_metadata.json")
    )

    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device(args.device)
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)

    # WAM configs are repo-relative; chdir only during load (read-only).
    abs_params = os.path.abspath(args.params_json)
    abs_ckpt = os.path.abspath(args.ckpt)
    prev_cwd = os.getcwd()
    os.chdir(WAM_REPO)
    try:
        model = load_model_from_checkpoint(abs_params, abs_ckpt)
    finally:
        os.chdir(prev_cwd)
    model.to(device)
    model.eval()

    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()

    image_files = sorted(
        f for f in os.listdir(args.input_dir) if f.lower().endswith(".png")
    )

    nbits = model.msg_processor.nbits if hasattr(model, "msg_processor") else 32

    all_metadata: List[Dict[str, Any]] = []
    for idx, image_name in enumerate(image_files):
        input_path = os.path.join(args.input_dir, image_name)
        img = Image.open(input_path).convert("RGB")
        img_tensor = to_tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            msgs = model.get_random_msg(1)  # b x k
            out = model.embed(img_tensor, msgs)
            img_w = out["imgs_w"].clamp(0, 1).squeeze(0).cpu()

        name_root, _ = os.path.splitext(image_name)
        wm_name = f"{name_root}_watermarked.png"
        to_pil(img_w).save(os.path.join(args.output_dir, wm_name))

        msg_bits = msgs.squeeze(0).to(torch.int32).tolist()
        all_metadata.append(
            {
                "index": idx,
                "input_image": image_name,
                "watermarked_image": wm_name,
                "message_bits": msg_bits,
                "watermark_method": "wam_mit",
                "nbits": len(msg_bits),
            }
        )
        print(f"[{idx:04d}] {image_name} -> {wm_name}")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({"images": all_metadata}, f, indent=2)
    print(f"Wrote metadata for {len(all_metadata)} images to: {metadata_path}")


if __name__ == "__main__":
    main()
