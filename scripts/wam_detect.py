"""Detect WAM watermarks and report bit accuracy.

Same output contract as integrity-clash watermark_detect.py: given a
metadata JSON produced by wam_embed.py and a directory of PNGs (plain
watermarked images or signed variants named
`<watermarked_stem>_signed_manifest_<manifest>.png`), decode the message per
image and write per-image + summary bit accuracies.

Detection details: WAM's extractor outputs per-pixel mask logit (channel 0)
and per-pixel message bits (channels 1..nbits). For a full-image watermark
(mask=1 during embed) we take a masked majority vote over pixels whose
predicted mask probability exceeds 0.5; if no such pixel exists we fall back
to a global majority over all pixels.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import torch
import torchvision.transforms as T
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAM_REPO = os.path.join(_REPO_ROOT, "third_party", "watermark-anything")
if WAM_REPO not in sys.path:
    sys.path.insert(0, WAM_REPO)
from notebooks.inference_utils import load_model_from_checkpoint  # noqa: E402


def get_device(requested: str) -> str:
    if requested and requested.lower() != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def decode_message(model: torch.nn.Module, img_tensor: torch.Tensor) -> torch.Tensor:
    """Return decoded bit vector (0/1) for one image."""
    with torch.no_grad():
        preds = model.detect(img_tensor)["preds"]  # 1 x (1+nbits) x H x W
    mask_probs = torch.sigmoid(preds[0, 0])  # H x W
    bit_logits = preds[0, 1:]  # nbits x H x W
    mask = (mask_probs > 0.5).flatten()
    bits = (bit_logits.flatten(1) > 0.0)  # nbits x (H*W)
    if mask.any():
        votes = bits[:, mask].float().mean(dim=1)
    else:
        votes = bits.float().mean(dim=1)
    return (votes > 0.5).float()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect WAM watermarks.")
    parser.add_argument("--original-dir", type=str, default=None)
    parser.add_argument("--watermarked-dir", type=str, required=True)
    parser.add_argument("--metadata-path", type=str, required=True)
    parser.add_argument("--output-json", type=str, default="wam_detection_results.json")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--params-json", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    with open(args.metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    image_entries: List[Dict[str, Any]] = metadata.get("images", [])
    if not image_entries:
        print(f"No image entries found in metadata: {args.metadata_path}")
        return

    device = get_device(args.device)
    print(f"Using device: {device}")

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

    results_per_image: List[Dict[str, Any]] = []
    acc_original_list: List[float] = []
    acc_watermarked_list: List[float] = []

    for entry in image_entries:
        idx = entry.get("index")
        input_name = entry.get("input_image")
        wm_name = entry.get("watermarked_image")
        msg_bits = entry.get("message_bits")
        if msg_bits is None:
            print(f"[{idx:04d}] {input_name} - missing message_bits, skipping.")
            continue
        msg_true = torch.tensor(msg_bits, dtype=torch.float32)

        # Resolve target image: exact watermarked name, else signed variant.
        wm_path = os.path.join(args.watermarked_dir, wm_name)
        if not os.path.isfile(wm_path):
            name_root, ext = os.path.splitext(wm_name)
            candidate_path = None
            for entry_name in os.listdir(args.watermarked_dir):
                if not entry_name.lower().endswith(ext.lower()):
                    continue
                if entry_name.startswith(f"{name_root}_signed_manifest_"):
                    candidate_path = os.path.join(args.watermarked_dir, entry_name)
                    break
            if candidate_path is None:
                print(f"[{idx:04d}] {wm_name} - no match in {args.watermarked_dir}, skipping.")
                continue
            wm_path = candidate_path

        orig_path = None
        if args.original_dir is not None and input_name is not None:
            cand = os.path.join(args.original_dir, input_name)
            if os.path.isfile(cand):
                orig_path = cand

        print(f"[{idx:04d}] input={input_name} watermarked={os.path.basename(wm_path)}")

        bit_acc_before = None
        if orig_path is not None:
            img_orig = to_tensor(Image.open(orig_path).convert("RGB")).unsqueeze(0).to(device)
            decoded_before = decode_message(model, img_orig).cpu()
            bit_acc_before = (decoded_before == msg_true).float().mean().item()
            acc_original_list.append(bit_acc_before)

        img_wm = to_tensor(Image.open(wm_path).convert("RGB")).unsqueeze(0).to(device)
        decoded_after = decode_message(model, img_wm).cpu()
        bit_acc_after = (decoded_after == msg_true).float().mean().item()
        acc_watermarked_list.append(bit_acc_after)

        results_per_image.append(
            {
                "index": idx,
                "input_image": input_name,
                "watermarked_image": wm_name,
                "bit_accuracy_original": bit_acc_before,
                "bit_accuracy_watermarked": bit_acc_after,
                "embedded_bits_preview": "".join(str(b) for b in msg_bits[:32]),
                "decoded_before_preview": (
                    "".join(str(int(b)) for b in decoded_before.tolist()[:32])
                    if bit_acc_before is not None
                    else None
                ),
                "decoded_after_preview": "".join(
                    str(int(b)) for b in decoded_after.tolist()[:32]
                ),
            }
        )

    summary: Dict[str, Any] = {
        "num_images": len(results_per_image),
        "average_bit_accuracy_original": (
            sum(acc_original_list) / len(acc_original_list) if acc_original_list else None
        ),
        "average_bit_accuracy_watermarked": (
            sum(acc_watermarked_list) / len(acc_watermarked_list)
            if acc_watermarked_list
            else None
        ),
    }

    output = {
        "metadata_path": os.path.abspath(args.metadata_path),
        "original_dir": os.path.abspath(args.original_dir) if args.original_dir else None,
        "watermarked_dir": os.path.abspath(args.watermarked_dir),
        "summary": summary,
        "images": results_per_image,
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote detection results for {summary['num_images']} images to: {args.output_json}")
    if summary["average_bit_accuracy_watermarked"] is not None:
        print(f"Average bit accuracy on watermarked images: {summary['average_bit_accuracy_watermarked']:.4f}")
    if summary["average_bit_accuracy_original"] is not None:
        print(f"Average bit accuracy on original images: {summary['average_bit_accuracy_original']:.4f}")


if __name__ == "__main__":
    main()
