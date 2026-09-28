"""Regional WAM embedding demo (Paper-B experiment).

Embeds a 32-bit WAM message into ONLY a central square region of each image
(area fraction = --region-ratio). Outside the region the pixels stay
untouched. Demonstrates the granularity mismatch between region-level
watermarks and image-level C2PA manifests.
"""
import argparse
import json
import os
import sys

import torch
import torchvision.transforms as T
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAM_REPO = os.path.join(_REPO_ROOT, "third_party", "watermark-anything")
if WAM_REPO not in sys.path:
    sys.path.insert(0, WAM_REPO)
from notebooks.inference_utils import load_model_from_checkpoint  # noqa: E402


def center_square_mask(h: int, w: int, ratio: float, device) -> torch.Tensor:
    side = int(round((h * w * ratio) ** 0.5))
    side = max(8, min(side, h, w))
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    mask = torch.zeros((1, 1, h, w), device=device)
    mask[:, :, y0:y0 + side, x0:x0 + side] = 1.0
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--metadata-path", default=None)
    ap.add_argument("--num-images", type=int, default=20)
    ap.add_argument("--region-ratio", type=float, default=0.25)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--params-json", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    abs_params = os.path.abspath(args.params_json)
    abs_ckpt = os.path.abspath(args.ckpt)
    prev_cwd = os.getcwd()
    os.chdir(WAM_REPO)
    try:
        model = load_model_from_checkpoint(abs_params, abs_ckpt)
    finally:
        os.chdir(prev_cwd)
    model.to(device).eval()

    os.makedirs(args.output_dir, exist_ok=True)
    to_tensor = T.ToTensor()
    to_pil = T.ToPILImage()

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".png"))[: args.num_images]
    torch.manual_seed(args.seed)

    metadata = []
    for i, fname in enumerate(files):
        img = to_tensor(Image.open(os.path.join(args.input_dir, fname)).convert("RGB")).unsqueeze(0).to(device)
        _, _, h, w = img.shape
        mask = center_square_mask(h, w, args.region_ratio, device)
        with torch.no_grad():
            msgs = model.get_random_msg(1)
            out = model.embed(img, msgs)
            img_wm = out["imgs_w"] * mask + img * (1 - mask)
        name_root = os.path.splitext(fname)[0]
        wm_name = f"{name_root}_regionwatermarked.png"
        to_pil(img_wm.clamp(0, 1).squeeze(0).cpu()).save(os.path.join(args.output_dir, wm_name))
        metadata.append({
            "index": i,
            "input_image": fname,
            "watermarked_image": wm_name,
            "message_bits": msgs.squeeze(0).to(torch.int32).tolist(),
            "watermark_method": "wam_mit_regional",
            "region_ratio": args.region_ratio,
        })
        print(f"[{i:04d}] {fname} -> {wm_name} (region {args.region_ratio})", flush=True)

    meta_path = args.metadata_path or os.path.join(args.output_dir, "watermark_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"images": metadata}, f, indent=2)
    print(f"Wrote {len(metadata)} entries to {meta_path}")


if __name__ == "__main__":
    main()
