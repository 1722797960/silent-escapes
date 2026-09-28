"""SAM-based subject segmentation for regional v2 experiment (Road B, step 2).

Runs SAM ViT-B automatic mask generation on the first N images of
data/originals_500 and keeps the mask covering the largest area among
those whose predicted-IoU / stability pass SAM's default quality filter —
i.e. "the most salient big subject", matching how a regional watermark
would be applied to the semantic content of an image.

Outputs one binary PNG mask (255 = subject, 0 = background) per image in
data/masks_200/, plus a JSON log with mask area fractions.

Usage (wam env):
  python experiments/sam_segment.py \
    --input-dir data/originals_500 --output-dir data/masks_200 \
    --num-images 200 --ckpt ckpts/sam/sam_vit_b_01ec64.pth
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-images", type=int, default=200)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--points-per-side", type=int, default=16)
    ap.add_argument("--pred-iou-thresh", type=float, default=0.86)
    ap.add_argument("--stability-thresh", type=float, default=0.90)
    ap.add_argument("--min-area-frac", type=float, default=0.02,
                    help="reject masks smaller than this fraction (speckles)")
    ap.add_argument("--max-area-frac", type=float, default=0.60,
                    help="reject masks larger than this fraction (background)")
    ap.add_argument("--min-subject-frac", type=float, default=0.03,
                    help="if the chosen mask is smaller, fall back to the largest candidate")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    sam = sam_model_registry["vit_b"](checkpoint=args.ckpt)
    sam.to(device).eval()
    gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
        min_mask_region_area=2000,  # drop speckles
    )

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".png"))
    files = [f for f in files][: args.num_images]

    log = []
    for i, fname in enumerate(files):
        img = np.array(Image.open(os.path.join(args.input_dir, fname)).convert("RGB"))
        with torch.no_grad():
            masks = gen.generate(img)
        if not masks:
            # fallback: full-frame mask so the pipeline never stalls on an image
            print(f"[{i:03d}] {fname}: NO MASK, using full frame", flush=True)
            m = np.ones(img.shape[:2], dtype=np.uint8) * 255
            frac = 1.0
            n_cand = 0
        else:
            # subject = best predicted-IoU among masks in a plausible area band
            n_px = img.shape[0] * img.shape[1]
            cand = [m for m in masks
                    if args.min_area_frac <= m["area"] / n_px <= args.max_area_frac]
            if not cand:
                # relax to whatever is closest to the band
                cand = [min(masks, key=lambda m: abs(m["area"] / n_px - 0.25))]
            best = max(cand, key=lambda m: m["predicted_iou"])
            # tiny masks embed too few watermark pixels to be meaningful
            if best["area"] / n_px < args.min_subject_frac:
                best = max(cand, key=lambda m: m["area"])
            m = (best["segmentation"].astype(np.uint8)) * 255
            frac = best["area"] / n_px
            n_cand = len(cand)
        out_name = os.path.splitext(fname)[0] + "_mask.png"
        Image.fromarray(m).save(os.path.join(args.output_dir, out_name))
        log.append({"index": i, "image": fname, "mask": out_name,
                    "area_fraction": round(frac, 4), "n_candidates": n_cand})
        print(f"[{i:03d}] {fname[:50]} frac={frac:.3f} cand={n_cand}", flush=True)

    with open(os.path.join(args.output_dir, "segmentation_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    fr = [e["area_fraction"] for e in log]
    print(f"Done {len(log)} masks; area fraction mean={np.mean(fr):.3f} "
          f"median={np.median(fr):.3f} min={np.min(fr):.3f} max={np.max(fr):.3f}")


if __name__ == "__main__":
    main()
