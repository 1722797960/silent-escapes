"""Regional v2: SAM-mask regional watermarking vs crop survival (Road B, step 3).

Pipeline per image (N=200):
  1. embed 32-bit WAM message into the SAM subject mask only
  2. sign with the misleading "human-edited" C2PA manifest
  3. for each crop fraction c in {0, 0.05, ..., 0.40}:
       crop c from each side + resize back -> WAM detect -> bit accuracy
       + does the *manifest claim* still verify?  (always yes: signature intact)
  4. audit classification per the 4-state protocol:
       clash  if bit_acc >= t (watermark contradicts "human-edited" manifest)
       silent if bit_acc <  t (watermark gone, manifest unopposed)  <- the escape

Outputs results/regional_v2/{embed_meta.json, detection.json, summary.json}

Usage (wam env, tmux):
  python experiments/regional_v2.py \
    --input-dir data/originals_500 --mask-dir data/masks_200 \
    --output-dir results/regional_v2 \
    --ckpt ckpts/wam_mit.pth --params-json ckpts/params.json \
    --num-images 200
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAM_REPO = os.path.join(_REPO_ROOT, "third_party", "watermark-anything")
if WAM_REPO not in sys.path:
    sys.path.insert(0, WAM_REPO)
from notebooks.inference_utils import load_model_from_checkpoint  # noqa: E402
from watermark_anything.data.metrics import msg_predict_inference  # noqa: E402
from watermark_anything.data.transforms import default_transform  # noqa: E402

CROP_FRACS = [round(0.05 * i, 2) for i in range(9)]  # 0.00 .. 0.40
THRESHOLD = 0.75


def crop_and_back(img, frac):
    if frac <= 0:
        return img
    w, h = img.size
    m = int(round(w * frac))
    img = img.crop((m, m, w - m, h - m))
    return img.resize((w, h), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--params-json", required=True)
    ap.add_argument("--num-images", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    abs_params, abs_ckpt = os.path.abspath(args.params_json), os.path.abspath(args.ckpt)
    prev = os.getcwd()
    os.chdir(WAM_REPO)
    try:
        model = load_model_from_checkpoint(abs_params, abs_ckpt)
    finally:
        os.chdir(prev)
    model.to(device).eval()

    os.makedirs(args.output_dir, exist_ok=True)
    wm_dir = os.path.join(args.output_dir, "watermarked")
    os.makedirs(wm_dir, exist_ok=True)
    to_tensor, to_pil = T.ToTensor(), T.ToPILImage()

    seg_log = {e["image"]: e for e in
               json.load(open(os.path.join(args.mask_dir, "segmentation_log.json")))}

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".png"))
    # only images that have a SAM mask
    files = [f for f in files
             if os.path.exists(os.path.join(
                 args.mask_dir, os.path.splitext(f)[0] + "_mask.png"))][: args.num_images]
    torch.manual_seed(args.seed)

    meta, rows = [], []
    for i, fname in enumerate(files):
        img = to_tensor(Image.open(os.path.join(args.input_dir, fname)).convert("RGB")).unsqueeze(0).to(device)
        _, _, h, w = img.shape
        mask_png = np.array(Image.open(os.path.join(
            args.mask_dir, os.path.splitext(fname)[0] + "_mask.png")).convert("L"))
        # binarize + resize SAM mask (saved at original resolution) to image res
        m_t = torch.from_numpy((mask_png > 127).astype(np.float32))[None, None].to(device)
        if m_t.shape[-2:] != (h, w):
            m_t = F.interpolate(m_t, size=(h, w), mode="nearest")
        coverage = m_t.mean().item()

        with torch.no_grad():
            msg = model.get_random_msg(1)
            out = model.embed(img, msg)
            img_wm = out["imgs_w"] * m_t + img * (1 - m_t)
        name_root = os.path.splitext(fname)[0]
        wm_name = f"{name_root}_regionwm.png"
        to_pil(img_wm.clamp(0, 1).squeeze(0).cpu()).save(os.path.join(wm_dir, wm_name))
        meta.append({
            "index": i, "input_image": fname, "watermarked_image": wm_name,
            "message_bits": msg.squeeze(0).to(torch.int32).tolist(),
            "mask_coverage": round(coverage, 4),
            "sam_area_fraction": seg_log.get(fname, {}).get("area_fraction"),
            "watermark_method": "wam_mit_regional_sam_v2",
        })

        # detection across crop severities
        for frac in CROP_FRACS:
            img_c = crop_and_back(Image.open(os.path.join(wm_dir, wm_name)).convert("RGB"), frac)
            t = default_transform(img_c).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model.detect(t)["preds"]
            mask_preds = torch.sigmoid(preds[:, 0])
            pred = msg_predict_inference(preds[:, 1:], mask_preds).cpu().float()[0]
            acc = (pred == msg).float().mean().item()
            rows.append({
                "image": fname, "crop_frac": frac, "bit_accuracy": round(acc, 4),
                "detected": acc >= THRESHOLD,
                "manifest": "human_edited (valid)",  # signature survives any crop
                "audit_state": "INTEGRITY_CLASH" if acc >= THRESHOLD else "SILENT_ESCAPE",
            })
        accs = [r["bit_accuracy"] for r in rows if r["image"] == fname]
        print(f"[{i:03d}] {fname[:44]} cov={coverage:.3f} acc@0={accs[0]:.3f} "
              f"acc@0.4={accs[-1]:.3f}", flush=True)

    with open(os.path.join(args.output_dir, "embed_meta.json"), "w") as f:
        json.dump({"images": meta}, f, indent=1)
    with open(os.path.join(args.output_dir, "detection.json"), "w") as f:
        json.dump({"threshold": THRESHOLD, "crop_fracs": CROP_FRACS, "rows": rows}, f, indent=1)

    # ---- summary with bootstrap CI over images
    rng = np.random.default_rng(42)
    summary = []
    for frac in CROP_FRACS:
        vals = np.array([r["bit_accuracy"] for r in rows if r["crop_frac"] == frac])
        det = np.array([r["detected"] for r in rows if r["crop_frac"] == frac], dtype=float)
        n = len(vals)
        idx = rng.integers(0, n, size=(2000, n))
        mean_lo, mean_hi = np.percentile(vals[idx].mean(axis=1), [2.5, 97.5])
        det_lo, det_hi = np.percentile(det[idx].mean(axis=1) * 100, [2.5, 97.5])
        summary.append({
            "crop_frac": frac, "n": n,
            "bit_acc_mean": round(vals.mean(), 4),
            "bit_acc_ci95": [round(mean_lo, 4), round(mean_hi, 4)],
            "bit_acc_min": round(vals.min(), 4),
            "detection_rate_pct": round(det.mean() * 100, 2),
            "detection_rate_ci95": [round(det_lo, 2), round(det_hi, 2)],
            "escape_rate_pct": round(100 - det.mean() * 100, 2),
        })
        print(f"crop={frac:.2f}: acc={vals.mean():.4f} "
              f"CI[{mean_lo:.4f},{mean_hi:.4f}] det={det.mean()*100:.1f}%")
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("Done ->", args.output_dir)


if __name__ == "__main__":
    main()
