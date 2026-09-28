"""End-to-end defense: regional embedding + signed region assertion (step 1/3).

Difference from the sketch (replay) version: the region assertion is a *real*
C2PA assertion, signed into the manifest, carried inside the image file, and
later read back with c2pa.Reader from the *cropped* file.

Per image (N=200):
  1. embed a 32-bit WAM message into the SAM subject mask
  2. compute the assertion payload:
       - bbox           minimal bounding box of the mask, normalized [x0,y0,x1,y1]
       - bitmap64       64x64 binary downsample of the mask, base64 (raw bytes)
       - payload_hash   SHA-256 of the embedded 32-bit message
       - alg            embedding algorithm tag
  3. sign with the misleading human-edited manifest + com.example.region_assertion

Outputs experiments output dir:
  watermarked_signed/    *_regionwm_signed_ra.png  (assertion inside)
  embed_meta.json        per-image message bits, bbox, bitmap64, coverage

Run with the wam conda env (WAM + torch), but sign via the integrity-clash
venv's c2pa — this script shells out to that venv for signing, or run
sign-only step separately. To keep deps simple we do WAM embedding here and
sign in the same process by importing c2pa if available.
"""
import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

WAM_REPO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party", "watermark-anything")
E2E_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C2PA_PY = os.environ.get(
    "C2PA_PY", os.path.join(E2E_ROOT, "venvs", "c2pa", "bin", "python"))
SIGN_HELPER = os.path.join(E2E_ROOT, "experiments", "sign_with_region_assertion.py")


def mask_to_bbox_bitmap(mask_hw, grid=64):
    """mask_hw: bool array [H,W] -> normalized bbox + base64 bitmap at grid res."""
    h, w = mask_hw.shape
    ys, xs = np.where(mask_hw)
    if len(ys) == 0:
        bbox = [0.0, 0.0, 1.0, 1.0]
    else:
        bbox = [float(xs.min()) / w, float(ys.min()) / h,
                float(xs.max() + 1) / w, float(ys.max() + 1) / h]
    m_img = Image.fromarray((mask_hw.astype(np.uint8)) * 255).resize((grid, grid), Image.BILINEAR)
    bits = (np.array(m_img) > 127).astype(np.uint8)
    # c2pa-python 0.37 quirk: numeric arrays in assertion data are mis-serialized
    # as raw bytes (float lists -> empty). Encode bbox as CSV string.
    bbox_csv = ",".join(f"{v:.6f}" for v in bbox)
    return bbox_csv, base64.b64encode(np.packbits(bits.flatten()).tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default=os.path.join(E2E_ROOT, "data/originals_500"))
    ap.add_argument("--mask-dir", default=os.path.join(E2E_ROOT, "data/masks_200"))
    ap.add_argument("--output-dir", default=os.path.join(E2E_ROOT, "results/defense_e2e"))
    ap.add_argument("--ckpt", default=os.path.join(E2E_ROOT, "ckpts/wam_mit.pth"))
    ap.add_argument("--params-json", default=os.path.join(E2E_ROOT, "ckpts/params.json"))
    ap.add_argument("--manifest-template", default=os.path.join(E2E_ROOT, "manifests/manifest_human_edited.json"))
    ap.add_argument("--cert", default=os.path.join(E2E_ROOT, "certs/ec_chain.pem"))
    ap.add_argument("--key", default=os.path.join(E2E_ROOT, "certs/ec_key.pem"))
    ap.add_argument("--num-images", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sys.path.insert(0, WAM_REPO)
    prev = os.getcwd()
    os.chdir(WAM_REPO)
    try:
        from notebooks.inference_utils import load_model_from_checkpoint
        model = load_model_from_checkpoint(os.path.abspath(args.params_json),
                                           os.path.abspath(args.ckpt))
    finally:
        os.chdir(prev)
    model.to(device).eval()

    wm_dir = os.path.join(args.output_dir, "watermarked_unsigned")
    signed_dir = os.path.join(args.output_dir, "watermarked_signed_ra")
    os.makedirs(wm_dir, exist_ok=True)
    os.makedirs(signed_dir, exist_ok=True)
    to_tensor, to_pil = T.ToTensor(), T.ToPILImage()

    seg_log = {e["image"]: e for e in
               json.load(open(os.path.join(args.mask_dir, "segmentation_log.json")))}

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".png"))
    files = [f for f in files
             if os.path.exists(os.path.join(
                 args.mask_dir, os.path.splitext(f)[0] + "_mask.png"))][: args.num_images]
    torch.manual_seed(args.seed)

    meta = []
    for i, fname in enumerate(files):
        img = to_tensor(Image.open(os.path.join(args.input_dir, fname)).convert("RGB")).unsqueeze(0).to(device)
        _, _, h, w = img.shape
        mask_png = np.array(Image.open(os.path.join(
            args.mask_dir, os.path.splitext(fname)[0] + "_mask.png")).convert("L"))
        m_hw = np.array(Image.fromarray(mask_png).resize((w, h), Image.NEAREST)) > 127
        m_t = torch.from_numpy(m_hw.astype(np.float32))[None, None].to(device)
        coverage = m_t.mean().item()

        with torch.no_grad():
            msg = model.get_random_msg(1)
            out = model.embed(img, msg)
            img_wm = out["imgs_w"] * m_t + img * (1 - m_t)
        name_root = os.path.splitext(fname)[0]
        wm_name = f"{name_root}_regionwm.png"
        wm_path = os.path.join(wm_dir, wm_name)
        to_pil(img_wm.clamp(0, 1).squeeze(0).cpu()).save(wm_path)

        msg_bits = msg.squeeze(0).to(torch.int32).tolist()
        msg_str = "".join(str(b) for b in msg_bits)
        bbox_csv, bitmap64 = mask_to_bbox_bitmap(m_hw)
        bbox = [float(v) for v in bbox_csv.split(",")]
        entry = {
            "index": i, "input_image": fname, "unsigned_image": wm_name,
            "message_bits": msg_bits,
            "payload_hash": hashlib.sha256(msg_str.encode()).hexdigest(),
            "bbox": [round(v, 6) for v in bbox],
            "bbox_csv": bbox_csv,
            "bitmap64": bitmap64,
            "mask_coverage": round(coverage, 4),
            "sam_area_fraction": seg_log.get(fname, {}).get("area_fraction"),
        }
        # sign via helper (c2pa venv): adds region assertion + misleading manifest
        signed_name = f"{name_root}_regionwm_signed_ra.png"
        entry["signed_image"] = signed_name
        meta.append(entry)
        subprocess.run([C2PA_PY, SIGN_HELPER,
                        "--input", wm_path,
                        "--output", os.path.join(signed_dir, signed_name),
                        "--manifest-template", args.manifest_template,
                        "--cert", args.cert, "--key", args.key,
                        "--region-json", "-"],
                       input=json.dumps({"bbox_csv": bbox_csv, "bitmap64": bitmap64,
                                         "payload_hash": entry["payload_hash"],
                                         "alg": "wam_mit_regional"}),
                       text=True, check=True, capture_output=True)
        print(f"[{i:03d}] {fname[:44]} cov={coverage:.3f} signed", flush=True)

    with open(os.path.join(args.output_dir, "embed_meta.json"), "w") as f:
        json.dump({"images": meta}, f, indent=1)
    print(f"Done {len(meta)} -> {args.output_dir}")


if __name__ == "__main__":
    main()
