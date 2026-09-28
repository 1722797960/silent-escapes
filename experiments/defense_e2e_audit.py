"""End-to-end defense audit (stage C).

For each re-signed cropped asset (produced by defense_e2e_run.py stage B):
  1. read the file with c2pa.Reader via the helper: manifest validity +
     com.example.region_assertion (bbox_csv + 64x64 bitmap + payload hash)
  2. verify the payload hash matches the audited message (binding check)
  3. map the asserted region through the crop transform (the auditor knows the
     crop geometry from the assertion's original-coordinates frame)
  4. run WAM detection: bit accuracy + per-pixel mask logits
  5. verdicts:
       v1 deployed audit   = manifest valid? + bit acc >= t  (no region info)
       v2 region-aware     = INTEGRITY_CLASH | TAMPERED_REGION |
                             WATERMARK_SUPPRESSED | SILENT_OTHER
"""
import argparse
import base64
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
READ_HELPER = os.path.join(E2E_ROOT, "experiments", "read_region_assertion.py")

THRESHOLD = 0.75
REGION_MIN_RETENTION = 0.05
LOGIT_MEAN_MIN = 0.5
CROP_FRACS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]


def map_bitmap_through_crop(bitmap_grid, frac):
    """Asserted bitmap in ORIGINAL image coords; centered crop keeps
    [frac, 1-frac]^2. Returns the retained bitmap window."""
    g = np.array(bitmap_grid, dtype=np.float32)
    if frac <= 0:
        return g
    lo, hi = int(round(frac * g.shape[0])), int(round((1 - frac) * g.shape[0]))
    kept = g[lo:hi, lo:hi]
    return kept if kept.size else np.zeros((1, 1), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e-dir", default=os.path.join(E2E_ROOT, "results/defense_e2e"))
    ap.add_argument("--output-json", default=os.path.join(E2E_ROOT, "results/defense_e2e/audit.json"))
    ap.add_argument("--ckpt", default=os.path.join(E2E_ROOT, "ckpts/wam_mit.pth"))
    ap.add_argument("--params-json", default=os.path.join(E2E_ROOT, "ckpts/params.json"))
    ap.add_argument("--crop-fracs", default=",".join(str(f) for f in CROP_FRACS))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    sys.path.insert(0, WAM_REPO)
    prev = os.getcwd()
    # absolutize user-supplied paths BEFORE os.chdir (they are relative to the
    # caller's CWD, not to the WAM repo)
    abs_params = os.path.abspath(args.params_json)
    abs_ckpt = os.path.abspath(args.ckpt)
    os.chdir(WAM_REPO)
    try:
        from notebooks.inference_utils import load_model_from_checkpoint
        model = load_model_from_checkpoint(abs_params, abs_ckpt)
    finally:
        os.chdir(prev)
    model.to(device).eval()
    from watermark_anything.data.metrics import msg_predict_inference
    from watermark_anything.data.transforms import default_transform

    meta = json.load(open(os.path.join(args.e2e_dir, "embed_meta.json")))["images"]
    fracs = [float(x) for x in args.crop_fracs.split(",")]
    rows = []
    import hashlib

    for i, entry in enumerate(meta):
        fname = entry["input_image"]
        bits = np.unpackbits(np.frombuffer(base64.b64decode(entry["bitmap64"]),
                                           dtype=np.uint8))
        grid = bits.reshape(64, 64).astype(np.float32)
        msg_str = "".join(str(b) for b in entry["message_bits"])
        msg = torch.tensor(entry["message_bits"], dtype=torch.float32)

        for frac in fracs:
            cpath = os.path.join(args.e2e_dir,
                                 f"attack_resign/crop{int(frac*100):02d}",
                                 entry["signed_image"])
            ra = json.loads(subprocess.run([C2PA_PY, READ_HELPER, cpath],
                                           capture_output=True, text=True,
                                           check=True).stdout)
            manifest_valid = ra["manifest_valid"]
            ra_present = ra["region_assertion_present"]
            region = ra.get("region") or {}
            hash_ok = (region.get("payload_hash") ==
                       hashlib.sha256(msg_str.encode()).hexdigest())

            kept_grid = map_bitmap_through_crop(grid, frac)
            region_occupancy = float(kept_grid.mean())
            region_retention = float(
                kept_grid.sum() / max(grid.sum(), 1e-6))

            img = Image.open(cpath).convert("RGB")
            x = default_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model.detect(x)["preds"]
            mask_logits = torch.sigmoid(preds[:, 0])
            pred_msg = msg_predict_inference(preds[:, 1:], mask_logits).cpu().float()[0]
            acc = (pred_msg == msg).float().mean().item()

            ml = F.interpolate(mask_logits.unsqueeze(1), size=kept_grid.shape,
                               mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            logit_in_region = float((ml * kept_grid).sum() / max(kept_grid.sum(), 1e-6))

            if acc >= THRESHOLD:
                v2 = "INTEGRITY_CLASH"
            elif not (ra_present and hash_ok):
                v2 = "ASSERTION_MISSING"  # binding broken — also a red flag
            elif region_retention < REGION_MIN_RETENTION:
                v2 = "TAMPERED_REGION"
            elif logit_in_region < LOGIT_MEAN_MIN:
                v2 = "WATERMARK_SUPPRESSED"
            else:
                v2 = "SILENT_OTHER"
            v1 = "INTEGRITY_CLASH" if acc >= THRESHOLD else "SILENT_ESCAPE"

            rows.append({
                "image": fname, "crop_frac": frac,
                "manifest_valid": manifest_valid,
                "region_assertion_present": ra_present,
                "payload_hash_ok": hash_ok,
                "bit_accuracy": round(acc, 4),
                # Keep the legacy field name for downstream compatibility, but
                # give it the paper's intended retained/original semantics.
                "region_in_frame": round(region_retention, 5),
                "region_retention": round(region_retention, 5),
                "region_occupancy": round(region_occupancy, 5),
                "logit_in_region": round(logit_in_region, 4),
                "audit_v1_deployed": v1,
                "audit_v2_region_aware": v2,
            })
        print(f"[{i:03d}] {fname[:44]} done ({len(fracs)} crops)", flush=True)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    json.dump({"threshold": THRESHOLD,
               "region_metric": "retained_mask_pixels/original_mask_pixels",
               "region_min_retention": REGION_MIN_RETENTION,
               "logit_mean_min": LOGIT_MEAN_MIN, "rows": rows},
              open(args.output_json, "w"), indent=1)

    print(f"\n{'crop':>6} {'n':>4} {'v1 SILENT':>10} {'v2 TAMPERED':>12} "
          f"{'v2 SUPPRESSED':>14} {'v2 CLASH':>9} {'ra_ok':>6} {'hash_ok':>8}")
    by_crop = {}
    for r in rows:
        c = by_crop.setdefault(r["crop_frac"],
                               {"n": 0, "v1": 0, "tam": 0, "sup": 0,
                                "clash": 0, "ra": 0, "hk": 0})
        c["n"] += 1
        c["v1"] += r["audit_v1_deployed"] == "SILENT_ESCAPE"
        c["tam"] += r["audit_v2_region_aware"] == "TAMPERED_REGION"
        c["sup"] += r["audit_v2_region_aware"] == "WATERMARK_SUPPRESSED"
        c["clash"] += r["audit_v2_region_aware"] == "INTEGRITY_CLASH"
        c["ra"] += r["region_assertion_present"]
        c["hk"] += r["payload_hash_ok"]
    for c in sorted(by_crop):
        b = by_crop[c]
        print(f"{c:>6.2f} {b['n']:>4} {b['v1']:>10} {b['tam']:>12} "
              f"{b['sup']:>14} {b['clash']:>9} {b['ra']:>6} {b['hk']:>8}")
    print("wrote", args.output_json)


if __name__ == "__main__":
    main()
