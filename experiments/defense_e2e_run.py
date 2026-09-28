"""End-to-end defense pipeline (driver): embed+sign -> crop -> re-sign -> audit.

Orchestrates the three stages as subprocess calls so each stage runs in its
correct environment:
  stage A  defense_e2e_embed.py      (wam env)     WAM regional embed + region
                                                   assertion signed at birth
  stage B  attack: crop each asset at 9 severities (PIL) + compliant re-sign
           with the region assertion carried into the new manifest
           (integrity-clash venv, resign_with_region_assertion.py)
  stage C  defense_e2e_audit.py      (wam env)     read assertion from the
           re-signed cropped file, map region through the crop, run WAM
           detection, emit region-aware verdicts

Usage (tmux, wam env):
  python experiments/defense_e2e_run.py --stage all --num-images 200
"""
import argparse
import base64
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

E2E_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAM_PY = sys.executable  # current interpreter (wam env)
C2PA_PY = os.environ.get(
    "C2PA_PY", os.path.join(E2E_ROOT, "venvs", "c2pa", "bin", "python"))
CERT = os.path.join(E2E_ROOT, "certs/ec_chain.pem")
KEY = os.path.join(E2E_ROOT, "certs/ec_key.pem")
CROP_FRACS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]


def run_stage_a(args):
    out = os.path.join(E2E_ROOT, "results/defense_e2e")
    subprocess.run([WAM_PY, os.path.join(E2E_ROOT, "experiments/defense_e2e_embed.py"),
                    "--output-dir", out, "--num-images", str(args.num_images)],
                   check=True)


def stage_b(args):
    """crop + compliant re-sign carrying the region assertion."""
    e2e = args.e2e_dir
    meta = json.load(open(os.path.join(e2e, "embed_meta.json")))["images"]
    signed_dir = os.path.join(e2e, "watermarked_signed_ra")
    for frac in CROP_FRACS:
        out_dir = os.path.join(e2e, f"attack_resign/crop{int(frac*100):02d}")
        os.makedirs(out_dir, exist_ok=True)
        for i, entry in enumerate(meta):
            src = os.path.join(signed_dir, entry["signed_image"])
            img = Image.open(src).convert("RGB")
            w, h = img.size
            if frac > 0:
                m = int(round(w * frac))
                img = img.crop((m, m, w - m, h - m)).resize((w, h), Image.LANCZOS)
            tmp = os.path.join(out_dir, "_tmp.png")
            img.save(tmp, format="PNG")
            region = {"bbox_csv": entry["bbox_csv"], "bitmap64": entry["bitmap64"],
                      "payload_hash": entry["payload_hash"], "alg": "wam_mit_regional"}
            dst = os.path.join(out_dir, entry["signed_image"])
            subprocess.run([C2PA_PY, os.path.join(E2E_ROOT, "experiments/resign_with_region_assertion.py"),
                            "--input", tmp, "--output", dst,
                            "--region-json", "-",
                            "--cert", CERT, "--key", KEY],
                           input=json.dumps(region), text=True, check=True,
                           capture_output=True)
            os.remove(tmp)
            if (i + 1) % 50 == 0:
                print(f"  crop={frac}: {i+1}/{len(meta)}", flush=True)
        print(f"crop={frac} re-signed {len(meta)} assets", flush=True)


def run_stage_c(args):
    subprocess.run([WAM_PY, os.path.join(E2E_ROOT, "experiments/defense_e2e_audit.py"),
                    "--e2e-dir", args.e2e_dir,
                    "--output-json", os.path.join(args.e2e_dir, "audit.json"),
                    "--ckpt", os.path.join(E2E_ROOT, "ckpts/wam_mit.pth"),
                    "--params-json", os.path.join(E2E_ROOT, "ckpts/params.json"),
                    "--crop-fracs", args.crop_fracs], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["a", "b", "c", "all"])
    ap.add_argument("--e2e-dir", default=os.path.join(E2E_ROOT, "results/defense_e2e"))
    ap.add_argument("--num-images", type=int, default=200)
    ap.add_argument("--crop-fracs", default=",".join(str(f) for f in CROP_FRACS))
    args = ap.parse_args()

    if args.stage in ("a", "all"):
        print("=== Stage A: embed + sign with region assertion ===", flush=True)
        run_stage_a(args)
    if args.stage in ("b", "all"):
        print("=== Stage B: crop + compliant re-sign ===", flush=True)
        stage_b(args)
    if args.stage in ("c", "all"):
        print("=== Stage C: region-aware audit ===", flush=True)
        run_stage_c(args)


if __name__ == "__main__":
    main()
