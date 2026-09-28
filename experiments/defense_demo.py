"""Granularity-aware audit demo (Road B, step 4 — the defense sketch).

Idea: image-level C2PA assertions cannot see *where* the watermark lives, so a
legal crop that removes the watermarked region silences the only contradicting
signal (SILENT_ESCAPE).  Defense sketch: bind the watermark to the manifest by
adding a `com.example.region_assertion` assertion that records the watermark
region (bounding box of the embedded mask).  The v2 auditor, after any
geometric transform, maps the asserted region through the same transform and
checks whether *any* watermarked area survives; if not, the asset can no
longer be attributed — the audit returns TAMPERED_REGION instead of a silent
pass.

This demo replays the regional_v2 detection rows (no GPU needed):
  - baseline audit (image-level manifest): crop>=0.4 -> SILENT_ESCAPE
  - region-aware audit:                     crop>=0.4 -> TAMPERED_REGION (flagged)

Usage:
  python experiments/defense_demo.py \
    --detection results/regional_v2/detection.json \
    --meta results/regional_v2/embed_meta.json \
    --output results/defense_demo/defense_demo.json
"""
import argparse
import json
import os

import numpy as np

THRESHOLD = 0.75


def survives_crop(mask_cov, crop_frac, margin=0.02):
    """Fraction of the watermarked region expected to survive a crop of
    `crop_frac` from each side, assuming the region is uniformly distributed
    over the frame (empirical approximation; the real audit computes it from
    the actual mask bbox)."""
    if crop_frac <= 0:
        return 1.0
    kept = (1 - 2 * crop_frac) ** 2  # kept area fraction of the frame
    # a region uniformly spread loses the same fraction; concentrated regions
    # lose more — we flag when expected surviving watermark pixels fall below
    # the decoder's practical floor (≈3% of frame, cf. WAM localized docs)
    return mask_cov * kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detection", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-surviving-frac", type=float, default=0.005,
                    help="audit flags TAMPERED_REGION when expected surviving "
                         "watermark area falls below this frame fraction")
    args = ap.parse_args()

    det = json.load(open(args.detection))
    meta = {m["input_image"]: m for m in json.load(open(args.meta))["images"]}
    rows = det["rows"]

    out_rows, flips = [], 0
    for r in rows:
        cov = meta[r["image"]]["mask_coverage"]
        surv = survives_crop(cov, r["crop_frac"])
        # v1 audit (image-level manifest): signature valid + no watermark signal
        # -> SILENT_ESCAPE, i.e. the asset passes as "human-edited".
        v1 = r["audit_state"]
        # v2 audit (region-aware): if the manifest asserts a watermark region
        # but that region is (almost) gone from the frame, the claim chain is
        # broken — flag it instead of passing silently.
        if r["detected"]:
            v2 = "INTEGRITY_CLASH"
        elif surv < args.min_surviving_frac:
            v2 = "TAMPERED_REGION"  # asserted region no longer present
            flips += 1
        else:
            v2 = "INTEGRITY_CLASH"  # region present but undetectable -> robustness gap
        out_rows.append({**r, "mask_coverage": round(cov, 4),
                         "expected_surviving_frac": round(surv, 5),
                         "audit_v1": v1, "audit_v2": v2})

    by_crop = {}
    for r in out_rows:
        c = by_crop.setdefault(r["crop_frac"], {"n": 0, "v1_escape": 0,
                                                "v2_tampered": 0, "v2_clash": 0})
        c["n"] += 1
        c["v1_escape"] += r["audit_v1"] == "SILENT_ESCAPE"
        c["v2_tampered"] += r["audit_v2"] == "TAMPERED_REGION"
        c["v2_clash"] += r["audit_v2"] == "INTEGRITY_CLASH"

    print(f"{'crop':>6} {'n':>5} {'v1 SILENT(undetected)':>22} {'v2 TAMPERED(flagged)':>21} {'v2 CLASH':>9}")
    for c in sorted(by_crop):
        b = by_crop[c]
        print(f"{c:>6.2f} {b['n']:>5} {b['v1_escape']:>22} {b['v2_tampered']:>21} {b['v2_clash']:>9}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    json.dump({"threshold": THRESHOLD, "min_surviving_frac": args.min_surviving_frac,
               "flips_silent_to_tampered": flips, "by_crop": by_crop,
               "rows": out_rows}, open(args.output, "w"), indent=1)
    print("wrote", args.output, f"(flipped {flips} escapes to TAMPERED_REGION)")


if __name__ == "__main__":
    main()
