"""Plot regional v2 survival curves with bootstrap CIs (Road B, step 5).

Reads results/regional_v2/summary.json (+ detection.json) and renders:
  - detection rate vs crop severity (with 95% CI band), SAM-mask N=200
  - per-image bit accuracy violin/strip at each crop severity
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regional-dir", default="results/regional_v2")
    args = ap.parse_args()

    summary = json.load(open(os.path.join(args.regional_dir, "summary.json")))
    rows = json.load(open(os.path.join(args.regional_dir, "detection.json")))["rows"]

    x = np.array([s["crop_frac"] for s in summary])
    y = np.array([s["detection_rate_pct"] for s in summary])
    lo = np.array([s["detection_rate_ci95"][0] for s in summary])
    hi = np.array([s["detection_rate_ci95"][1] for s in summary])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.plot(x, y, "o-", color="#4477aa", lw=2, label="detection rate")
    ax.fill_between(x, lo, hi, color="#4477aa", alpha=0.18, lw=0, label="95% CI (bootstrap)")
    ax.axhline(100, color="grey", lw=0.5)
    ax.axvline(0.40, color="#cc6677", ls="--", lw=1)
    ax.text(0.395, 8, "escape onset", color="#cc6677", ha="right", fontsize=9)
    ax.set_xlabel("Crop fraction (each side)")
    ax.set_ylabel("Watermark detection rate (%)")
    ax.set_ylim(-3, 105)
    ax.set_title(f"Silent Escape vs crop severity — SAM subject mask")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    fracs = sorted({r["crop_frac"] for r in rows})
    data = [[r["bit_accuracy"] for r in rows if r["crop_frac"] == f] for f in fracs]
    parts = ax.violinplot(data, positions=fracs, widths=0.045, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4477aa"); pc.set_alpha(0.5)
    ax.axhline(0.75, color="#cc6677", ls="--", lw=1, label="threshold t = 0.75")
    ax.axhline(0.5, color="grey", ls=":", lw=1, label="chance = 0.5")
    ax.set_xlabel("Crop fraction (each side)")
    ax.set_ylabel("Bit accuracy")
    ax.set_title("Per-image bit accuracy distribution")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(frameon=False, loc="lower left")

    fig.tight_layout()
    out = os.path.join(args.regional_dir, "regional_v2_survival.png")
    fig.savefig(out, dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main()
