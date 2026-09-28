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

matplotlib.rcParams["pdf.fonttype"] = 42

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATEX_FIGS = os.path.join(os.path.dirname(ROOT), "figs", "v7")


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

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax = axes[0]
    ax.plot(x, y, "o-", color="#4477aa", lw=2, label="detection rate")
    ax.fill_between(x, lo, hi, color="#4477aa", alpha=0.18, lw=0, label="95% CI (bootstrap)")
    ax.axhline(100, color="grey", lw=0.5)
    for crop, label in ((0.20, "20%"), (0.30, "30%")):
        ax.axvline(crop, color="#777777", ls="--", lw=1, alpha=0.8)
        ax.text(crop, 103, label, color="#666666", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Crop fraction (each side)")
    ax.set_ylabel("Watermark detection rate (%)")
    ax.set_ylim(-3, 108)
    ax.set_title("Silent escape vs. crop severity - SAM subject mask")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1]
    fracs = sorted({r["crop_frac"] for r in rows})
    data = [[r["bit_accuracy"] for r in rows if r["crop_frac"] == f] for f in fracs]
    parts = ax.violinplot(data, positions=fracs, widths=0.045, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4477aa"); pc.set_alpha(0.5)
    ax.axhline(0.75, color="#222222", ls="--", lw=1.8,
               label="operating threshold t = 0.75")
    ax.axhline(0.55, color="#777777", ls="-.", lw=1.2,
               label="sensitivity reference t = 0.55")
    ax.axhline(0.5, color="#aaaaaa", ls=":", lw=1, label="chance = 0.5")
    ax.set_xlabel("Crop fraction (each side)")
    ax.set_ylabel("Bit accuracy")
    ax.set_title("Per-image bit accuracy distribution")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(frameon=False, loc="lower left")

    fig.tight_layout()
    os.makedirs(LATEX_FIGS, exist_ok=True)
    out = os.path.join(LATEX_FIGS, "regional_survival.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
