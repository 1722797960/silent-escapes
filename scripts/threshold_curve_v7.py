#!/usr/bin/env python3
"""Threshold sensitivity curve (Experiment 2 of Road B).

Re-bins per-image bit accuracies from existing detection JSONs across a dense
threshold grid, adds bootstrap 95% CIs, and plots TPR/FPR curves for both
watermarking schemes. Pure post-processing — no GPU needed.

Inputs (results/, produced by earlier 500-image runs):
  TPR side (watermarked + valid misleading C2PA manifest):
    wm_signed_human.json        WAM 500 images
    ps_attack_*.json?           -> PixelSeal watermarked uses ps_* watermarked acc
  FPR side (untouched originals):
    wm_not_signed.json          WAM 500 originals
    ps_not_signed.json          PixelSeal 500 originals

Outputs:
  results/threshold_curve.json  dense TPR/FPR per scheme per threshold (+CI)
  results/threshold_curve.png   figure for the paper
"""
import json
import glob
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
LATEX_FIGS = os.path.join(os.path.dirname(ROOT), "figs", "v7")
OUT_JSON = os.path.join(RES, "threshold_curve_v7.json")
OUT_PDF = os.path.join(LATEX_FIGS, "threshold_curve.pdf")

THRESHOLDS = np.round(np.arange(0.50, 0.9501, 0.01), 2)
N_BOOT = 2000
CI = 95
SEED = 42


def load_accs(path, key="bit_accuracy_watermarked"):
    d = json.load(open(path))
    return np.array([img[key] for img in d["images"]], dtype=float)


def bootstrap_rate(bits, thr, n_boot=N_BOOT, ci=CI, seed=SEED):
    """Detection rate at `thr` with bootstrap CI."""
    bits = np.asarray(bits)
    rng = np.random.default_rng(seed)
    n = len(bits)
    rates = (bits[rng.integers(0, n, size=(n_boot, n))] >= thr).mean(axis=1)
    lo, hi = np.percentile(rates, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return rates.mean(), lo, hi


def main():
    # ---- TPR side: watermarked images under the misleading-manifest pipeline
    # WAM: signed_human 500-image run
    wam_tpr = load_accs(os.path.join(RES, "wm_signed_human.json"))
    # PixelSeal: ps_* jsons are per-attack; baseline watermarked = ps_not_signed's
    # watermarked column (bit_accuracy_watermarked of the watermarked pre-sign set)
    ps_tpr = load_accs(os.path.join(RES, "ps_not_signed.json"))
    # WAM FPR side: untouched originals
    wam_fpr = load_accs(os.path.join(RES, "wm_not_signed.json"),
                        key="bit_accuracy_original")
    ps_fpr = load_accs(os.path.join(RES, "ps_not_signed.json"),
                       key="bit_accuracy_original")

    schemes = {
        "WAM": {"tpr": wam_tpr, "fpr": wam_fpr},
        "PixelSeal": {"tpr": ps_tpr, "fpr": ps_fpr},
    }

    out = {"thresholds": THRESHOLDS.tolist(), "n_tpr": int(len(wam_tpr)),
           "n_fpr": int(len(wam_fpr)), "n_boot": N_BOOT, "schemes": {}}
    for name, s in schemes.items():
        tpr_m, tpr_lo, tpr_hi = [], [], []
        fpr_m, fpr_lo, fpr_hi = [], [], []
        for t in THRESHOLDS:
            m, lo, hi = bootstrap_rate(s["tpr"], t)
            tpr_m.append(m); tpr_lo.append(lo); tpr_hi.append(hi)
            m, lo, hi = bootstrap_rate(s["fpr"], t)
            fpr_m.append(m); fpr_lo.append(lo); fpr_hi.append(hi)
        out["schemes"][name] = {
            "tpr": tpr_m, "tpr_ci95_lo": tpr_lo, "tpr_ci95_hi": tpr_hi,
            "fpr": fpr_m, "fpr_ci95_lo": fpr_lo, "fpr_ci95_hi": fpr_hi,
        }
        print(f"{name}: TPR n={len(s['tpr'])} mean@0.75={tpr_m[25]:.4f}; "
              f"FPR n={len(s['fpr'])} FPR@0.55={fpr_m[5]:.4f} FPR@0.70={fpr_m[20]:.4f}")

    json.dump(out, open(OUT_JSON, "w"), indent=1)
    print("wrote", OUT_JSON)

    # ---- plot: FPR is the main panel; the flat TPR evidence is retained as an inset.
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = {"WAM": "#1f77b4", "PixelSeal": "#ff7f0e"}
    styles = {
        "WAM": {"ls": "-", "marker": "o"},
        "PixelSeal": {"ls": "--", "marker": "s"},
    }
    x = np.array(out["thresholds"])
    for name, s in out["schemes"].items():
        y = np.array(s["fpr"]) * 100
        lo = np.array(s["fpr_ci95_lo"]) * 100
        hi = np.array(s["fpr_ci95_hi"]) * 100
        ax.plot(x, y, color=colors[name], lw=2, label=name,
                ls=styles[name]["ls"], marker=styles[name]["marker"],
                markevery=5, ms=4)
        ax.fill_between(x, lo, hi, color=colors[name], alpha=0.16, lw=0)

    ax.axvline(0.55, color="#777777", ls="-.", lw=1.2)
    ax.axvline(0.75, color="#222222", ls="--", lw=1.8)
    ax.text(0.552, 0.96, "$t=0.55$", transform=ax.get_xaxis_transform(),
            color="#666666", fontsize=8, va="top")
    ax.text(0.752, 0.96, "$t=0.75$", transform=ax.get_xaxis_transform(),
            color="#222222", fontsize=8, va="top")
    ax.set_xlim(0.50, 0.80)
    ax.set_ylim(-1, 60)
    ax.set_xlabel("Bit-accuracy threshold $t$")
    ax.set_ylabel("False positive rate (%)")
    ax.set_title("Detection-threshold sensitivity on the common 500-image set")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)

    inset = ax.inset_axes([0.53, 0.50, 0.43, 0.40])
    for name, s in out["schemes"].items():
        y = np.array(s["tpr"]) * 100
        inset.plot(x, y, color=colors[name], lw=1.5,
                   ls=styles[name]["ls"], marker=styles[name]["marker"],
                   markevery=5, ms=2.5)
    inset.axvline(0.55, color="#777777", ls="-.", lw=0.8)
    inset.axvline(0.75, color="#222222", ls="--", lw=1.0)
    inset.set_xlim(0.50, 0.80)
    inset.set_ylim(98.5, 100.4)
    inset.set_title("TPR (watermarked)", fontsize=8)
    inset.set_ylabel("TPR (%)", fontsize=7)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.2)
    fig.tight_layout()
    os.makedirs(LATEX_FIGS, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT_PDF)


if __name__ == "__main__":
    main()
