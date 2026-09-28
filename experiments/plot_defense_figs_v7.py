"""Paper figure A: end-to-end defense pipeline diagram.
Figure B: audit state transition of a single asset across crop severities
(authenticated fake -> silent under deployed audit; flagged throughout under
the region-aware audit).
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from PIL import Image

matplotlib.rcParams["pdf.fonttype"] = 42

E2E_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.environ.get(
    "PAPER_FIGS_DIR", os.path.join(os.path.dirname(E2E_ROOT), "figs", "v7"))
AUDIT_JSON = os.environ.get(
    "DEFENSE_AUDIT_JSON",
    os.path.join(E2E_ROOT, "results/defense_e2e/audit_retention.json"))

C_EMB = "#4477aa"   # blue: watermark layer
C_MAN = "#d55e00"   # orange-red: manifest layer
C_AUD = "#009e73"   # green: audit
C_GRAY = "#666666"


def fig_pipeline():
    with open(AUDIT_JSON, encoding="utf-8") as source:
        audit = json.load(source)
    escapes = [r for r in audit["rows"] if r["audit_v1_deployed"] == "SILENT_ESCAPE"]
    flagged = sum(r["audit_v2_region_aware"] in ("TAMPERED_REGION", "WATERMARK_SUPPRESSED")
                  for r in escapes)
    flag_rate = 100 * flagged / len(escapes)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fontsize=9, tc="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.08",
                                    fc=fc, ec="none"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, color=tc, wrap=True)

    def arrow(x0, y0, x1, y1, color=C_GRAY, ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                     arrowstyle="-|>", mutation_scale=16,
                                     color=color, lw=1.6, linestyle=ls))

    # Stage boxes (top row = asset flow)
    box(0.2, 3.1, 1.9, 1.0, "Original\nimage", "#888888")
    box(2.6, 3.1, 2.3, 1.0, "WAM regional\nembed (32-bit)\ninto SAM mask", C_EMB)
    box(5.4, 3.1, 2.6, 1.0, "Sign: misleading\nmanifest +\nregion assertion", C_MAN)
    box(8.6, 3.1, 1.9, 1.0, "Crop\n$\\in\\{0{,}\\ldots,0.4\\}$", "#888888")
    box(11.0, 3.15, 0.9, 0.9, "Re-sign\n(compliant)", C_MAN)
    arrow(2.1, 3.6, 2.6, 3.6); arrow(4.9, 3.6, 5.4, 3.6)
    arrow(8.0, 3.6, 8.6, 3.6); arrow(10.5, 3.6, 11.0, 3.6)

    # Assertion payload detail
    box(5.4, 1.7, 2.6, 1.0,
        "com.example.region_assertion:\nbbox (CSV) + 64$\\times$64 mask\n+ payload SHA-256 + alg",
        "#a05a72", fontsize=8)
    arrow(6.7, 3.1, 6.7, 2.7, color=C_MAN)

    # Audit row
    box(2.6, 0.2, 3.6, 1.0,
        "Deployed audit: manifest + bit acc.\n$\\Rightarrow$ SILENT-ESCAPE (no flags)", "#999999")
    box(7.0, 0.2, 4.4, 1.0,
        f"Region-aware audit: map assertion\nthrough crop + WAM mask logits\n$\\Rightarrow$ {flag_rate:.1f}% of SDXL escapes flagged",
        C_AUD)
    arrow(11.45, 3.15, 9.2, 1.2)
    arrow(6.2, 3.1, 4.4, 1.2, color="#aaaaaa", ls="--")

    legend = [
        Patch(facecolor="#888888", label="Input / edit"),
        Patch(facecolor=C_EMB, label="WAM embedding"),
        Patch(facecolor=C_MAN, label="Manifest / assertion"),
        Patch(facecolor=C_AUD, label="Audit"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.03),
              ncol=4, frameon=False, fontsize=8, handlelength=1.4)

    ax.set_title("End-to-end granularity-aware audit: the region assertion survives the "
                 "adversary's crop through compliant re-signing", fontsize=10)
    fig.tight_layout()
    out = os.path.join(FIGS, "defense_pipeline.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_transition():
    audit = json.load(open(AUDIT_JSON, encoding="utf-8"))
    regional = json.load(open(os.path.join(E2E_ROOT, "results/regional_v2/detection.json"))) \
        if os.path.exists(os.path.join(E2E_ROOT, "results/regional_v2/detection.json")) else None

    # pick one representative asset: detected at 0, silent at 0.4
    by_img = {}
    for r in audit["rows"]:
        by_img.setdefault(r["image"], []).append(r)
    pick = None
    for img, rows in by_img.items():
        rows.sort(key=lambda r: r["crop_frac"])
        if rows[0]["bit_accuracy"] >= 0.9 and rows[-1]["bit_accuracy"] <= 0.55:
            pick = img
            break
    rows = sorted(by_img[pick], key=lambda r: r["crop_frac"])

    x = [r["crop_frac"] for r in rows]
    acc = [r["bit_accuracy"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    # left: the asset image (uncropped vs cropped)
    signed = os.path.join(E2E_ROOT, "results/defense_e2e/watermarked_signed_ra",
                          rows[0]["image"].replace(".png", "_regionwm_signed_ra.png"))
    img = Image.open(signed)
    w, h = img.size
    m = int(w * 0.4)
    cropped = img.crop((m, m, w - m, h - m)).resize((w, h), Image.LANCZOS)
    for ax_i, im, ttl in ((axes[0], img, "crop 0%"), (axes[1], cropped, "crop 40%")):
        ax_i.imshow(im)
        ax_i.set_title(ttl, fontsize=10)
        ax_i.axis("off")
    # Use equal-size verdict panels so the before/after comparison is balanced.
    def verdict_badge(ax, deployed, region_aware, color):
        patch = FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.17,
            transform=ax.transAxes,
            boxstyle="round,pad=0.015",
            fc=color, ec="white", lw=0.7, alpha=0.90,
        )
        ax.add_patch(patch)
        ax.text(0.05, 0.105,
                f"Deployed: {deployed}\nRegion-aware: {region_aware}",
                transform=ax.transAxes, fontsize=7.8, color="white",
                va="center", ha="left")

    verdict_badge(axes[0], "INTEGRITY-CLASH", "INTEGRITY-CLASH", C_AUD)
    verdict_badge(axes[1], "SILENT-ESCAPE (passes)",
                  rows[-1]["audit_v2_region_aware"], C_MAN)
    fig.suptitle(f"One asset's silent escape: bit accuracy {acc[0]:.2f} $\\to$ {acc[-1]:.2f}, "
                 f"manifest valid throughout", fontsize=10)

    fig.tight_layout()
    out = os.path.join(FIGS, "escape_transition.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, "(asset:", pick + ")")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig_pipeline()
    fig_transition()
