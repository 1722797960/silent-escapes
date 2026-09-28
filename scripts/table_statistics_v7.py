#!/usr/bin/env python3
"""Derive publication-table statistics from existing per-image results.

This script performs deterministic post-processing only. It does not run any
watermark model, image generator, segmentation model, or C2PA pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "table_statistics_v7.json"
N_BOOT = 2_000
SEED = 42
THRESHOLD = 0.75


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def bootstrap_mean_ci(values: np.ndarray) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    samples = values[rng.integers(0, len(values), size=(N_BOOT, len(values)))]
    return np.percentile(samples.mean(axis=1), [2.5, 97.5]).tolist()


def bootstrap_rate_ci(events: np.ndarray) -> list[float]:
    events = np.asarray(events, dtype=float)
    rng = np.random.default_rng(SEED)
    samples = events[rng.integers(0, len(events), size=(N_BOOT, len(events)))]
    return (100 * np.percentile(samples.mean(axis=1), [2.5, 97.5])).tolist()


def summarize_accuracy(values: np.ndarray, positive: bool = True) -> dict:
    values = np.asarray(values, dtype=float)
    classified = values >= THRESHOLD if positive else values < THRESHOLD
    return {
        "mean": float(values.mean()),
        "mean_ci95": bootstrap_mean_ci(values),
        "min": float(values.min()),
        "classified_n": int(classified.sum()),
        "n": int(len(values)),
        "classified_pct": float(100 * classified.mean()),
    }


def image_values(filename: str, key: str) -> np.ndarray:
    data = load_json(RESULTS / filename)
    return np.asarray([row[key] for row in data["images"]], dtype=float)


def table2() -> dict:
    wam_base = load_json(RESULTS / "wm_not_signed.json")
    wam_ai = load_json(RESULTS / "wm_signed_ai.json")
    wam_human = load_json(RESULTS / "wm_signed_human.json")
    ps_base = load_json(RESULTS / "ps_not_signed.json")

    def vals(data: dict, key: str) -> np.ndarray:
        return np.asarray([row[key] for row in data["images"]], dtype=float)

    return {
        "WAM": {
            "P0 original": summarize_accuracy(vals(wam_base, "bit_accuracy_original"), positive=False),
            "P1 watermarked": summarize_accuracy(vals(wam_base, "bit_accuracy_watermarked")),
            "P2a AI manifest": summarize_accuracy(vals(wam_ai, "bit_accuracy_watermarked")),
            "P2b human manifest": summarize_accuracy(vals(wam_human, "bit_accuracy_watermarked")),
        },
        "PixelSeal": {
            "P0 original": summarize_accuracy(vals(ps_base, "bit_accuracy_original"), positive=False),
            "P1 watermarked": summarize_accuracy(vals(ps_base, "bit_accuracy_watermarked")),
            "P2a AI manifest": summarize_accuracy(vals(ps_base, "bit_accuracy_watermarked")),
            "P2b human manifest": summarize_accuracy(vals(ps_base, "bit_accuracy_watermarked")),
        },
    }


def table3() -> dict:
    files = {
        "JPEG Q80": {
            "WAM": "wm_attack_jpeg_q80_human.json",
            "PixelSeal": "ps_attack_jpeg_q80.json",
        },
        "Crop 10%": {
            "WAM": "wm_attack_crop10_human.json",
            "PixelSeal": "ps_attack_crop10.json",
        },
        "Screenshot": {
            "WAM": "wm_attack_social_human.json",
            "PixelSeal": "ps_attack_social.json",
        },
    }
    out: dict[str, dict] = {}
    for condition, schemes in files.items():
        out[condition] = {}
        for scheme, filename in schemes.items():
            out[condition][scheme] = summarize_accuracy(
                image_values(filename, "bit_accuracy_watermarked")
            )
    return out


def grouped_detection(filename: Path) -> dict:
    rows = load_json(filename)["rows"]
    out: dict[str, dict] = {}
    for crop in sorted({float(row["crop_frac"]) for row in rows}):
        group = [row for row in rows if float(row["crop_frac"]) == crop]
        acc = np.asarray([row["bit_accuracy"] for row in group], dtype=float)
        detected = np.asarray([bool(row["detected"]) for row in group], dtype=bool)
        out[f"{crop:.2f}"] = {
            "n": len(group),
            "bit_accuracy_mean": float(acc.mean()),
            "bit_accuracy_ci95": bootstrap_mean_ci(acc),
            "bit_accuracy_min": float(acc.min()),
            "detection_n": int(detected.sum()),
            "detection_pct": float(100 * detected.mean()),
            "detection_ci95": bootstrap_rate_ci(detected),
            "escape_n": int((~detected).sum()),
            "escape_pct": float(100 * (~detected).mean()),
        }
    return out


def published_regional_summary(filename: Path) -> dict:
    """Normalize the bootstrap summaries saved by the original experiment."""
    rows = load_json(filename)
    return {
        f"{float(row['crop_frac']):.2f}": {
            "n": int(row["n"]),
            "bit_accuracy_mean": float(row["bit_acc_mean"]),
            "bit_accuracy_ci95": [float(x) for x in row["bit_acc_ci95"]],
            "bit_accuracy_min": float(row["bit_acc_min"]),
            "detection_pct": float(row["detection_rate_pct"]),
            "detection_ci95": [float(x) for x in row["detection_rate_ci95"]],
            "escape_pct": float(row["escape_rate_pct"]),
        }
        for row in rows
    }


def table6() -> dict:
    rows = load_json(RESULTS / "defense_e2e" / "audit_retention.json")["rows"]
    escapes = [row for row in rows if row["audit_v1_deployed"] == "SILENT_ESCAPE"]

    def summarize(group: list[dict]) -> dict:
        tampered = sum(row["audit_v2_region_aware"] == "TAMPERED_REGION" for row in group)
        suppressed = sum(row["audit_v2_region_aware"] == "WATERMARK_SUPPRESSED" for row in group)
        flagged = tampered + suppressed
        total = len(group)
        return {
            "silent_n": total,
            "tampered_n": tampered,
            "suppressed_n": suppressed,
            "flagged_n": flagged,
            "flagged_pct": 100 * flagged / total if total else 0.0,
            "remaining_n": total - flagged,
            "remaining_pct": 100 * (total - flagged) / total if total else 0.0,
        }

    selected = {}
    for crop in (0.10, 0.20, 0.30, 0.40):
        selected[f"{crop:.2f}"] = summarize(
            [row for row in escapes if float(row["crop_frac"]) == crop]
        )
    selected["overall_all_9_severities"] = summarize(escapes)
    return selected


def main() -> None:
    output = {
        "metadata": {
            "source": "existing per-image JSON results",
            "n_boot": N_BOOT,
            "bootstrap_seed": SEED,
            "detection_threshold": THRESHOLD,
        },
        "table2": table2(),
        "table3": table3(),
        "table4_sdxl": published_regional_summary(RESULTS / "regional_v2" / "summary.json"),
        "table5_sdxl": published_regional_summary(RESULTS / "regional_v2" / "summary.json"),
        "table5_pixart": published_regional_summary(RESULTS / "regional_v2_pixart" / "summary.json"),
        "table6": table6(),
    }
    with OUTPUT.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False)
    overall = output["table6"]["overall_all_9_severities"]
    if (overall["flagged_n"], overall["silent_n"]) != (686, 752):
        raise RuntimeError(f"Unexpected defense aggregate: {overall}")
    print(f"wrote {OUTPUT}")
    print(
        "verified defense aggregate: "
        f"{overall['flagged_pct']:.1f}% "
        f"({overall['flagged_n']}/{overall['silent_n']})"
    )


if __name__ == "__main__":
    main()
