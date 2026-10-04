"""Summarize paired WAM granularity controls and draw the comparison curve."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from granularity_control_common_v1 import load_config, load_jsonl, validate_conditions


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "granularity_control_v1.json"
DEFAULT_EXPERIMENT = ROOT / "results" / "granularity_control_v1" / "pilot"


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [float("nan"), float("nan")]
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denominator
    return [center - half, center + half]


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, repeats: int) -> list[float]:
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    return [float(v) for v in np.percentile(values[indices].mean(axis=1), [2.5, 97.5])]


def rho50(points: list[dict]) -> float | None:
    points = sorted(points, key=lambda item: item["rho"])
    for left, right in zip(points, points[1:]):
        if left["escape_rate_pct"] >= 50:
            return left["rho"]
        if left["escape_rate_pct"] < 50 <= right["escape_rate_pct"]:
            span = right["escape_rate_pct"] - left["escape_rate_pct"]
            return left["rho"] + (50 - left["escape_rate_pct"]) / span * (right["rho"] - left["rho"])
    return points[-1]["rho"] if points and points[-1]["escape_rate_pct"] >= 50 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT))
    args = parser.parse_args()
    config = load_config(Path(args.config))
    conditions = validate_conditions(config["conditions"])
    experiment = Path(args.experiment_dir).resolve()
    embed_rows = load_jsonl(experiment / "embed_rows.jsonl")
    expected_images = len(embed_rows)
    expected_rhos = len(config["removed_area_fractions"])
    if expected_images == 0:
        raise SystemExit("no embed rows")

    by_condition = {}
    for condition in conditions:
        rows = load_jsonl(experiment / f"detection_{condition}.jsonl")
        expected = expected_images * expected_rhos
        if len(rows) != expected:
            raise RuntimeError(f"{condition}: {len(rows)} rows, expected {expected}")
        by_condition[condition] = rows
    clean = load_jsonl(experiment / "detection_clean_reference.jsonl")
    if len(clean) != expected_images:
        raise RuntimeError(f"clean_reference: {len(clean)} rows, expected {expected_images}")

    repeats = int(config["bootstrap_resamples"])
    rng = np.random.default_rng(42)
    subject_lookup = {
        (int(row["image_index"]), float(row["rho_requested"])): row
        for row in by_condition["subject_region"]
    }
    summary = []
    for condition in conditions:
        grouped = defaultdict(list)
        for row in by_condition[condition]:
            grouped[float(row["rho_requested"])].append(row)
        baseline_detected = {
            int(row["image_index"])
            for row in grouped[0.0] if bool(row["detected"])
        }
        for rho, group in sorted(grouped.items()):
            bit = np.asarray([row["bit_accuracy"] for row in group], dtype=float)
            detected = np.asarray([bool(row["detected"]) for row in group], dtype=bool)
            escapes = ~detected
            qualified = [row for row in group if int(row["image_index"]) in baseline_detected]
            qualified_detected = sum(bool(row["detected"]) for row in qualified)
            paired = np.asarray([
                float(not row["detected"]) - float(not subject_lookup[(int(row["image_index"]), rho)]["detected"])
                for row in group
            ], dtype=float)
            interval = wilson(int(escapes.sum()), len(group))
            survival_interval = wilson(qualified_detected, len(qualified))
            summary.append({
                "condition": condition,
                "rho": rho,
                "n": len(group),
                "bit_accuracy_mean": round(float(bit.mean()), 6),
                "bit_accuracy_ci95": [round(v, 6) for v in bootstrap_mean(bit, rng, repeats)],
                "escape_rate_pct": round(float(escapes.mean() * 100), 3),
                "escape_rate_ci95_pct": [round(v * 100, 3) for v in interval],
                "baseline_qualified_n": len(qualified),
                "baseline_qualified_survival_pct": round(100 * qualified_detected / len(qualified), 3),
                "baseline_qualified_survival_ci95_pct": [round(v * 100, 3) for v in survival_interval],
                "paired_escape_difference_vs_subject_pp": round(float(paired.mean() * 100), 3),
                "paired_difference_ci95_pp": [round(v * 100, 3) for v in bootstrap_mean(paired, rng, repeats)],
            })

    quality = {}
    for condition in conditions:
        mses = np.asarray([row["conditions"][condition]["mse"] for row in embed_rows], dtype=float)
        psnrs = np.asarray([row["conditions"][condition]["psnr_db"] for row in embed_rows], dtype=float)
        marked = np.asarray([row["conditions"][condition]["marked_pixel_mse"] for row in embed_rows], dtype=float)
        quality[condition] = {
            "mean_mse": float(mses.mean()),
            "mean_psnr_db": float(psnrs.mean()),
            "mean_marked_pixel_mse": float(marked.mean()),
        }
    fpr_count = sum(bool(row["detected"]) for row in clean)
    fpr_interval = wilson(fpr_count, len(clean))
    ious = np.asarray([row["subject_random_iou"] for row in embed_rows], dtype=float)
    centroid_distances = np.asarray([
        row["subject_random_centroid_distance_normalized"] for row in embed_rows
    ], dtype=float)
    random_condition = next(value for value in conditions if value.startswith("random_"))
    if random_condition == "random_compact_equal_area":
        area_matches = sum(
            int(row["random_translation"]["foreground_pixels"])
            == int(row["subject_mask_pixels"])
            for row in embed_rows)
    else:
        area_matches = len(embed_rows)
    validation = {
        "subject_rgb_hash_matches": sum(bool(row["subject_hash_match"]) for row in embed_rows),
        "energy_match_max_relative_error": max(
            float(row["energy_match_relative_error"]) for row in embed_rows),
        "random_area_matches": area_matches,
        "random_wrap_count": sum(
            bool(row["random_translation"].get("wrapped", False)) for row in embed_rows),
        "subject_random_iou_mean": float(ious.mean()),
        "subject_random_iou_quartiles": [
            float(value) for value in np.percentile(ious, [25, 50, 75])],
        "subject_random_centroid_distance_normalized_mean": float(
            centroid_distances.mean()),
        "subject_random_centroid_distance_normalized_quartiles": [
            float(value) for value in np.percentile(centroid_distances, [25, 50, 75])],
    }
    aggregate = {}
    for condition in conditions:
        points = sorted([row for row in summary if row["condition"] == condition], key=lambda row: row["rho"])
        x = np.asarray([row["rho"] for row in points], dtype=float)
        y = np.asarray([row["escape_rate_pct"] / 100 for row in points], dtype=float)
        aggregate[condition] = {
            "auec": float(np.trapezoid(y, x)),
            "rho50": rho50(points),
        }
    payload = {
        "schema": "granularity-control-summary-1",
        "config": config,
        "image_count": expected_images,
        "clean_reference_fpr": {
            "false_positives": fpr_count,
            "n": len(clean),
            "rate_pct": 100 * fpr_count / len(clean),
            "wilson_ci95_pct": [100 * v for v in fpr_interval],
        },
        "embedding_validation": validation,
        "quality": quality,
        "summary": summary,
        "aggregate": aggregate,
    }
    (experiment / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (experiment / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    labels = {
        "subject_region": "Subject-aligned region",
        "random_translated_region": "Translated same-area region",
        "random_compact_equal_area": "Random compact equal-area region",
        "fullframe_native": "Full-frame (native strength)",
        "fullframe_energy_matched": "Full-frame (energy matched)",
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for condition in conditions:
        points = sorted([row for row in summary if row["condition"] == condition], key=lambda row: row["rho"])
        x = np.asarray([row["rho"] for row in points])
        y = np.asarray([row["escape_rate_pct"] for row in points])
        low = np.asarray([row["escape_rate_ci95_pct"][0] for row in points])
        high = np.asarray([row["escape_rate_ci95_pct"][1] for row in points])
        ax.plot(x, y, marker="o", linewidth=2, label=labels[condition])
        ax.fill_between(x, low, high, alpha=0.10)
    ax.set_xlabel("Removed image-area fraction $\\rho$")
    ax.set_ylabel("Watermark escape rate (%)")
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(experiment / "fig_granularity_control.png", dpi=180, bbox_inches="tight")
    fig.savefig(experiment / "fig_granularity_control.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {experiment / 'summary.json'}")


if __name__ == "__main__":
    main()
