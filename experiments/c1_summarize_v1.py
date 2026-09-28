"""Summarize C1 crop attacks with Wilson and paired-bootstrap intervals."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_C1 = ROOT / "results" / "c1_v1"
DIRECTIONS = ("left", "right", "top", "bottom")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator,
                   repeats: int) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    low, high = np.percentile(values[indices].mean(axis=1), [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def wilson_interval(successes: int, total: int,
                    z: float = 1.959963984540054) -> list[float]:
    """Return a two-sided 95% Wilson score interval as proportions."""
    if total <= 0:
        return [float("nan"), float("nan")]
    p_hat = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(p_hat * (1.0 - p_hat) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return [round(center - half_width, 4), round(center + half_width, 4)]


def add_mask_aware(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        if row["method"] in DIRECTIONS:
            grouped[(row["input_image"], float(row["rho_requested"]))].append(row)
    derived = []
    priority = {method: index for index, method in enumerate(DIRECTIONS)}
    for candidates in grouped.values():
        if len(candidates) != 4:
            raise ValueError("mask-aware derivation requires all four directions")
        chosen = min(candidates, key=lambda row: (
            float(row["region_retention"]), priority[row["method"]]))
        item = dict(chosen)
        item["source_direction"] = chosen["method"]
        item["method"] = "mask_aware"
        item["attack_id"] = chosen["attack_id"] + "|mask_aware"
        derived.append(item)
    return rows + derived


def rho50(points: list[dict]) -> float | None:
    points = sorted(points, key=lambda item: item["rho"])
    for left, right in zip(points, points[1:]):
        if left["escape_rate_pct"] >= 50:
            return left["rho"]
        if left["escape_rate_pct"] < 50 <= right["escape_rate_pct"]:
            span = right["escape_rate_pct"] - left["escape_rate_pct"]
            if span == 0:
                return right["rho"]
            weight = (50 - left["escape_rate_pct"]) / span
            return round(left["rho"] + weight * (right["rho"] - left["rho"]), 4)
    if points and points[-1]["escape_rate_pct"] >= 50:
        return points[-1]["rho"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1-dir", default=str(DEFAULT_C1))
    args = parser.parse_args()
    c1_dir = Path(args.c1_dir).resolve()
    config = json.loads((c1_dir / "config_effective.json").read_text(encoding="utf-8"))
    repeats = int(config["bootstrap_resamples"])
    rows = add_mask_aware(load_jsonl(c1_dir / "audit_rows.jsonl"))
    rng = np.random.default_rng(42)

    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], float(row["rho_requested"]))].append(row)

    center_lookup = {
        (row["input_image"], float(row["rho_requested"])): row
        for row in rows if row["method"] == "center"
    }
    summary = []
    for (method, rho), group in sorted(grouped.items()):
        bit = np.array([row["bit_accuracy"] for row in group], dtype=float)
        escape = np.array([not row["detected"] for row in group], dtype=float)
        retention = np.array([row["region_retention"] for row in group], dtype=float)
        valid = np.array([row["manifest_valid"] for row in group], dtype=float)
        escaped_rows = [row for row in group if not row["detected"]]
        defense_flags = np.array([
            row["audit_v2_region_aware"] in ("TAMPERED_REGION", "WATERMARK_SUPPRESSED")
            for row in escaped_rows], dtype=float)
        paired = []
        if method != "center":
            for row in group:
                baseline = center_lookup[(row["input_image"], rho)]
                paired.append(float(not row["detected"]) - float(not baseline["detected"]))
        paired_array = np.array(paired, dtype=float)
        summary.append({
            "method": method,
            "rho": rho,
            "n": len(group),
            "bit_accuracy_mean": round(float(bit.mean()), 6),
            "bit_accuracy_ci95": bootstrap_mean(bit, rng, repeats),
            "escape_rate_pct": round(float(escape.mean() * 100), 3),
            "escape_rate_ci95_pct": [
                round(x * 100, 2)
                for x in wilson_interval(int(escape.sum()), len(escape))
            ],
            "region_retention_mean": round(float(retention.mean()), 6),
            "manifest_valid_rate_pct": round(float(valid.mean() * 100), 3),
            "defense_flag_rate_among_escapes_pct": (
                round(float(defense_flags.mean() * 100), 3) if len(defense_flags) else None),
            "paired_escape_difference_vs_center_pp": (
                round(float(paired_array.mean() * 100), 3) if len(paired_array) else 0.0),
            "paired_difference_ci95_pp": (
                [round(x * 100, 2)
                 for x in bootstrap_mean(paired_array, rng, repeats)]
                if len(paired_array) else [0.0, 0.0]),
        })

    methods = sorted({item["method"] for item in summary})
    aggregate = {}
    for method in methods:
        points = sorted([item for item in summary if item["method"] == method],
                        key=lambda item: item["rho"])
        x = np.array([item["rho"] for item in points], dtype=float)
        y = np.array([item["escape_rate_pct"] / 100 for item in points], dtype=float)
        aggregate[method] = {
            "auec": round(float(np.trapezoid(y, x)), 6) if len(x) > 1 else None,
            "rho50": rho50(points),
        }

    payload = {"config": config, "summary": summary, "aggregate": aggregate}
    (c1_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with (c1_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    preferred = ["center", "left", "right", "top", "bottom", "random", "mask_aware"]
    display_names = {"mask_aware": "mask-aware"}
    colors = plt.cm.tab10(np.linspace(0, 1, len(preferred)))
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for method, color in zip(preferred, colors):
        points = sorted([item for item in summary if item["method"] == method],
                        key=lambda item: item["rho"])
        if not points:
            continue
        x = np.array([item["rho"] for item in points])
        y = np.array([item["escape_rate_pct"] for item in points])
        low = np.array([item["escape_rate_ci95_pct"][0] for item in points])
        high = np.array([item["escape_rate_ci95_pct"][1] for item in points])
        linewidth = 2.6 if method in ("center", "mask_aware") else 1.5
        ax.plot(x, y, marker="o", linewidth=linewidth,
                label=display_names.get(method, method), color=color)
        ax.fill_between(x, low, high, color=color, alpha=0.10)
    ax.set_xlabel("Removed image-area fraction $\\rho$")
    ax.set_ylabel("Silent-escape rate (%)")
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(c1_dir / "fig_c1_escape_by_attack.pdf", bbox_inches="tight")
    fig.savefig(c1_dir / "fig_c1_escape_by_attack.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    random_rows = [row for row in rows if row["method"] == "random"]
    random_groups = defaultdict(list)
    for row in random_rows:
        random_groups[float(row["rho_requested"])].append(float(row["bit_accuracy"]))
    rhos = sorted(random_groups)
    if rhos:
        fig, ax = plt.subplots(figsize=(8.2, 4.7))
        ax.boxplot([random_groups[rho] for rho in rhos], tick_labels=[str(rho) for rho in rhos],
                   showfliers=False)
        ax.axhline(float(config["watermark_threshold"]), color="black", linestyle="--",
                   linewidth=1.5, label="Detection threshold")
        ax.set_xlabel("Removed image-area fraction $\\rho$")
        ax.set_ylabel("Per-image bit accuracy")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(c1_dir / "fig_c1_random_distribution.pdf", bbox_inches="tight")
        fig.savefig(c1_dir / "fig_c1_random_distribution.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    print(f"wrote {c1_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
