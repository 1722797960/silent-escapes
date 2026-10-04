"""Inventory protocol-v2 pixel-cache coverage for the full semantics matrix."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from protocol_v2_cache import PixelCache, decoded_rgb_key
from protocol_v2_pilot_v1 import atomic_json, load_jsonl


ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = ("left", "right", "top", "bottom")


def mask_aware_rows(c1_dir: Path, *, rho: float) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(c1_dir / "audit_rows.jsonl"):
        if row["method"] in DIRECTIONS and float(row["rho_requested"]) == rho:
            grouped[row["input_image"]].append(row)
    priority = {method: index for index, method in enumerate(DIRECTIONS)}
    selected: list[dict[str, Any]] = []
    for image in sorted(grouped):
        candidates = grouped[image]
        if len(candidates) != 4:
            raise ValueError(f"mask-aware selection requires four directions: {image}")
        chosen = min(
            candidates,
            key=lambda row: (float(row["region_retention"]), priority[row["method"]]),
        )
        selected.append(chosen)
    return selected


def mask_aware_paths(c1_dir: Path, *, rho: float) -> list[Path]:
    return [c1_dir / row["output_relpath"] for row in mask_aware_rows(c1_dir, rho=rho)]


def dataset_paths(e2e_root: Path, *, c1_rho: float) -> dict[str, list[Path]]:
    results = e2e_root / "results"
    benign = results / "benign_controls_v1_full" / "unsigned"
    centered = results / "benign_centered_crops_v1" / "unsigned"
    inpainting = results / "inpainting_e2e_v1_full" / "inpainted_unsigned"
    c1 = results / "c1_v1" / "full"
    return {
        "identity_resave": sorted((benign / "identity_resave").glob("*.png")),
        "jpeg_q90": sorted((benign / "jpeg_q90").glob("*.png")),
        "resize_90": sorted((benign / "resize_90").glob("*.png")),
        "safe_crop_1pct": sorted((benign / "safe_crop_1pct").glob("*.png")),
        "centered_crop_10pct_per_side": sorted((centered / "crop10").glob("*.png")),
        "centered_crop_20pct_per_side": sorted((centered / "crop20").glob("*.png")),
        "lama_inpainting": sorted(inpainting.glob("*.png")),
        f"mask_aware_crop_rho_{c1_rho:.1f}": mask_aware_paths(c1, rho=c1_rho),
    }


def inventory(
    datasets: dict[str, list[Path]], *, cache: PixelCache, expected_per_dataset: int
) -> dict[str, Any]:
    per_dataset: dict[str, Any] = {}
    all_keys: set[str] = set()
    all_missing: set[str] = set()
    total_rows = 0
    for name, paths in datasets.items():
        if len(paths) != expected_per_dataset:
            raise RuntimeError(
                f"{name} expected {expected_per_dataset} images, observed {len(paths)}"
            )
        keys: list[str] = []
        hits = 0
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            key, _ = decoded_rgb_key(path)
            keys.append(key)
            if cache.get(key) is not None:
                hits += 1
            else:
                all_missing.add(key)
        unique = set(keys)
        per_dataset[name] = {
            "image_count": len(paths),
            "unique_pixel_count": len(unique),
            "cache_hits": hits,
            "cache_misses": len(paths) - hits,
        }
        total_rows += len(paths)
        all_keys.update(unique)
    return {
        "dataset_count": len(datasets),
        "expected_per_dataset": expected_per_dataset,
        "total_rows": total_rows,
        "unique_pixels": len(all_keys),
        "cached_unique_pixels": len(all_keys - all_missing),
        "missing_unique_pixels": len(all_missing),
        "gpu_inference_required_for_full_exact_cache": bool(all_missing),
        "datasets": per_dataset,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e-root", type=Path, default=ROOT)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/protocol_v2/cache_inventory_v1",
    )
    parser.add_argument("--expected-per-dataset", type=int, default=200)
    parser.add_argument("--c1-rho", type=float, default=0.4)
    args = parser.parse_args()
    e2e_root = args.e2e_root.resolve(strict=True)
    pilot = args.pilot_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = inventory(
        dataset_paths(e2e_root, c1_rho=args.c1_rho),
        cache=PixelCache(pilot / "pixel_cache"),
        expected_per_dataset=args.expected_per_dataset,
    )
    summary["c1_selection"] = {
        "method": "mask_aware",
        "rho": args.c1_rho,
        "policy": "minimum signed-region retention among left/right/top/bottom",
    }
    atomic_json(output / "summary.json", summary)
    lines = [
        "# Protocol v2 full-matrix cache inventory",
        "",
        f"Datasets: {summary['dataset_count']}; rows: {summary['total_rows']}; "
        f"unique decoded pixel arrays: {summary['unique_pixels']}.",
        f"Cached unique pixels: {summary['cached_unique_pixels']}; "
        f"missing unique pixels: {summary['missing_unique_pixels']}.",
        "",
        "This command only decodes images and inspects the cache. It never loads the detector.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
