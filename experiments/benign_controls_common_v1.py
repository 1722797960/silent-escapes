"""Dependency-light helpers for the benign-edit control experiment."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np


DIRECTIONS = ("left", "right", "top", "bottom")


def decode_bitmap64(encoded: str) -> np.ndarray:
    packed = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
    bits = np.unpackbits(packed)
    if bits.size != 64 * 64:
        raise ValueError(f"expected a 64x64 bitmap, observed {bits.size} bits")
    return bits.reshape(64, 64).astype(np.float32)


def crop_grid(grid: np.ndarray, side: str, fraction: float) -> np.ndarray:
    """Map an asserted grid through a one-sided crop."""
    if side not in DIRECTIONS:
        raise ValueError(f"unknown crop side: {side}")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    height, width = grid.shape
    if side in ("left", "right"):
        pixels = max(1, round(fraction * width))
        return grid[:, pixels:] if side == "left" else grid[:, : width - pixels]
    pixels = max(1, round(fraction * height))
    return grid[pixels:, :] if side == "top" else grid[: height - pixels, :]


def select_max_retention_crop(
    grid: np.ndarray, fraction: float
) -> tuple[str, float]:
    """Choose a mild one-sided crop using geometry only, never detector output."""
    denominator = max(float(grid.sum()), 1e-12)
    scored = [
        (side, float(crop_grid(grid, side, fraction).sum() / denominator))
        for side in DIRECTIONS
    ]
    priority = {side: index for index, side in enumerate(DIRECTIONS)}
    return max(scored, key=lambda item: (item[1], -priority[item[0]]))


def classify_benign(
    *,
    expected_region_assertion: bool,
    manifest_valid: bool,
    assertion_present: bool,
    payload_hash_ok: bool | None,
    bit_accuracy: float,
    region_retention: float | None,
    logit_in_region: float,
    bit_threshold: float = 0.75,
    retention_floor: float = 0.05,
    logit_threshold: float = 0.5,
) -> tuple[str, bool]:
    """Return the benign-control verdict and whether it is a region false flag."""
    if not manifest_valid:
        return "PIPELINE_INVALID_MANIFEST", False
    if not expected_region_assertion:
        return "NO_REGION_CLAIM", False
    if not assertion_present or payload_hash_ok is not True:
        return "PIPELINE_INVALID_ASSERTION", False
    if bit_accuracy >= bit_threshold:
        return "CONSISTENT_SYNTHETIC", False
    if region_retention is None:
        return "PIPELINE_INVALID_GEOMETRY", False
    if region_retention < retention_floor:
        return "TAMPERED_REGION", True
    if logit_in_region < logit_threshold:
        return "WATERMARK_SUPPRESSED", True
    return "BENIGN_NO_FLAG", False


def require_pipeline_valid(verdict: str, context: dict[str, Any]) -> None:
    if verdict.startswith("PIPELINE_INVALID"):
        raise RuntimeError(f"{verdict}: {context}")
