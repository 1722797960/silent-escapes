"""Shared, dependency-light helpers for the LaMa inpainting experiment."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable, Sequence


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def select_stratified_indices(
    coverages: Sequence[float], pilot_count: int
) -> list[int]:
    """Choose deterministic mask-coverage quantiles, including both extremes."""
    n = len(coverages)
    if not 0 < pilot_count <= n:
        raise ValueError(f"pilot_count must be in [1, {n}], got {pilot_count}")
    if pilot_count == n:
        return list(range(n))

    ordered = sorted(range(n), key=lambda idx: (coverages[idx], idx))
    if pilot_count == 1:
        return [ordered[n // 2]]

    positions = [round(i * (n - 1) / (pilot_count - 1)) for i in range(pilot_count)]
    selected = [ordered[pos] for pos in positions]
    if len(set(selected)) != pilot_count:
        raise AssertionError("quantile selection unexpectedly produced duplicates")
    return selected


def classify_inpainting(
    bit_accuracy: float,
    assertion_present: bool,
    payload_hash_ok: bool,
    region_retention: float,
    logit_in_region: float,
    bit_threshold: float = 0.75,
    retention_floor: float = 0.05,
    logit_threshold: float = 0.5,
) -> tuple[str, str]:
    """Return baseline and region-aware verdicts using the paper's ordering."""
    baseline = "INTEGRITY_CLASH" if bit_accuracy >= bit_threshold else "SILENT_ESCAPE"
    if bit_accuracy >= bit_threshold:
        region_aware = "INTEGRITY_CLASH"
    elif not (assertion_present and payload_hash_ok):
        region_aware = "ASSERTION_MISSING"
    elif region_retention < retention_floor:
        region_aware = "TAMPERED_REGION"
    elif logit_in_region < logit_threshold:
        region_aware = "WATERMARK_SUPPRESSED"
    else:
        region_aware = "SILENT_OTHER"
    return baseline, region_aware


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def all_true(values: Iterable[bool]) -> bool:
    return all(bool(value) for value in values)
