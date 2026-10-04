"""Pure helpers for the WAM granularity-control experiment."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


CONDITIONS = (
    "subject_region",
    "random_translated_region",
    "fullframe_native",
    "fullframe_energy_matched",
)


def validate_conditions(conditions: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    values = tuple(conditions)
    required = {"subject_region", "fullframe_native", "fullframe_energy_matched"}
    if not required.issubset(values):
        raise ValueError(f"missing required conditions: {sorted(required - set(values))}")
    random_controls = [value for value in values if value.startswith("random_")]
    if len(random_controls) != 1:
        raise ValueError("configuration must contain exactly one random control")
    return values


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config["message_bits"]) != 32:
        raise ValueError("this experiment expects the 32-bit WAM checkpoint")
    threshold = int(config["matched_bits_threshold"])
    if not 0 <= threshold <= int(config["message_bits"]):
        raise ValueError("matched_bits_threshold is outside the payload range")
    return config


def stable_seed(seed: int, image_name: str, namespace: str) -> int:
    material = f"{seed}|{namespace}|{image_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def centered_crop_box(width: int, height: int, rho: float) -> tuple[int, int, int, int]:
    """Match C1's centered square crop for removed-area fraction rho."""
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"rho must be in [0, 1), got {rho}")
    if rho == 0.0:
        return 0, 0, width, height
    scale = math.sqrt(1.0 - rho)
    crop_w = max(1, min(width, int(round(width * scale))))
    crop_h = max(1, min(height, int(round(height * scale))))
    x0 = (width - crop_w) // 2
    y0 = (height - crop_h) // 2
    return x0, y0, x0 + crop_w, y0 + crop_h


def crop_and_resize(image: Image.Image, rho: float) -> tuple[Image.Image, tuple[int, int, int, int], float]:
    width, height = image.size
    box = centered_crop_box(width, height, rho)
    x0, y0, x1, y1 = box
    actual = 1.0 - ((x1 - x0) * (y1 - y0) / (width * height))
    if box == (0, 0, width, height):
        return image.copy(), box, actual
    cropped = image.crop(box).resize((width, height), Image.Resampling.LANCZOS)
    return cropped, box, actual


def translate_mask_same_area(mask: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """Reposition a binary mask without clipping or changing its area.

    A bounded translation is preferred. If the foreground bounding box already
    spans the canvas, use a recorded cyclic translation instead of silently
    returning the original mask.
    """
    binary = np.asarray(mask, dtype=bool)
    ys, xs = np.where(binary)
    if not len(xs):
        raise ValueError("cannot translate an empty mask")
    height, width = binary.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    patch = binary[y0:y1, x0:x1]
    patch_h, patch_w = patch.shape
    max_x = width - patch_w
    max_y = height - patch_h
    rng = np.random.default_rng(seed)
    candidates = (max_x + 1) * (max_y + 1)
    wrapped = False
    operation = "bounded_translation"
    if candidates <= 1:
        wrapped = True
        operation = "cyclic_translation"
        translated = binary.copy()
        offset_x = offset_y = 0
        if width > 1 or height > 1:
            for _ in range(32):
                candidate_x = int(rng.integers(0, width)) if width > 1 else 0
                candidate_y = int(rng.integers(0, height)) if height > 1 else 0
                if candidate_x == 0 and candidate_y == 0:
                    continue
                candidate = np.roll(binary, shift=(candidate_y, candidate_x), axis=(0, 1))
                if not np.array_equal(candidate, binary):
                    translated = candidate
                    offset_x, offset_y = candidate_x, candidate_y
                    break
        ys_new, xs_new = np.where(translated)
        new_bbox = [int(xs_new.min()), int(ys_new.min()),
                    int(xs_new.max()) + 1, int(ys_new.max()) + 1]
        new_x, new_y = new_bbox[0], new_bbox[1]
    else:
        original_index = y0 * (max_x + 1) + x0
        draw = int(rng.integers(0, candidates - 1))
        if draw >= original_index:
            draw += 1
        new_y, new_x = divmod(draw, max_x + 1)
        translated = np.zeros_like(binary)
        translated[new_y:new_y + patch_h, new_x:new_x + patch_w] = patch
        offset_x, offset_y = new_x - x0, new_y - y0
        new_bbox = [new_x, new_y, new_x + patch_w, new_y + patch_h]
    if int(translated.sum()) != int(binary.sum()):
        raise AssertionError("translated mask changed foreground area")
    return translated, {
        "operation": operation,
        "wrapped": wrapped,
        "original_bbox_px": [x0, y0, x1, y1],
        "translated_bbox_px": new_bbox,
        "offset_px": [offset_x, offset_y],
        "translation_degenerate": bool(np.array_equal(translated, binary)),
    }


def random_compact_mask_same_area(
    shape: tuple[int, int], foreground_pixels: int, seed: int
) -> tuple[np.ndarray, dict]:
    """Create a contiguous near-square mask with an exact requested area.

    The compact patch is placed uniformly over every fitting top-left position.
    It never wraps or clips. A partially filled final row preserves the exact
    foreground-pixel count while keeping the support 4-connected.
    """
    height, width = shape
    count = int(foreground_pixels)
    if height <= 0 or width <= 0:
        raise ValueError("mask shape must be positive")
    if not 0 < count <= height * width:
        raise ValueError("foreground_pixels is outside the canvas area")
    patch_w = min(width, max(1, int(math.ceil(math.sqrt(count * width / height)))))
    patch_h = int(math.ceil(count / patch_w))
    while patch_h > height:
        patch_w += 1
        if patch_w > width:
            raise AssertionError("failed to fit compact equal-area mask")
        patch_h = int(math.ceil(count / patch_w))
    full_rows, remainder = divmod(count, patch_w)
    used_h = full_rows + int(remainder > 0)
    rng = np.random.default_rng(seed)
    x0 = int(rng.integers(0, width - patch_w + 1))
    y0 = int(rng.integers(0, height - used_h + 1))
    result = np.zeros((height, width), dtype=bool)
    if full_rows:
        result[y0:y0 + full_rows, x0:x0 + patch_w] = True
    if remainder:
        result[y0 + full_rows, x0:x0 + remainder] = True
    if int(result.sum()) != count:
        raise AssertionError("compact mask area mismatch")
    return result, {
        "operation": "compact_equal_area_random_placement",
        "wrapped": False,
        "foreground_pixels": count,
        "compact_bbox_px": [x0, y0, x0 + patch_w, y0 + used_h],
        "compact_patch_width": patch_w,
        "compact_patch_height": used_h,
        "partial_final_row_pixels": remainder,
        "translation_degenerate": False,
    }


def quantize_rgb01(array: np.ndarray) -> np.ndarray:
    """Match torchvision ToPILImage's float-to-uint8 truncation."""
    # WAM and torchvision operate on float32 tensors before ToPILImage.
    clipped = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    return (clipped * 255.0).astype(np.uint8).astype(np.float64) / 255.0


def mse(left: np.ndarray, right: np.ndarray) -> float:
    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.mean(diff * diff))


def psnr_from_mse(value: float) -> float | None:
    if value <= 0.0:
        return None
    return float(10.0 * math.log10(1.0 / value))


def energy_match_fullframe(
    original: np.ndarray,
    candidate: np.ndarray,
    target_mse: float,
    iterations: int = 28,
) -> tuple[np.ndarray, float, float]:
    """Scale a full-frame candidate to match a target post-quantization MSE."""
    original_q = quantize_rgb01(original)
    candidate = np.asarray(candidate, dtype=np.float64)
    original = np.asarray(original, dtype=np.float64)
    native_q = quantize_rgb01(candidate)
    native_mse = mse(native_q, original_q)
    if target_mse < 0.0:
        raise ValueError("target MSE must be non-negative")
    if target_mse == 0.0 or native_mse == 0.0:
        return original_q, 0.0, 0.0
    if target_mse > native_mse * (1.0 + 1e-12):
        raise ValueError("target MSE exceeds native full-frame MSE")
    delta = candidate - original
    low, high = 0.0, 1.0
    best = original_q
    best_alpha = 0.0
    best_mse = 0.0
    best_error = abs(target_mse)
    for _ in range(iterations):
        alpha = (low + high) / 2.0
        trial = quantize_rgb01(original + alpha * delta)
        trial_mse = mse(trial, original_q)
        error = abs(trial_mse - target_mse)
        if error < best_error:
            best, best_alpha, best_mse, best_error = trial, alpha, trial_mse, error
        if trial_mse < target_mse:
            low = alpha
        else:
            high = alpha
    return best, float(best_alpha), float(best_mse)


def decoded_rgb_sha256(path: Path) -> str:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
