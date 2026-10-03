"""Unit tests for deterministic declared-geometry perturbations."""
from __future__ import annotations

import unittest

import numpy as np

from protocol_v2_geometry_lie_v1 import (
    adaptive_high_logit_transform,
    grid_positions,
    is_full_frame_transform,
    out_of_bounds_transform,
    scaled_transform,
    shifted_transform,
)
from read_actions_v2 import parse_crop_transform


TRANSFORM = {
    "schema_version": "1",
    "coordinate_system": "source-pixel-half-open",
    "source_size_px": "100,80",
    "crop_box_px": "10,5,90,75",
    "output_size_px": "100,80",
    "resize_filter": "LANCZOS",
}


class GeometryPerturbationTests(unittest.TestCase):
    def test_shifted_and_scaled_stay_valid_and_change_box(self) -> None:
        for candidate in (shifted_transform(TRANSFORM, 0.1), scaled_transform(TRANSFORM, 0.8)):
            parsed = parse_crop_transform({"com.example.crop_transform": candidate})
            self.assertEqual(parsed["status"], "valid")
            self.assertNotEqual(candidate["crop_box_px"], TRANSFORM["crop_box_px"])

    def test_out_of_bounds_is_rejected(self) -> None:
        candidate = out_of_bounds_transform(TRANSFORM)
        parsed = parse_crop_transform({"com.example.crop_transform": candidate})
        self.assertEqual(parsed["status"], "invalid")
        self.assertIn("outside source bounds", " ".join(parsed["errors"]))

    def test_grid_is_sorted_unique_and_covers_endpoints(self) -> None:
        values = grid_positions(3, 17)
        self.assertEqual(values, [0, 1, 2, 3])

    def test_full_frame_is_explicitly_detected(self) -> None:
        full = dict(TRANSFORM, crop_box_px="0,0,100,80")
        self.assertTrue(is_full_frame_transform(full))
        self.assertFalse(is_full_frame_transform(TRANSFORM))

    def test_adaptive_search_is_deterministic_and_not_truthful(self) -> None:
        region = {
            "bitmap64": __import__("base64").b64encode(
                np.packbits(np.ones((64, 64), dtype=np.uint8)).tobytes()
            ).decode("ascii")
        }
        logits = np.tile(np.linspace(0, 1, 64, dtype=np.float32), (64, 1))
        first, first_meta = adaptive_high_logit_transform(
            TRANSFORM, region=region, mask_logits=logits,
            grid_points=5, retention_floor=0.05,
        )
        second, second_meta = adaptive_high_logit_transform(
            TRANSFORM, region=region, mask_logits=logits,
            grid_points=5, retention_floor=0.05,
        )
        self.assertEqual(first, second)
        self.assertEqual(first_meta, second_meta)
        self.assertNotEqual(first["crop_box_px"], TRANSFORM["crop_box_px"])
        self.assertTrue(first_meta["selection_uses_detector_logits"])


if __name__ == "__main__":
    unittest.main()
