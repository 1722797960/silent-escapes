"""CPU-only unit tests for protocol-v2 parsing and validation."""
from __future__ import annotations

import hashlib
import unittest

from read_actions_v2 import parse_crop_transform, validate_region_assertion


def valid_transform() -> dict:
    return {
        "com.example.crop_transform": {
            "schema_version": "1",
            "coordinate_system": "source-pixel-half-open",
            "source_size_px": "64,48",
            "crop_box_px": "8,4,56,44",
            "output_size_px": "48,40",
            "resize_filter": "NONE",
        }
    }


class CropTransformTests(unittest.TestCase):
    def test_valid_csv_transform(self) -> None:
        parsed = parse_crop_transform(valid_transform())
        self.assertEqual(parsed["status"], "valid")
        self.assertEqual(parsed["transform"]["crop_box_px"], [8, 4, 56, 44])

    def test_missing_transform_is_explicit(self) -> None:
        self.assertEqual(parse_crop_transform({})["status"], "missing")

    def test_numeric_array_is_rejected(self) -> None:
        transform = valid_transform()
        transform["com.example.crop_transform"]["crop_box_px"] = [8, 4, 56, 44]
        parsed = parse_crop_transform(transform)
        self.assertEqual(parsed["status"], "invalid")

    def test_out_of_bounds_is_rejected(self) -> None:
        transform = valid_transform()
        transform["com.example.crop_transform"]["crop_box_px"] = "8,4,65,44"
        parsed = parse_crop_transform(transform)
        self.assertEqual(parsed["status"], "invalid")


class RegionAssertionTests(unittest.TestCase):
    def test_reference_round_trip_contract(self) -> None:
        bits = "01011010010110100101101001011010"
        region = {
            "schema_version": "2",
            "reference_bits": bits,
            "reference_length": "32",
            "required_matches": "24",
            "payload_hash": hashlib.sha256(bits.encode("ascii")).hexdigest(),
        }
        checked = validate_region_assertion(region)
        self.assertEqual(checked["status"], "valid")
        self.assertEqual(checked["reference_bits"], bits)
        self.assertTrue(checked["payload_hash_ok"])

    def test_inconsistent_digest_is_rejected(self) -> None:
        bits = "0" * 32
        region = {
            "schema_version": "2",
            "reference_bits": bits,
            "reference_length": "32",
            "required_matches": "24",
            "payload_hash": "0" * 64,
        }
        self.assertEqual(validate_region_assertion(region)["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
