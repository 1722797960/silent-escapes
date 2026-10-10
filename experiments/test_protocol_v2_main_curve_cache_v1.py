from __future__ import annotations

import unittest

from protocol_v2_main_curve_cache_v1 import inventory


class MainCurveCacheTest(unittest.TestCase):
    def test_inventory_groups_hits_by_rho(self) -> None:
        class Cache:
            @staticmethod
            def contains(key: str) -> bool:
                return key.endswith("hit")

        records = [
            {"pixel_key": "a-hit", "rho_requested": 0.0},
            {"pixel_key": "b-miss", "rho_requested": 0.0},
            {"pixel_key": "c-miss", "rho_requested": 0.8},
        ]
        result = inventory(records, Cache())
        self.assertEqual(result["unique_pixel_count"], 3)
        self.assertEqual(result["cache_hits"], 1)
        self.assertEqual(result["cache_misses"], 2)
        self.assertEqual(result["per_rho"]["0.0"]["cache_hits"], 1)
        self.assertEqual(result["per_rho"]["0.8"]["cache_misses"], 1)


if __name__ == "__main__":
    unittest.main()
