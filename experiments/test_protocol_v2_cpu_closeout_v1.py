"""CPU fixtures for comparison identities, geometry and inventory invariants."""
import base64
import unittest
from unittest.mock import patch

from protocol_v2_pixart_derived_v1 import transition_summary, unique_index
from protocol_v2_c1_inventory_v1 import (
    checked_geometry, checked_pairs, mask_aware_selection, summarize_inventory, write_results,
)
from c1_attack_suite_v1 import VALID_METHODS, crop_box, stable_rng


class CloseoutTests(unittest.TestCase):
    def test_net_change_is_not_total_change(self):
        rows = [{"match_count": 23, "legacy_evidence": "region-removed", "evidence": "silent-other"},
                {"match_count": 23, "legacy_evidence": "silent-other", "evidence": "signal-suppressed"},
                {"match_count": 23, "legacy_evidence": "region-removed", "evidence": "signal-suppressed"}]
        result = transition_summary(rows)
        self.assertEqual(result["changed_evidence"], 3)
        self.assertEqual((result["review_to_silent"], result["silent_to_review"]), (1, 1))
        self.assertEqual((result["legacy_reviews"], result["v2_reviews"]), (2, 2))

    def test_joins_use_image_names_not_order_indices(self):
        rows = [{"image": "pixart_0112.png", "crop_frac": .15}]
        result = unique_index(rows, lambda r: (r["image"], round(r["crop_frac"], 2)))
        self.assertIn(("pixart_0112.png", .15), result)
        with self.assertRaises(RuntimeError):
            unique_index(rows + rows, lambda r: r["image"])

    def test_full_c1_cartesian_pairing(self):
        rows = [{"attack_id": f"{i:04d}|{m}|{r/10:.3f}", "method": m}
                for i in range(200) for m in VALID_METHODS for r in range(9)]
        first, second = checked_pairs(rows, [dict(r) for r in rows])
        self.assertEqual(len(first), 10800)
        self.assertEqual(len(second), 10800)
        with self.assertRaises(RuntimeError):
            checked_pairs(rows[:-1], rows)
        changed = [dict(r) for r in rows]
        changed[0]["method"] = "random"
        with self.assertRaises(RuntimeError):
            checked_pairs(rows, changed)
        with self.assertRaises(RuntimeError):
            checked_pairs(rows[:-1]+rows[:1], rows)

    def geometry_fixture(self, method="center", rho=.4):
        meta = {"input_image": "fixture.png", "signed_image": "fixture_signed.png",
                "bitmap64": base64.b64encode(bytes([255]*512)).decode()}
        box = crop_box(method, rho, 1024, 1024, stable_rng(20260928, meta["input_image"], rho))
        x0, y0, x1, y1 = box
        from c1_attack_suite_v1 import mask_retention, unpack_mask
        retained = mask_retention(unpack_mask(meta["bitmap64"]), box, 1024, 1024)
        row = {"method": method, "rho_requested": rho, "original_size": [1024, 1024],
               "crop_box_px": list(box), "input_image": meta["input_image"],
               "signed_image": meta["signed_image"], "mask_retention_geometry": round(retained, 8),
               "rho_actual": round(1-(x1-x0)*(y1-y0)/(1024*1024), 8)}
        return row, meta, {"random_seed": 20260928}

    def test_geometry_rounding_and_random_seed(self):
        for method in VALID_METHODS:
            row, meta, config = self.geometry_fixture(method)
            checked_geometry(row, meta, config)
        row, meta, config = self.geometry_fixture("random")
        row["crop_box_px"][0] += 1
        with self.assertRaises(RuntimeError):
            checked_geometry(row, meta, config)

    def test_removed_area_is_not_per_side_fraction(self):
        row, meta, config = self.geometry_fixture()
        self.assertAlmostEqual(row["rho_actual"], .4, delta=.002)
        self.assertNotAlmostEqual(1-(1-2*.4)**2, row["rho_actual"], delta=.1)
        row["rho_actual"] = .96
        with self.assertRaises(RuntimeError):
            checked_geometry(row, meta, config)

    def test_mask_aware_ties_do_not_use_detector(self):
        rows = [{"method": m, "image_index": 0, "rho_requested": .4,
                 "region_retention_unrounded": .1, "attack_id": m,
                 "pixel_key": m, "cache_hit": False} for m in VALID_METHODS if m not in ("center", "random")]
        selected = mask_aware_selection(list(reversed(rows)))
        self.assertEqual(selected[0]["selected_attack_id"], "left")
        rows[1]["region_retention_unrounded"] = 0
        self.assertEqual(mask_aware_selection(rows)[0]["selected_attack_id"], "right")
        with self.assertRaises(RuntimeError):
            mask_aware_selection(rows[:-1])

    def test_inventory_counts_records_and_unique_pixels_separately(self):
        rows = [{"method": m, "image_index": 0, "rho_requested": 0., "input_image": "fixture.png",
                 "region_retention_unrounded": 1., "region_retention": 1., "attack_id": m,
                 "pixel_key": "same", "cache_hit": True, "legacy_match_count": 24}
                for m in ("left", "right", "top", "bottom")]
        context = {"cache_states": {"same": "0"*32}, "model_hash": "params", "checkpoint_hash": "weights"}
        with patch("protocol_v2_c1_inventory_v1.sha256_file", return_value="fixture-hash"):
            summary, missing, selected = summarize_inventory(rows, {r["attack_id"]: r for r in rows}, context)
        self.assertEqual((summary["records"], summary["unique_pixels"]), (4, 1))
        self.assertEqual((summary["record_cache_hits"], summary["unique_cache_hits"]), (4, 1))
        self.assertEqual(missing, [])
        self.assertEqual(len(selected), 1)
        with patch("protocol_v2_c1_inventory_v1.atomic_json"), patch("pathlib.Path.write_text") as output:
            write_results(rows, summary, missing, selected)
        self.assertIn("4 paired records; 1 unique", output.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
