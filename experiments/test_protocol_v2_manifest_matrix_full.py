from __future__ import annotations

import unittest

from protocol_v2_manifest_matrix_full_v1 import center_crop_box


class FullManifestMatrixTests(unittest.TestCase):
    def test_center_crop_geometry_matches_generation_rounding(self) -> None:
        self.assertEqual(center_crop_box(1024, 1024, 0.10), [102, 102, 922, 922])
        self.assertEqual(center_crop_box(1024, 1024, 0.20), [205, 205, 819, 819])


if __name__ == "__main__":
    unittest.main()
