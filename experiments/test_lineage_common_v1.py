"""Unit tests for lineage traversal and fail-closed policy classification."""
from __future__ import annotations

import unittest

from lineage_common_v1 import resolve_region_assertion, strict_continuity_status


REGION = {"payload_hash": "abc", "bitmap64": "AA=="}


def manifest(*, region=None, parent=None, edited=True):
    store = {}
    if region is not None:
        store["com.example.region_assertion"] = region
    if parent is not None:
        store["c2pa.ingredient.v3"] = {
            "relationship": "parentOf",
            "activeManifest": {"url": f"self#jumbf=/c2pa/{parent}"},
        }
    if edited:
        store["c2pa.actions.v2"] = {"actions": [{"action": "c2pa.edited"}]}
    return {"assertion_store": store}


class LineageCommonTests(unittest.TestCase):
    def test_current_assertion(self):
        data = {"active_manifest": "active", "manifests": {
            "active": manifest(region=REGION, parent="parent"),
            "parent": manifest(region=REGION, edited=False),
        }}
        self.assertEqual(resolve_region_assertion(data)["source"], "current")
        self.assertEqual(strict_continuity_status(data), "CURRENT_ASSERTION")

    def test_ancestor_assertion(self):
        data = {"active_manifest": "active", "manifests": {
            "active": manifest(parent="parent"),
            "parent": manifest(region=REGION, edited=False),
        }}
        result = resolve_region_assertion(data)
        self.assertEqual(result["source"], "ancestor")
        self.assertEqual(result["depth"], 1)
        self.assertEqual(strict_continuity_status(data), "ANCESTOR_ASSERTION")

    def test_edited_manifest_without_parent_fails_closed(self):
        data = {"active_manifest": "active", "manifests": {
            "active": manifest(),
        }}
        self.assertEqual(
            strict_continuity_status(data), "PROVENANCE_DISCONTINUITY"
        )

    def test_new_root_without_region_is_not_inferred_as_an_edit(self):
        data = {"active_manifest": "active", "manifests": {
            "active": manifest(edited=False),
        }}
        self.assertEqual(strict_continuity_status(data), "NO_REGION_CLAIM")


if __name__ == "__main__":
    unittest.main()
