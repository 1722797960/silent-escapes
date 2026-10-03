"""Unit tests for protocol-v2 gates and integer bit decisions."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from protocol_v2_common import (
    REGION_LABEL,
    audit_protocol_v2,
    chain_semantics,
    confined_path,
    contains_forbidden_oracle_path,
    lineage_state,
    matching_bit_count,
    resolve_region_reference,
)


class ConfinedPathTests(unittest.TestCase):
    def test_accepts_file_below_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "inputs" / "asset.png"
            source.parent.mkdir()
            source.write_bytes(b"png")
            self.assertEqual(
                confined_path(source, [root], field="source", must_exist=True, require_file=True),
                source.resolve(),
            )

    def test_rejects_path_outside_allowed_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            allowed = base / "allowed"
            allowed.mkdir()
            outside = base / "outside.png"
            outside.write_bytes(b"png")
            with self.assertRaises(ValueError):
                confined_path(
                    outside, [allowed], field="source", must_exist=True, require_file=True
                )


def manifest(*, region: dict | None = None, parent: str | None = None,
             ai: bool = False, edited: bool = False) -> dict:
    actions = []
    if ai:
        actions.append({
            "action": "c2pa.created",
            "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
        })
    if edited:
        actions.append({"action": "c2pa.edited"})
    store = {"c2pa.actions.v2": {"actions": actions}}
    if region is not None:
        store[REGION_LABEL] = region
    if parent:
        store["c2pa.ingredient.v3"] = {
            "relationship": "parentOf",
            "activeManifest": {"url": f"self#jumbf=/c2pa/{parent}"},
        }
    return {"assertion_store": store}


def detailed(active: str, manifests: dict, *, parent_validated: bool = False) -> dict:
    result = {"active_manifest": active, "manifests": manifests, "validation_results": {}}
    if parent_validated:
        result["validation_results"] = {
            "ingredientDeltas": [{
                "validationDeltas": {"success": [{"code": "ingredient.manifest.validated"}]}
            }]
        }
    return result


class BitDecisionTests(unittest.TestCase):
    def test_integer_boundary(self) -> None:
        reference = "0" * 32
        self.assertEqual(matching_bit_count(reference, "0" * 24 + "1" * 8), 24)
        accepted = audit_protocol_v2(
            lineage="current", reference_state="signed-current",
            semantics="no-ai-disclosure-in-chain", reference_bits=reference,
            decoded_bits="0" * 24 + "1" * 8, region_retention=1.0,
            logit_in_region=0.9,
        )
        rejected = audit_protocol_v2(
            lineage="current", reference_state="signed-current",
            semantics="no-ai-disclosure-in-chain", reference_bits=reference,
            decoded_bits="0" * 23 + "1" * 9, region_retention=1.0,
            logit_in_region=0.9,
        )
        self.assertEqual(accepted["evidence"], "message-detected")
        self.assertEqual(rejected["evidence"], "silent-other")

    def test_silent_other_is_preserved(self) -> None:
        result = audit_protocol_v2(
            lineage="ancestor", reference_state="signed-ancestor",
            semantics="no-ai-disclosure-in-chain", reference_bits="0" * 32,
            decoded_bits="0" * 20 + "1" * 12, region_retention=0.8,
            logit_in_region=0.7,
        )
        self.assertEqual(result["evidence"], "silent-other")
        self.assertEqual(result["policy_verdict"], "silent-other")


class LineageTests(unittest.TestCase):
    def test_current_reference(self) -> None:
        data = detailed("child", {"child": manifest(region={"reference_bits": "0" * 32})})
        resolved = resolve_region_reference(data, manifest_valid=True)
        self.assertEqual(resolved["state"], "signed-current")
        self.assertEqual(lineage_state(data, manifest_valid=True, reference_state=resolved["state"]), "current")

    def test_ancestor_reference_requires_validated_parent(self) -> None:
        manifests = {
            "child": manifest(parent="parent", edited=True),
            "parent": manifest(region={"reference_bits": "0" * 32}, ai=True),
        }
        invalid = detailed("child", manifests)
        valid = detailed("child", manifests, parent_validated=True)
        self.assertEqual(resolve_region_reference(invalid, manifest_valid=True)["state"], "no-reference")
        resolved = resolve_region_reference(valid, manifest_valid=True)
        self.assertEqual(resolved["state"], "signed-ancestor")
        self.assertEqual(chain_semantics(valid, manifest_valid=True), "ai-disclosed-in-chain")

    def test_cut_chain_fails_before_reference(self) -> None:
        data = detailed("child", {"child": manifest(edited=True)})
        resolved = resolve_region_reference(data, manifest_valid=True)
        lineage = lineage_state(data, manifest_valid=True, reference_state=resolved["state"])
        self.assertEqual(lineage, "discontinuity")
        result = audit_protocol_v2(
            lineage=lineage, reference_state="no-reference", semantics="no-ai-disclosure-in-chain",
            reference_bits=None, decoded_bits=None, region_retention=None, logit_in_region=None,
        )
        self.assertEqual(result["policy_verdict"], "discontinuity")

    def test_oracle_paths_are_explicitly_forbidden(self) -> None:
        self.assertTrue(contains_forbidden_oracle_path(["results/embed_meta.json"]))
        self.assertTrue(contains_forbidden_oracle_path(["attack_manifest.jsonl"]))
        self.assertFalse(contains_forbidden_oracle_path(["asset.png", "manifest.json"]))


if __name__ == "__main__":
    unittest.main()
