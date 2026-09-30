"""Shared parsing and policy helpers for the C2PA lineage ablation."""
from __future__ import annotations

import re
from collections import deque
from typing import Any


REGION_LABEL = "com.example.region_assertion"
INGREDIENT_PREFIX = "c2pa.ingredient"


def _manifest_label_from_url(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"/c2pa/([^/]+)(?:/|$)", url)
    return match.group(1) if match else None


def parent_manifest_labels(manifest: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    store = manifest.get("assertion_store", {})
    for assertion_label, assertion in store.items():
        if not assertion_label.startswith(INGREDIENT_PREFIX):
            continue
        if assertion.get("relationship") != "parentOf":
            continue
        parent = _manifest_label_from_url(
            (assertion.get("activeManifest") or {}).get("url")
        )
        if parent:
            labels.append(parent)
    return labels


def manifest_has_edit_action(manifest: dict[str, Any]) -> bool:
    store = manifest.get("assertion_store", {})
    actions = store.get("c2pa.actions.v2") or store.get("c2pa.actions") or {}
    return any(
        item.get("action") in {"c2pa.opened", "c2pa.edited"}
        for item in actions.get("actions", [])
    )


def resolve_region_assertion(detailed: dict[str, Any]) -> dict[str, Any]:
    active = detailed.get("active_manifest")
    manifests = detailed.get("manifests", {})
    if not active or active not in manifests:
        return {
            "source": "missing",
            "depth": None,
            "manifest": None,
            "region": None,
            "parent_chain": [],
        }

    queue: deque[tuple[str, int, list[str]]] = deque([(active, 0, [active])])
    visited: set[str] = set()
    while queue:
        label, depth, chain = queue.popleft()
        if label in visited or label not in manifests:
            continue
        visited.add(label)
        manifest = manifests[label]
        region = manifest.get("assertion_store", {}).get(REGION_LABEL)
        if region is not None:
            return {
                "source": "current" if depth == 0 else "ancestor",
                "depth": depth,
                "manifest": label,
                "region": region,
                "parent_chain": chain,
            }
        for parent in parent_manifest_labels(manifest):
            queue.append((parent, depth + 1, [*chain, parent]))

    return {
        "source": "missing",
        "depth": None,
        "manifest": None,
        "region": None,
        "parent_chain": [active],
    }


def strict_continuity_status(detailed: dict[str, Any]) -> str:
    resolved = resolve_region_assertion(detailed)
    if resolved["source"] == "current":
        return "CURRENT_ASSERTION"
    if resolved["source"] == "ancestor":
        return "ANCESTOR_ASSERTION"

    active = detailed.get("active_manifest")
    manifests = detailed.get("manifests", {})
    manifest = manifests.get(active, {})
    if manifest_has_edit_action(manifest) and not parent_manifest_labels(manifest):
        return "PROVENANCE_DISCONTINUITY"
    return "NO_REGION_CLAIM"


def parent_validation_succeeded(detailed: dict[str, Any]) -> bool:
    results = detailed.get("validation_results", {})
    for delta in results.get("ingredientDeltas", []):
        validation = delta.get("validationDeltas", {})
        for item in validation.get("success", []):
            if item.get("code") == "ingredient.manifest.validated":
                return True
    return False
