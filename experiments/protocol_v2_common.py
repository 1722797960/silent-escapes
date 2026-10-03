"""Pure protocol-v2 lineage, evidence, and policy helpers.

The functions in this module do not import C2PA, Torch, or experiment metadata.
They operate only on a validated manifest-store representation and explicit
detector measurements supplied by the caller.
"""
from __future__ import annotations

import re
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Sequence


REGION_LABEL = "com.example.region_assertion"
INGREDIENT_PREFIX = "c2pa.ingredient"
AI_SOURCE_MARKERS = (
    "trainedAlgorithmicMedia",
    "compositeWithTrainedAlgorithmicMedia",
    "compositedWithTrainedAlgorithmicMedia",
)
EDIT_ACTIONS = {"c2pa.opened", "c2pa.edited", "c2pa.cropped"}


def confined_path(
    raw_value: str | Path,
    roots: Sequence[str | Path],
    *,
    field: str,
    must_exist: bool,
    require_file: bool = False,
) -> Path:
    """Resolve a request path and require it to stay inside an allowed root."""
    allowed = tuple(Path(root).expanduser().resolve(strict=True) for root in roots)
    if not allowed:
        raise ValueError(f"{field} has no allowed roots")
    candidate = Path(raw_value).expanduser().resolve(strict=must_exist)
    if not any(candidate.is_relative_to(root) for root in allowed):
        raise ValueError(f"{field} is outside the allowed roots: {candidate}")
    if require_file and not candidate.is_file():
        raise FileNotFoundError(f"{field} is not a file: {candidate}")
    return candidate


def manifest_label_from_url(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"/c2pa/([^/]+)(?:/|$)", url)
    return match.group(1) if match else None


def parent_manifest_labels(manifest: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    store = manifest.get("assertion_store", {})
    if not isinstance(store, dict):
        return labels
    for label, assertion in store.items():
        if not str(label).startswith(INGREDIENT_PREFIX) or not isinstance(assertion, dict):
            continue
        if assertion.get("relationship") != "parentOf":
            continue
        parent = manifest_label_from_url((assertion.get("activeManifest") or {}).get("url"))
        if parent and parent not in labels:
            labels.append(parent)
    return labels


def action_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    store = manifest.get("assertion_store", {})
    actions = store.get("c2pa.actions.v2") or store.get("c2pa.actions") or {}
    values = actions.get("actions", []) if isinstance(actions, dict) else []
    return [item for item in values if isinstance(item, dict)]


def has_edit_action(manifest: dict[str, Any]) -> bool:
    return any(item.get("action") in EDIT_ACTIONS for item in action_items(manifest))


def parent_validation_succeeded(detailed: dict[str, Any]) -> bool:
    results = detailed.get("validation_results", {})
    for delta in results.get("ingredientDeltas", []):
        validations = delta.get("validationDeltas", {}) if isinstance(delta, dict) else {}
        for item in validations.get("success", []):
            if isinstance(item, dict) and item.get("code") == "ingredient.manifest.validated":
                return True
    return False


def reachable_manifests(detailed: dict[str, Any]) -> dict[str, Any]:
    """Return the active manifest's validated parentOf closure.

    The C2PA Reader must already have returned an overall valid state. The
    `parent_validation_succeeded` result is kept as a separate invariant for an
    ancestor reference.
    """
    manifests = detailed.get("manifests", {})
    active = detailed.get("active_manifest")
    if not active or active not in manifests:
        return {"active": None, "nodes": [], "depth": {}, "errors": ["active manifest missing"]}
    queue: deque[tuple[str, int]] = deque([(active, 0)])
    depth: dict[str, int] = {}
    errors: list[str] = []
    while queue:
        label, current_depth = queue.popleft()
        if label in depth:
            if current_depth < depth[label]:
                depth[label] = current_depth
            continue
        if label not in manifests:
            errors.append(f"referenced parent manifest missing: {label}")
            continue
        depth[label] = current_depth
        for parent in parent_manifest_labels(manifests[label]):
            queue.append((parent, current_depth + 1))
    ordered = sorted(depth, key=lambda item: (depth[item], item))
    return {"active": active, "nodes": ordered, "depth": depth, "errors": errors}


def resolve_region_reference(
    detailed: dict[str, Any], *, manifest_valid: bool
) -> dict[str, Any]:
    graph = reachable_manifests(detailed)
    if not manifest_valid or graph["active"] is None:
        return {
            "state": "no-reference",
            "manifest": None,
            "depth": None,
            "region": None,
            "errors": ["manifest is not valid", *graph["errors"]],
        }
    manifests = detailed.get("manifests", {})
    hits = []
    for label in graph["nodes"]:
        region = manifests[label].get("assertion_store", {}).get(REGION_LABEL)
        if region is not None:
            hits.append((graph["depth"][label], label, region))
    if not hits:
        return {
            "state": "no-reference",
            "manifest": None,
            "depth": None,
            "region": None,
            "errors": graph["errors"],
        }
    nearest_depth = min(item[0] for item in hits)
    nearest = [item for item in hits if item[0] == nearest_depth]
    if len(nearest) != 1:
        return {
            "state": "no-reference",
            "manifest": None,
            "depth": None,
            "region": None,
            "errors": ["ambiguous region assertions at the same lineage depth", *graph["errors"]],
        }
    depth, label, region = nearest[0]
    if depth > 0 and not parent_validation_succeeded(detailed):
        return {
            "state": "no-reference",
            "manifest": None,
            "depth": None,
            "region": None,
            "errors": ["ancestor manifest was not reported as validated", *graph["errors"]],
        }
    return {
        "state": "signed-current" if depth == 0 else "signed-ancestor",
        "manifest": label,
        "depth": depth,
        "region": region,
        "errors": graph["errors"],
    }


def lineage_state(
    detailed: dict[str, Any], *, manifest_valid: bool, reference_state: str
) -> str:
    graph = reachable_manifests(detailed)
    active = graph["active"]
    if not manifest_valid or active is None:
        return "no-manifest"
    if reference_state == "signed-current":
        return "current"
    if reference_state == "signed-ancestor":
        return "ancestor"
    manifest = detailed["manifests"][active]
    if has_edit_action(manifest) and not parent_manifest_labels(manifest):
        return "discontinuity"
    return "fresh-root"


def chain_semantics(detailed: dict[str, Any], *, manifest_valid: bool) -> str:
    if not manifest_valid:
        return "unknown"
    graph = reachable_manifests(detailed)
    if graph["active"] is None or graph["errors"]:
        return "unknown"
    manifests = detailed.get("manifests", {})
    for label in graph["nodes"]:
        for action in action_items(manifests[label]):
            source_type = str(action.get("digitalSourceType", ""))
            if any(marker in source_type for marker in AI_SOURCE_MARKERS):
                return "ai-disclosed-in-chain"
    return "no-ai-disclosure-in-chain"


def normalize_bits(bits: str | Sequence[int | bool]) -> str:
    if isinstance(bits, str):
        normalized = bits
    else:
        normalized = "".join("1" if int(value) else "0" for value in bits)
    if not normalized or set(normalized) - {"0", "1"}:
        raise ValueError("bits must be a non-empty binary sequence")
    return normalized


def matching_bit_count(
    reference: str | Sequence[int | bool], decoded: str | Sequence[int | bool]
) -> int:
    left, right = normalize_bits(reference), normalize_bits(decoded)
    if len(left) != len(right):
        raise ValueError(f"bit-length mismatch: reference={len(left)}, decoded={len(right)}")
    return sum(a == b for a, b in zip(left, right))


def classify_evidence(
    *,
    reference_state: str,
    match_count: int | None,
    required_matches: int,
    region_retention: float | None,
    logit_in_region: float | None,
    retention_floor: float = 0.05,
    suppression_threshold: float = 0.5,
) -> str:
    if reference_state not in {"signed-current", "signed-ancestor"}:
        return "no-reference"
    if match_count is None:
        raise ValueError("match_count is required when a signed reference exists")
    if match_count >= required_matches:
        return "message-detected"
    if region_retention is not None and region_retention < retention_floor:
        return "region-removed"
    if logit_in_region is not None and logit_in_region < suppression_threshold:
        return "signal-suppressed"
    return "silent-other"


def policy_verdict(*, lineage: str, evidence: str, semantics: str) -> str:
    if lineage in {"discontinuity", "no-manifest"}:
        return "discontinuity" if lineage == "discontinuity" else "no-reference"
    if evidence == "no-reference":
        return "no-reference"
    if evidence == "message-detected":
        return "consistent-synthetic" if semantics == "ai-disclosed-in-chain" else "integrity-clash"
    if evidence == "region-removed":
        return "review-evidence-removal"
    if evidence == "signal-suppressed":
        return "review-signal-suppression"
    return "silent-other"


def audit_protocol_v2(
    *,
    lineage: str,
    reference_state: str,
    semantics: str,
    reference_bits: str | Sequence[int | bool] | None,
    decoded_bits: str | Sequence[int | bool] | None,
    required_matches: int = 24,
    region_retention: float | None,
    logit_in_region: float | None,
) -> dict[str, Any]:
    match_count = None
    bit_length = None
    if reference_state in {"signed-current", "signed-ancestor"}:
        if reference_bits is None or decoded_bits is None:
            raise ValueError("signed reference audit requires both reference_bits and decoded_bits")
        normalized_reference = normalize_bits(reference_bits)
        match_count = matching_bit_count(normalized_reference, decoded_bits)
        bit_length = len(normalized_reference)
        if not 0 <= required_matches <= bit_length:
            raise ValueError("required_matches is outside the payload length")
    evidence = classify_evidence(
        reference_state=reference_state,
        match_count=match_count,
        required_matches=required_matches,
        region_retention=region_retention,
        logit_in_region=logit_in_region,
    )
    return {
        "lineage": lineage,
        "reference": reference_state,
        "semantics": semantics,
        "match_count": match_count,
        "bit_length": bit_length,
        "required_matches": required_matches,
        "evidence": evidence,
        "policy_verdict": policy_verdict(lineage=lineage, evidence=evidence, semantics=semantics),
    }


def contains_forbidden_oracle_path(values: Iterable[str]) -> bool:
    forbidden = {"embed_meta.json", "attack_manifest.jsonl"}
    return any(str(value).replace("\\", "/").rsplit("/", 1)[-1] in forbidden for value in values)
