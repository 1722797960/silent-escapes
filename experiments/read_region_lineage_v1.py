"""Resolve a regional-watermark assertion from the active or ancestor manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import c2pa

from lineage_common_v1 import (
    parent_manifest_labels,
    parent_validation_succeeded,
    resolve_region_assertion,
    strict_continuity_status,
)


def inspect(path: Path) -> dict:
    reader = c2pa.Reader(str(path))
    detailed = json.loads(reader.detailed_json())
    resolved = resolve_region_assertion(detailed)
    active_label = detailed.get("active_manifest")
    active = detailed.get("manifests", {}).get(active_label, {})
    return {
        "manifest_valid": reader.get_validation_state() == "Valid",
        "validation_state": reader.get_validation_state(),
        "manifest_count": len(detailed.get("manifests", {})),
        "region_assertion_present": resolved["source"] != "missing",
        "region_assertion_source": resolved["source"],
        "lineage_depth": resolved["depth"],
        "region_manifest": resolved["manifest"],
        "region": resolved["region"],
        "parent_chain": resolved["parent_chain"],
        "active_parent_count": len(parent_manifest_labels(active)),
        "parent_manifest_validated": parent_validation_succeeded(detailed),
        "strict_continuity_status": strict_continuity_status(detailed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("asset")
    args = parser.parse_args()
    print(json.dumps(inspect(Path(args.asset)), ensure_ascii=False))


if __name__ == "__main__":
    main()
