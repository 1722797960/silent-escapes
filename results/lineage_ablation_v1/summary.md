# C2PA lineage ablation v1

All conditions reuse identical crop-40 pixels and saved WAM metrics; only the manifest topology changes.

| Condition | N | Valid manifest | Pixel-identical | Parent validated | Region source / policy status |
|---|---:|---:|---:|---:|---|
| propagated | 200 | 200 | 200 | 200 | CURRENT_ASSERTION: 200 |
| ancestor-only | 200 | 200 | 200 | 200 | ANCESTOR_ASSERTION: 200 |
| cut-chain | 200 | 200 | 200 | 0 | PROVENANCE_DISCONTINUITY: 200 |

`PROVENANCE_DISCONTINUITY` is a strict consumer-policy result for a manifest that declares an edit but provides no `parentOf` link. It is not included in the paper's region-evidence flag rate.
