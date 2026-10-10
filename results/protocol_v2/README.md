# Protocol v2 release artifacts

This directory contains the public, lightweight outputs of the protocol-v2
pilot, full signed-chain migration, declared-geometry stress test, and editor-
trust ablation. Generated pixels and detector outputs are reused through exact
decoded-RGB cache keys; manifest and policy evaluation performs no GPU
inference.

## Included outputs

- `pilot_v1/audit_rows.jsonl`: one verifier result per selected case.
- `pilot_v1/summary.json`: aggregate lineage, evidence, reference, and policy
  states.
- `pilot_v1/environment.json`: detector runtime and checkpoint hashes.
- `pilot_v1/report.md`: short interpretation note.
- `sdk_spike_stage3/`: summary and environment for the C2PA action-schema
  round-trip test.
- `geometry_lie_v1/rows.jsonl`: 84 structured cells from the 14-case declared-
  geometry stress test, including six explicit not-applicable cells.
- `geometry_lie_v1/summary.json`: geometry-gate, policy-transition, cache, and
  adaptive false-box aggregates.
- `geometry_lie_v1/environment.json` and `report.md`: runtime provenance and a
  concise interpretation of the stress test.
- `manifest_matrix_pilot_v2/`: 100 valid rows from the corrected 20-case by
  five-condition manifest-semantics gate.
- `manifest_matrix_full_v2/rows.jsonl`: 8,000 independently signed and audited
  rows spanning eight pixel conditions, 200 assets, and five manifest
  conditions. `summary.json` contains the global acceptance checks.
- `manifest_matrix_full_v2/selected_cases.json` and `parent_index.json`:
  repository-relative pixel selection, geometry, cache keys, and hashes for the
  600 independently validated reusable parents.
- `full_pixel_cache_v1/warm_full_v1/`: 1,600 cache-provenance rows plus the
  final coverage and model/checkpoint-hash acceptance summary. Detector logits
  themselves are private large artifacts and are not included.
- `main_curve_full_v2/full_v1/`: 1,800 SDXL signed-chain audit rows and the
  deployable protocol-v2 summary. The 752 decoder escapes are unchanged from
  the legacy curve; 684 enter evidence review under trusted-editor geometry.
- `sdxl_derived_v1/`: 1,800 derived SDXL rows, the 4-by-3 threshold grid, and
  region/signal ablations. These outputs re-read valid chains and exact caches
  without signing or neural inference.
- `pixart_full_v1/`: 1,800 PixArt signed-chain audit and independent-
  verification rows plus compact run summaries. Of 940 decoder escapes, 869
  enter evidence review under trusted-editor geometry.
- `pixart_derived_v1/`: 1,800 derived PixArt rows, 30 legacy-to-v2 transition
  rows, the threshold grid, and region/signal ablations.
- `c1_full_v1/`: compact acceptance summaries and 1,800 mask-aware rows derived
  from the 10,800 signed C1 base rows. The full signed assets, logits, and large
  intermediate row sets remain in the private archive.
- `geometry_lie_full_v1/`: 2,400 rows covering 400 cases and six declared-
  geometry conditions, including detector-adaptive false boxes.
- `geometry_trust_ablation_v1/`: 4,800 policy rows applying trusted- and
  untrusted-editor gates to the 2,400 geometry-stress rows.

## Intentionally omitted

Generated signed assets, parent assets, detector logits, partial shard files,
and request files containing machine-local paths are retained in the private
experiment archive but are not published here. They can be regenerated with
the scripts and configuration under `experiments/` and `configs/` when the
required datasets, model checkpoints, and signing credentials are available.

The verifier obtains reference bits only from a validated current or ancestor
manifest. Oracle references in `audit_rows.jsonl` are recorded for evaluation
only and are not inputs to the policy verdict.

Signature validity does not prove that declared geometry is truthful. The
trusted-editor results measure the raw geometry boundary; the untrusted-editor
policy refuses to consume otherwise valid crop geometry and routes it to
`review-untrusted-geometry`. This is a deployment-policy tradeoff, not an
automatic cryptographic defense.
