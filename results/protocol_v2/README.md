# Protocol v2 boundary pilot

This directory contains the public, lightweight outputs of the protocol-v2
boundary pilot. The pilot deliberately selects boundary and control cases; it
is a protocol validation suite, not a statistical estimate of deployment
performance.

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

## Intentionally omitted

Generated signed assets, parent assets, detector logits, partial shard files,
and request files containing machine-local paths are retained in the private
experiment archive but are not published here. They can be regenerated with
the scripts and configuration under `experiments/` and `configs/` when the
required datasets, model checkpoints, and signing credentials are available.

The verifier obtains reference bits only from a validated current or ancestor
manifest. Oracle references in `audit_rows.jsonl` are recorded for evaluation
only and are not inputs to the policy verdict.
