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

## Intentionally omitted

Generated signed assets, parent assets, detector caches, partial shard files,
and request files containing machine-local paths are retained in the private
experiment archive but are not published here. They can be regenerated with
the scripts and configuration under `experiments/` and `configs/` when the
required datasets, model checkpoints, and signing credentials are available.

The verifier obtains reference bits only from a validated current or ancestor
manifest. Oracle references in `audit_rows.jsonl` are recorded for evaluation
only and are not inputs to the policy verdict.
