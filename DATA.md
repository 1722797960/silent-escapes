# Data and artifact policy

The Git repository contains compact, machine-readable outputs needed to audit
the paper's reported numbers. It does not contain model weights, full image
corpora, signed intermediate assets, or detector caches.

## Included

- Aggregate and per-condition JSON/CSV results used by the paper.
- Regional escape-curve summaries for SDXL and PixArt-alpha.
- Position-aware crop row-level attack/audit records, summaries, and plots.
- Region-aware defense audit tables and threshold-sensitivity results.
- Protocol-v2 signed-chain row records and summaries for the full SDXL and
  PixArt centered-crop curves.
- Protocol-v2 aggregate and derived records for the 10,800-row position-aware
  C1 grid.
- Full declared-geometry stress-test and editor-trust-ablation rows.

## Excluded

- Approximately 12.3 GB of model weights.
- Approximately 11.6 GB of generated images, masks, and embeddings.
- Approximately 6.7 GB of signed and attacked intermediate outputs.
- Local logs, caches, paths, and signing credentials.

The public protocol-v2 rows contain content hashes, geometry, evidence states,
and policy verdicts, but not signed images, parent assets, private keys, request
files, or cached detector logits. Aggregate summaries retain hashes of the
private/generated inputs so that a full local reproduction can be bound to the
same run without publishing those artifacts.

These files are excluded because they are large or security-sensitive, not
because they affect the reported aggregate results. Dataset generation uses the
fixed commands and seeds documented in `REPRODUCIBILITY.md`.

## Expected local directories

```text
ckpts/
  wam_mit.pth
  params.json
  sam/sam_vit_b_01ec64.pth
  pixelseal_checkpoint.pth
  pixart_*/
data/
  originals_500/
  masks_200/
  pixart_200/
  masks_pixart_200/
results/
  ... committed summaries plus locally generated intermediates
```

Do not commit regenerated credentials, model weights, or intermediate image
trees. The repository `.gitignore` blocks their standard locations.
