# Data and artifact policy

The Git repository contains compact, machine-readable outputs needed to audit
the paper's reported numbers. It does not contain model weights, full image
corpora, signed intermediate assets, or detector caches.

## Included

- Aggregate and per-condition JSON/CSV results used by the paper.
- Regional escape-curve summaries for SDXL and PixArt-alpha.
- Position-aware crop summaries and plots.
- Region-aware defense audit tables and threshold-sensitivity results.

## Excluded

- Approximately 12.3 GB of model weights.
- Approximately 11.6 GB of generated images, masks, and embeddings.
- Approximately 6.7 GB of signed and attacked intermediate outputs.
- Local logs, caches, paths, and signing credentials.

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
