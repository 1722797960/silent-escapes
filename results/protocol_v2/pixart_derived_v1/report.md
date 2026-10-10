# PixArt v2 derived analysis

All inputs were re-read from valid signed chains and exact pixel caches.
Results assume the editor's declared geometry is consumed (trusted-editor).
No GPU inference or signing was performed. No-edit warnings are not benign FPR.

## Threshold grid

| Retention floor | Logit threshold | Reviewed / escapes | Rate [Wilson 95% CI] | No-edit reviews |
|---:|---:|---:|---:|---:|
| 1% | 0.3 | 808/940 | 85.96% [83.59, 88.03] | 2/200 |
| 1% | 0.5 | 841/940 | 89.47% [87.34, 91.27] | 2/200 |
| 1% | 0.7 | 882/940 | 93.83% [92.11, 95.20] | 2/200 |
| 5% | 0.3 | 844/940 | 89.79% [87.69, 91.56] | 2/200 |
| 5% | 0.5 | 869/940 | 92.45% [90.58, 93.97] | 2/200 |
| 5% | 0.7 | 902/940 | 95.96% [94.50, 97.04] | 2/200 |
| 10% | 0.3 | 866/940 | 92.13% [90.23, 93.68] | 2/200 |
| 10% | 0.5 | 886/940 | 94.26% [92.58, 95.57] | 2/200 |
| 10% | 0.7 | 912/940 | 97.02% [95.73, 97.93] | 2/200 |
| 20% | 0.3 | 888/940 | 94.47% [92.82, 95.76] | 2/200 |
| 20% | 0.5 | 901/940 | 95.85% [94.38, 96.95] | 2/200 |
| 20% | 0.7 | 920/940 | 97.87% [96.74, 98.62] | 2/200 |

## Geometry and signal ablations

| Condition | Reviewed / escapes | Rate | c=0.40 reviewed / escapes |
|---|---:|---:|---:|
| bbox-only | 404/940 | 42.98% | 144/199 |
| bitmap-retention-only | 627/940 | 66.70% | 168/199 |
| bitmap-plus-signal | 869/940 | 92.45% | 197/199 |

BBox-only uses continuous bbox intersection with signed half-open source-pixel
crop coordinates, divided by original bbox area. Bitmap-only uses the v2
floor/ceil mask-cell retention rule, with no signal branch. Their difference
measures geometry-representation gain, not signal-check gain. The difference
between bitmap-only and bitmap-plus-signal isolates the added signal branch.

## Legacy comparison

Old 870/940; v2 869/940. 5 reviews become silent and 4 silent cases become reviews.

| Legacy evidence | V2 evidence | Cases |
|---|---|---:|
| message-detected | message-detected | 860 |
| region-removed | region-removed | 627 |
| region-removed | signal-suppressed | 21 |
| region-removed | silent-other | 5 |
| signal-suppressed | signal-suppressed | 217 |
| silent-other | signal-suppressed | 4 |
| silent-other | silent-other | 66 |

The legacy corrected audit was reconstructed on the server from hash-identical original audit/metadata and the retained CPU correction script. Its LF-normalized hash matches the local historical file; original files remain untouched. Legacy logits are saved rounded values, not new inference. V2 uses signed half-open pixel bounds with floor/ceil mask inclusion and fixed-64 sampling, rather than rounded normalized bounds and variable-window sampling. Per-row ordered input swaps separate decision changes when retention is replaced first; they are not order-independent causal attribution. Intervals are row-level Wilson, not source-cluster-adjusted uncertainty.
