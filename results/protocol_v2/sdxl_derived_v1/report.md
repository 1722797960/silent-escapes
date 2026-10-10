# SDXL v2 derived analysis

All inputs were re-read from valid signed chains and exact pixel caches.
Results assume the editor's declared geometry is consumed (trusted-editor).
No GPU inference or signing was performed. No-edit warnings are not benign FPR.

## Threshold grid

| Retention floor | Logit threshold | Reviewed / escapes | Rate [Wilson 95% CI] | No-edit reviews |
|---:|---:|---:|---:|---:|
| 1% | 0.3 | 618/752 | 82.18% [79.28, 84.75] | 1/200 |
| 1% | 0.5 | 650/752 | 86.44% [83.80, 88.70] | 1/200 |
| 1% | 0.7 | 685/752 | 91.09% [88.84, 92.92] | 1/200 |
| 5% | 0.3 | 660/752 | 87.77% [85.23, 89.92] | 1/200 |
| 5% | 0.5 | 684/752 | 90.96% [88.69, 92.80] | 1/200 |
| 5% | 0.7 | 711/752 | 94.55% [92.69, 95.96] | 1/200 |
| 10% | 0.3 | 679/752 | 90.29% [87.97, 92.21] | 1/200 |
| 10% | 0.5 | 697/752 | 92.69% [90.60, 94.34] | 1/200 |
| 10% | 0.7 | 716/752 | 95.21% [93.44, 96.52] | 1/200 |
| 20% | 0.3 | 706/752 | 93.88% [91.94, 95.38] | 1/200 |
| 20% | 0.5 | 718/752 | 95.48% [93.75, 96.75] | 1/200 |
| 20% | 0.7 | 727/752 | 96.68% [95.14, 97.74] | 1/200 |

## Geometry and signal ablations

| Condition | Reviewed / escapes | Rate | c=0.40 reviewed / escapes |
|---|---:|---:|---:|
| bbox-only | 383/752 | 50.93% | 134/199 |
| bitmap-retention-only | 578/752 | 76.86% | 153/199 |
| bitmap-plus-signal | 684/752 | 90.96% | 196/199 |

BBox-only uses continuous bbox intersection with signed half-open source-pixel
crop coordinates, divided by original bbox area. Bitmap-only uses the v2
floor/ceil mask-cell retention rule, with no signal branch. Their difference
measures geometry-representation gain, not signal-check gain. The difference
between bitmap-only and bitmap-plus-signal isolates the added signal branch.
