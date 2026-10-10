# Protocol v2 SDXL per-side centered-crop main curve

Cases and valid manifests: 1800/1800.
Escapes: 752; trusted-editor evidence reviews: 684 (91.0%).
Trusted-editor silent-other: 68.
The untrusted-editor policy does not consume declared crop geometry; valid crops are routed to review-untrusted-geometry.
GPU inference calls during manifest evaluation: 0.

## Historical comparison

| Quantity | Legacy known-transform audit | V2 signed-geometry audit |
|---|---:|---:|
| Assets | 1,800 | 1,800 |
| Watermark escapes | 752 | 752 |
| Reviewed escapes | 686 | 684 |
| Silent-other | 66 | 68 |
| Reviewed / escapes | 91.2234% | 90.9574% |

The v2 evidence split is 1,048 message detections, 578 removed-region cases,
106 suppressed-signal cases, and 68 silent-other cases. Integer message-match
counts agree with all 1,800 historical rows. Source images, detector checkpoint,
message threshold and regional thresholds were not changed.

The signed pixel crop box is mapped to the 64x64 region mask using floor/ceil
intersecting-cell bounds; legacy code rounded normalized mask bounds. V2 also
samples logits on a fixed 64x64 grid instead of the legacy kept-window grid.
These differences change 24 evidence labels: six reviewed cases become silent,
four silent cases become suppression reviews, and fourteen removed-region cases
become suppression reviews. The net change is two fewer reviews, or 0.2660
percentage points among 752 escapes.

## Per-side crop curve

| c removed from each side | Assets | Escapes | Reviewed escapes | Silent-other |
|---|---:|---:|---:|---:|
| 0.00 | 200 | 2 | 1 | 1 |
| 0.05 | 200 | 12 | 4 | 8 |
| 0.10 | 200 | 32 | 15 | 17 |
| 0.15 | 200 | 51 | 38 | 13 |
| 0.20 | 200 | 75 | 68 | 7 |
| 0.25 | 200 | 99 | 94 | 5 |
| 0.30 | 200 | 119 | 112 | 7 |
| 0.35 | 200 | 163 | 156 | 7 |
| 0.40 | 200 | 199 | 196 | 3 |

The c=0.10 and c=0.20 review numerators remain 15 and 68, respectively; the
honest-crop cost comparisons at those geometries are therefore unchanged.

## Verification and interpretation

Every manifest and source-pixel hash passes server-side validation. The
reference and crop transform are read back from the signed chain. Signing and
auditing make 1,800 exact pixel-cache hits and zero detector calls. Independent
recomputation from cache comparisons agrees with the final evidence transition
matrix; `python tools/verify_main_curve_v2.py` verifies the compact JSON result.
The result archive SHA-256 is
`88125cc6ebf4a2008cbe05a52d749ca3d030af3bc07ff5d25f8f9da3dcaaa12c`.

The 684/752 rate assumes the verifier consumes geometry from an authorized
editor. Under the separate untrusted-editor policy, all 1,600 actual crops are
routed to geometry review by construction; this is a deployment-policy cost,
not a detector-recovery rate. Signature validity does not prove transform
truthfulness. Private shard fields named rho are compatibility aliases for
per-side c in this run; `crop_frac` and `per_crop` are the canonical outputs.

The rejected C1 removed-area run remains archived separately. No manuscript
source or GitHub repository was modified by this experiment stage.
