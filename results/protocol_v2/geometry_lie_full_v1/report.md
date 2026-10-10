# Protocol v2 full declared-geometry stress test

Cases: 400; rows: 2400; valid manifests and exact cache hits: 2400/2400.
Correct declarations reproduce 400/400 frozen manifest-matrix baselines. GPU inference calls: 0.

The adaptive condition is defense-aware: it uses cached detector logits to select a false equal-size box. It is an upper-bound stress test, not the paper's primary attacker.

## centered_crop_20pct_per_side

Adaptive policy changes: 59/200; new silent-other: 3/200; review-to-silent-other: 3/200.

## mask_aware_crop_rho_0.4

Adaptive policy changes: 99/200; new silent-other: 1/200; review-to-silent-other: 1/200.
