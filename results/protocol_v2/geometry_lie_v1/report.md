# Protocol v2 declared-geometry stress test

Cases: 14; rows: 84; signed manifests: 78; not applicable: 6; pixel-cache hits: 84; GPU inference calls: 0.

This is a deliberate protocol stress test, not a statistical estimate.
Correct declarations reproduce 14/14 baseline verdicts. Missing or invalid geometry is rejected in 26/26 cases that reach the geometry gate.
The defense-aware adaptive condition creates 0 new silent-other verdicts and changes 4 policy verdicts.
The adaptive condition maximizes the regional logit among false equal-size boxes on the configured grid and is reported as an upper-bound attack.
Equal-size shifted and adaptive declarations are not applicable to three full-frame controls.
A cut-chain case is retained as a negative control: the lineage gate precedes geometry.
