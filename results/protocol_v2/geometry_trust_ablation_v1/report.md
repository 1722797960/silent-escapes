# Protocol v2 geometry editor-trust policy ablation

Input rows: 2400; policy rows: 4800.
GPU inference calls: 0; manifest signing calls: 0.

Trusted editors preserve the raw geometry experiment, including 4/400 adaptive review evasions.
Untrusted valid crop geometry is not consumed: adaptive review evasions become 0/400, with valid rows routed to review-untrusted-geometry.

This is a deployment-policy tradeoff, not cryptographic proof that a trusted editor's transform is truthful. A compromised trusted editor remains inside the measured raw attack boundary.
