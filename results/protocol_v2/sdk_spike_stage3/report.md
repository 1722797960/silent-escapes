# Protocol v2 SDK spike

Overall: **PASS**

| Condition | Manifest | Reference | Geometry | Ground truth | Pixels | Reader agreement |
|---|---:|---:|---:|---:|---:|---:|
| parent_assertion_v2 | True | signed-current | missing | None | True | True |
| child_correct | True | signed-ancestor | valid | True | True | True |
| child_missing_geometry | True | signed-ancestor | missing | None | True | True |
| child_identity_lie | True | signed-ancestor | valid | False | True | True |
| child_propagated | True | signed-current | valid | True | True | True |

The identity-lie row is deliberately syntactically valid. Only the controlled
harness knows the real crop, so this row documents the trust boundary rather than
claiming that C2PA alone proves edit geometry.
