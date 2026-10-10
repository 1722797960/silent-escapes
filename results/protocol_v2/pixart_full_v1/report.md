# PixArt verifier-v2 full-migration report

The cache, signed-chain audit, and independent verification completed on
2026-10-06. The cohort contains 200 PixArt-alpha source images, nine per-side
crop levels, and 1,800 unique pixel samples. It is separate from the C1
removed-area cohort.

## Full-run results

- The cache stage performed 1,780 new GPU inferences and reused 20 accepted
  pilot entries. Every integer message-match count agrees with the frozen
  legacy record.
- All 1,800 signed manifests validate. Manifest auditing and independent
  verification perform no new GPU inference.
- Among 940 decoder escapes, 869 enter evidence review under trusted-editor
  geometry (92.4468%) and 71 remain `silent-other`.
- Evidence states are 860 `message-detected`, 627 `region-removed`, 242
  `signal-suppressed`, and 71 `silent-other`.
- Under the untrusted-editor policy, all 1,600 actual crops are routed to
  `review-untrusted-geometry`. The 200 no-edit controls produce 196
  `integrity-clash`, two `review-signal-suppression`, and two `silent-other`
  verdicts. The 1,600 reviews are a policy cost of refusing untrusted declared
  geometry, not pixel-level detection of 1,600 attacks.

| Per-side crop `c` | Assets | Decoder escapes | Trusted-editor evidence reviews |
|---:|---:|---:|---:|
| 0.00 | 200 | 4 | 2 |
| 0.05 | 200 | 39 | 30 |
| 0.10 | 200 | 66 | 54 |
| 0.15 | 200 | 78 | 71 |
| 0.20 | 200 | 98 | 86 |
| 0.25 | 200 | 128 | 117 |
| 0.30 | 200 | 151 | 144 |
| 0.35 | 200 | 177 | 168 |
| 0.40 | 200 | 199 | 197 |
| Total | 1,800 | 940 | 869 |

These proportions describe repeated crops of 200 source images, not 1,800
independent sources. No-edit reviews are not a benign-edit false-positive rate.
Signature validity also does not prove that declared geometry is truthful.

## Independent verification and environment

The independent CPU stage reopens all 1,800 signed assets and recomputes chain
inputs, exact-cache bindings, message bits, and regional evidence. Every row
agrees with the signed audit. The maximum elementwise difference between the
original float32 logits and their float16 cache representation is
0.000244140625; the maximum regional-mean difference is 0.0001048445702. No
evidence class or policy verdict changes because of quantization.

The GPU cache stage used PyTorch 2.5.1+cu124 on one assigned GPU and took
1,015.92 s. The CPU audit used PyTorch 2.6.0+cu124,
`c2pa-python==0.37.10`, NumPy 2.2.6, and Pillow 12.3.0 and took 2,052.47 s.
These wall-clock values describe a shared instance and are not controlled
performance benchmarks.

## Legacy comparison and traceability

The protocol-v2 review count is 869 rather than the legacy known-transform
count of 870. The 940 integer decoder escapes are unchanged. Protocol v2 maps
signed source-pixel geometry to a fixed detector grid, so the one-count net
difference must not be interpreted as a new attack or a threshold change.

The released row hashes are:

- `audit_rows.jsonl`:
  `c492aa4a657189e34a48572abf31834198fc342ce12b60c0603a497709dfa5a0`
- `verification_rows.jsonl`:
  `e7e481d4238544a2a7d6628f788cfa8c56c1de8f716496d35dbde58f4eb4c258`

The compact public release omits signed PNGs, private keys, float32 detector
arrays, and the pixel cache. The committed JSON/JSONL rows are unchanged from
the accepted server run.
