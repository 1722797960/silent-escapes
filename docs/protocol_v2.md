# Protocol v2 freeze

Status: frozen for the SDK compatibility spike. This document describes the
protocol that later experiments must implement. It does not change any paper
number or historical result.

## 1. Measurement claim

The main experiment studies a fully AI-generated image whose WAM signal is
localized to a SAM subject region. A selectively disclosing signing workflow
issues a signed region assertion but omits the AI-generation disclosure. A
downstream editor can crop or otherwise edit and re-sign the asset. The audit
asks whether validated provenance semantics remain consistent with the signed
regional watermark evidence.

This is not yet an experiment on a real photograph with only a local
generative edit. Removing the complete AI-touched region from such a photograph
would require a different semantic interpretation and is out of scope for the
main protocol.

## 2. Actors and trust boundaries

1. The claim generator creates the fully AI-generated source asset.
2. The watermark embedder embeds a 32-bit WAM codeword only in the SAM region.
3. The signing workflow signs the region assertion while selectively omitting
   the AI disclosure from the active provenance claim.
4. A downstream editor crops or otherwise edits the asset and re-signs it. The
   edit can be accidental or deliberate.
5. The consumer validates the C2PA chain, resolves the signed reference,
   evaluates regional evidence, and applies a provenance-semantics policy.

A valid signature proves integrity and signer attribution. It does not prove
that the signer's semantic assertions are truthful. Research credentials are
self-signed and do not model production credential trust or credential theft.

## 3. Region assertion v2

The assertion label remains `com.example.region_assertion`; the data gains an
explicit `schema_version` of `2`.

Required fields:

- `schema_version`: string `"2"`.
- `alg`: `wam_mit_regional` for the current experiment.
- `reference_bits`: exactly 32 ASCII characters in `{0,1}`.
- `reference_length`: decimal string `"32"`.
- `required_matches`: decimal string `"24"`.
- `payload_hash`: lowercase SHA-256 of the ASCII `reference_bits`. This is an
  integrity consistency digest and identifier, not a hiding commitment.
- `bbox_csv`: normalized `x0,y0,x1,y1` string.
- `bitmap64`: base64 encoding of the packed 64 by 64 signed mask.

The verifier must obtain `reference_bits` from a validated current or ancestor
manifest. It must not read `embed_meta.json`, attack-generation metadata, or
any other experiment-side oracle. Evaluation-only rows may use an oracle, but
must be labelled `oracle-evaluation-only` and must never be presented as a
deployable verdict.

The WAM decision is integer based: at least 24 matching bits out of 32. The
PixelSeal control uses at least 192 matching bits out of 256 when reporting the
legacy 0.75 threshold. Matched-FPR thresholds are calibrated and reported
separately.

## 4. Crop transform schema

A compliant editor records `c2pa.opened` and `c2pa.cropped` in
`c2pa.actions.v2`. The opened action references a `parentOf` ingredient. The
cropped action stores a namespaced transform object under
`parameters.com.example.crop_transform`.

The transform contains string values only for c2pa-python 0.37.10
compatibility:

- `schema_version`: `"1"`
- `coordinate_system`: `source-pixel-half-open`
- `source_size_px`: `width,height`
- `crop_box_px`: `x0,y0,x1,y1`
- `output_size_px`: `width,height`
- `resize_filter`: `LANCZOS` or `NONE`

Coordinates are integer source-pixel coordinates with an inclusive lower bound
and exclusive upper bound. The transform is accepted only after parsing,
bounds, positive-area, and output-size checks. A declaration does not make the
geometry truthful; targeted geometry lies are a separate adversarial condition.

If the SDK cannot round-trip the nested namespaced object, the first fallback
is a CSV-only flat namespaced parameter set. The second fallback is a signed
`com.example.crop_transform` assertion. An SDK upgrade is considered only
after both representations are tested.

## 5. Ordered audit gates

The verifier returns independent structured fields rather than collapsing all
failures into one label.

1. **Lineage gate**: `current`, `ancestor`, `discontinuity`, `fresh-root`, or
   `no-manifest`.
2. **Reference gate**: `signed-current`, `signed-ancestor`,
   `oracle-evaluation-only`, or `no-reference`.
3. **Evidence gate**: `message-detected`, `region-removed`,
   `signal-suppressed`, `silent-other`, or `no-reference`.
4. **Semantics gate**: `ai-disclosed-in-chain`, `no-ai-disclosure-in-chain`,
   or `unknown`.
5. **Policy verdict**: `consistent-synthetic`, `integrity-clash`,
   `review-evidence-removal`, `review-signal-suppression`, `discontinuity`,
   `no-reference`, or `silent-other`.

AI disclosure is searched across the complete validated parent chain, not only
the active manifest. A cut chain fails the lineage gate before evidence policy
is applied. `SILENT_OTHER` and `NO_REFERENCE` remain explicit outputs.

## 6. Cache boundary

The pixel cache is keyed by SHA-256 over decoded RGB bytes plus dimensions. It
stores decoded message bits, two-dimensional mask logits, model/checkpoint
hashes, and preprocessing version. Logits are compressed as float16 and may
vary within a documented numerical tolerance; decoded bits must match exactly.
The integer match count belongs to the manifest/audit layer because it depends
on the signed reference, not on pixels alone.

The manifest/audit cache is separate. It stores the manifest hash, validated
lineage, resolved assertion and reference source, parsed crop transform,
chain-wide disclosure state, evidence status, and policy verdict. Changing a
manifest schema must not trigger GPU inference when decoded pixels and model
inputs are unchanged.

## 7. Security boundaries and known risks

- A fresh root that claims `digitalCapture` without an edit action can evade a
  continuity rule unless deployment policy requires trusted capture
  credentials or external soft-binding lookup.
- A malicious editor can lie about crop geometry. The planned test includes a
  defense-adaptive lie that maps the signed region to a high-logit location;
  this is reported as an upper-bound attack, not as the same information model
  as mask-aware cropping.
- The regional audit currently requires a detector with per-pixel localization
  output and is validated only with WAM.
- A missing signed reference is a protocol boundary. It is not silently
  replaced by experiment metadata.
- The signing helper treats request JSON as untrusted file input. Every source
  path must resolve below an explicit `--input-root`, and every generated
  request, parent, and child asset must resolve below `--output-root`. Existing
  output files are never overwritten.
- The 32-bit digest does not hide the message because the input space is only
  2^32. Publishing `reference_bits` therefore does not materially weaken a
  hiding property that the old assertion did not provide.

## 8. Spike acceptance criteria

The SDK spike passes only if all of the following are observed:

- `c2pa.opened` and `c2pa.cropped` survive signing and reading.
- the `parentOf` ingredient and ancestor assertion are available and validated;
- the namespaced transform round-trips without empty numeric arrays;
- `reference_bits` round-trips exactly and its digest is consistent;
- decoded RGB pixels are unchanged by signing;
- missing and false geometry are distinguishable from valid geometry;
- the parser uses only C2PA data from the tested asset; and
- a separate reader process agrees with the direct SDK inspection.

The spike proves serialization compatibility, not security against false
geometry and not interoperability with every C2PA implementation.
