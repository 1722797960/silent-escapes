# Experiment Design Notes

## Paired WAM spatial-support control (`granularity_control_v2`)

### Measurement goal

This experiment asks whether the regional escape curve is caused by generic
WAM crop fragility or by the spatial support of the signal. It reuses the same
200 images, image-specific 32-bit messages, and candidate WAM perturbations
across four support conditions: the original SAM subject region, a compact
equal-area region at a deterministic random location, native full frame, and a
full-frame perturbation scaled per image to match the subject-region total MSE
after uint8 quantization.

### Controls and invariants

- Every marked condition contains 200 images and the same nine centered
  removed-area levels, `rho = 0.0, 0.1, ..., 0.8`.
- Detection uses the integer boundary `matched_bits >= 24` rather than a
  floating-point comparison with 0.75.
- The compact control has exactly the same foreground-pixel count as its paired
  SAM mask, is contiguous and near-square, is placed without wraparound, and
  uses seed `20260928`.
- The energy-matched full-frame condition must match the subject-region MSE to
  within 0.5% per image after 8-bit quantization; the observed maximum relative
  error is 0.349%.
- Subject-region detections are imported from the frozen C1 center-crop cache
  and verified against all 200 marked-image SHA-256 values.
- A separate 200-image clean-reference condition estimates detector FPR and is
  not mixed into marked-image robustness rates.

### Interpretation boundary

Native full-frame WAM nearly eliminates escapes under this crop grid, while the
equal-PSNR full-frame control reduces severe-crop escape without eliminating it.
The compact equal-area control changes shape and compactness as well as
location, so it is a same-area spatial-support geometry control, not a pure
position ablation. The experiment quantifies the cost of localization and
support geometry; it does not claim that every region shape or every crop
severity behaves identically.

## LaMa exact-mask inpainting (`inpainting_e2e_v1`)

### Threat model

The editor knows the signed 64x64 region bitmap and has access to the original
SAM mask used for the controlled experiment. It removes the localized WAM
signal by inpainting exactly that mask, then re-signs the unchanged-size image
while carrying the original region assertion. The experiment does not model
credential theft: all credentials are self-signed research keys.

### Security decisions

- The primary experiment applies **zero mask dilation**. This makes the attack
  region identical to the signed semantic region and avoids silently enlarging
  the perturbation.
- Pixels outside the binary mask are restored from the source uint8 array and
  checked after PNG re-open. This isolates the intervention to the declared
  region.
- Existing crop results and scripts are never overwritten. New outputs live in
  `results/inpainting_e2e_v1/` and refuse overwrite unless explicitly enabled.
- Subprocesses use argument arrays with `shell=False`. The demo private key is
  passed by path and is never copied into result metadata or logs.
- Geometry is unchanged, so retained/original mask area is exactly 1.0. The
  experiment therefore isolates the regional signal-logit branch of the audit.

### Trust boundaries

- LaMa source and checkpoint are external research dependencies. Runs record
  the source commit and checkpoint SHA-256.
- C2PA validity, assertion presence, and payload-hash binding are rechecked on
  every output.
- WAM and C2PA remain research prototypes and use a self-signed credential.

### Known risks and limitations

- Exact-mask inpainting may leave boundary signal outside the binary mask. A
  dilated-mask sensitivity analysis is intentionally deferred and must be
  reported as a separate condition if added.
- LaMa can introduce visible semantic changes inside large SAM masks. Contact
  sheets and mask coverage are retained for visual audit.
- The pilot uses ten deterministic coverage quantiles. It validates the
  pipeline but is not an estimator for the full 200-image attack rate.

## Benign edit controls (`benign_controls_v1`)

### Measurement goal

This experiment separates two costs that should not be conflated. The **benign
region false-flag rate** is the fraction of honestly described, watermarked
benign edits that receive `TAMPERED_REGION` or `WATERMARK_SUPPRESSED`. The
**clean detector false-positive rate** is measured separately on paired,
unwatermarked images with an honest AI manifest and no regional-watermark
assertion.

### Conditions and invariants

- `identity_resave`: PNG decode/encode and honest re-signing control.
- `jpeg_q90`: JPEG quality 90, 4:4:4 round trip, then PNG and honest re-signing.
- `resize_90`: LANCZOS downscale to 90% and restore to 1024 square pixels.
- `safe_crop_1pct`: remove 1% from the side with maximum signed-mask retention,
  chosen from geometry alone, then resize to the original dimensions.
- `clean_negative`: paired generated original with no embedded watermark and no
  regional claim.

All watermarked edits carry the original payload hash and signed 64x64 region
bitmap in a manifest that truthfully records both AI generation and subsequent
editing. The crop policy never reads detector outputs. Manifest validity,
assertion presence, and payload binding are treated as pipeline invariants, not
as statistical outcomes.

### Interpretation boundary

The released run contains 200 assets per condition and reports Wilson intervals
for every proportion. Strict rates retain all sources; baseline-qualified rates
exclude a source only when its no-edit reference audit already fails. These
edits are intentionally mild and do not estimate false flags for arbitrary user
editing or independently calibrate deployment thresholds.

## Protocol-v2 signed references and manifest semantics

### Verification order

The protocol separates four decisions that must not be collapsed into one
verdict. The verifier first checks lineage continuity and reachable-manifest
action structure, then resolves a signed 32-bit reference from the current or
nearest validated ancestor manifest. It next classifies regional evidence using
an integer match boundary (`>=24` of 32 bits), retained signed-mask area, and
regional detector logits. Only after those gates does it apply disclosure policy
to the verified manifest chain. `silent-other` remains an explicit evidence
state rather than being silently accepted.

The reachable chain receives an additional structural check because the C2PA
SDK can report an ingredient hash match without surfacing a malformed action in
the embedded parent. A non-empty de-novo `c2pa.created` action must carry a
`digitalSourceType`; a `c2pa.opened` action must reference a `parentOf`
ingredient. A failure produces `invalid-chain` and no usable signed reference.

### Controlled manifest conditions

The full matrix keeps decoded pixels fixed and signs each of 1,600 assets under
five conditions: AI disclosure on both current and parent manifests; AI
disclosure only in the ancestor while the editing child omits a source type;
the issuer's misleading `digitalCreation` claim; its misleading
`digitalCapture` claim; and a fresh-root edit that cuts the parent chain. The
child omission is standards-valid because `c2pa.opened` does not require a
`digitalSourceType`. A root `c2pa.created` omission is not used: C2PA 2.3 and the
tested SDK reject it as malformed.

The two misleading source types are policy inputs, not ground truth inferred by
the verifier. `digitalCreation` denotes human creation with non-generative
digital tools, while `digitalCapture` denotes camera-like capture. Both omit the
true AI origin in this controlled experiment. They are kept separate to avoid
equating absence of an AI marker with one particular false provenance story.

### Cache and reproducibility boundary

Detector output is cached by a SHA-256 key over decoded RGB dimensions, mode,
and bytes. Every entry binds the exact WAM params hash, checkpoint hash,
preprocessing version, decoded bits, and two-dimensional float16 logits.
Manifest signing and policy evaluation are a separate layer and are forbidden
from invoking the detector. The completed cache contains 1,600 unique decoded
pixel arrays from eight 200-image conditions; the semantics matrix makes 8,000
exact cache hits and zero GPU calls.

The public release includes the cache provenance rows and hashes but excludes
the logits, signed assets, parents, requests containing machine-local paths,
and model weights. Consequently, the compact release can verify experimental
coverage and reported counts, while reproducing per-pixel detector output still
requires the checkpoint and source assets.

## Change history

### 2026-10-05 - Add paired WAM spatial-support control

**Change:** Added four paired support conditions across nine removed-area
levels, a 200-image clean-reference check, per-image energy matching, summary
plots, and machine-checkable verification gates.

**Reason:** Separate the granularity effect from generic WAM crop fragility and
from the lower total distortion of a localized embedding.

**Impact:** At `rho=0.80`, escape is 55.0% for SAM subject support, 18.5% for
compact equal-area support, 0.5% for native full frame, and 27.0% for the
equal-PSNR full-frame control. The control strengthens the causal interpretation
while preserving the limitation that compact equal-area support changes shape
as well as location.

### 2026-10-04 - Freeze protocol-v2 semantics and full matrix

**Change:** Added signed reference bits, integer match decisions, exact decoded-
pixel caching, reachable-chain action validation, a 20-case boundary matrix,
and a full 1,600-case by five-condition manifest-semantics evaluation.

**Reason:** A payload hash alone cannot support partial bit matching, detector
re-execution should not be coupled to manifest schema changes, and an ingredient
hash match is insufficient evidence that an embedded ancestor's action schema
is valid.

**Impact:** The full run contains 8,000 valid manifests and preserves regional
evidence across all four parent-preserving semantic conditions in 1,600/1,600
cases. Changing disclosure semantics changes the policy label for detected
messages but not the detector evidence. Cutting lineage removes the signed
reference in 1,600/1,600 cases.

**Decision basis:** The original `c2pa.created`-without-source condition was
retained only as an archived malformed preflight. The valid matrix replaces it
with the IPTC `digitalCreation` source type and independently verifies every
prepared parent before child signing.

### 2026-09-30 - Add C2PA parent-chain lineage ablation

**Change:** Added `parentOf`/`c2pa.opened` re-signing, recursive recovery of the
regional assertion from validated ancestor manifests, and a strict
`PROVENANCE_DISCONTINUITY` policy result for an edited manifest without a parent.
The three conditions use identical crop-40 pixels and reuse saved WAM metrics.

**Reason:** The original prototype copied the regional assertion into every new
manifest but did not test an adversary who selectively omitted that assertion.
The new experiment separates assertion propagation, ancestor recovery, and an
explicitly severed provenance chain.

**Impact:** Existing regional flag rates are unchanged. Lineage-policy results
are reported separately because missing ancestry is not the same event as
geometric region loss or watermark suppression. A fresh-root manifest that makes
no edit claim remains indistinguishable from genuinely new provenance without
external trust or deployment policy.

**Trust boundary:** The experiment entry points are local research CLIs. File and
interpreter paths are supplied by the operator, checked for existence before the
run, and execute with the operator's existing permissions; they are not remote or
less-privileged inputs. Child processes use argument arrays with `shell=False`.
Accordingly, accepting an absolute path is intentional reproducibility behavior,
not a privilege-crossing path traversal surface.

### 2026-09-30 - Clarify instance allocation

**Change:** Distinguished the shared host's eight-GPU capacity from the
experiment instance's allocation of three GPUs, 18 vCPUs, and 12 GB RAM; each
experiment command used one assigned GPU.

**Reason:** Prevent the environment table from conflating physical-host capacity,
instance-visible resources, and per-command GPU use.

**Impact:** Documentation and environment metadata only; experimental outputs
and reported metrics are unchanged.

### 2026-09-30 - Add honest centered-crop cost controls

**Change:** Re-signed the existing 10% and 20% per-side centered crop pixels
with an honest AI-created-plus-edited manifest, recomputed retained signed-mask
area, and released 400 per-asset audit rows plus environment metadata.

**Reason:** Mild, geometry-aware controls favor the defense. These two stronger
but legitimate edits quantify the audit's explicit cost when signed-region
evidence is genuinely removed.

**Impact:** Strict flags increase to 15/200 (7.5%) and 68/200 (34.0%). A flag is
an evidence-removal signal rather than a malicious-intent classifier; attack,
LaMa, and earlier benign-control results remain unchanged.

### 2026-09-30 - Publish full benign-control evaluation

**Change:** Added the benign suite at 200 assets per condition, released 1,000
per-asset audit rows, and documented strict and baseline-qualified false-flag
estimands.

**Reason:** Quantify the audit's cost on honest edits and separate it from the
watermark detector's behavior on clean, unwatermarked images.

**Impact:** Adds the paper's benign-control table and threshold-grid evidence;
the attack and LaMa result sets are unchanged.

### 2026-09-29 - Add isolated LaMa inpainting experiment

**Change:** Added exact-mask inference, re-signing, auditing, summarization, and
unit-test scripts.

**Reason:** Test whether the region-aware defense catches signal removal when
the signed region remains geometrically present.

**Impact:** Adds a new result directory and external LaMa runtime; historical
crop experiments and paper numbers are unchanged until the full run is reviewed.
