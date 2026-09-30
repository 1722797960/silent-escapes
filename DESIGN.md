# Experiment Design Notes

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

## Change history

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
