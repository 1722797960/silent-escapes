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

## Change history

### 2026-09-29 - Add isolated LaMa inpainting experiment

**Change:** Added exact-mask inference, re-signing, auditing, summarization, and
unit-test scripts.

**Reason:** Test whether the region-aware defense catches signal removal when
the signed region remains geometrically present.

**Impact:** Adds a new result directory and external LaMa runtime; historical
crop experiments and paper numbers are unchanged until the full run is reviewed.
