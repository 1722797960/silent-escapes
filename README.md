# Silent Escapes

**The Granularity Gap Between Localized Watermarks and Region-Agnostic Provenance Audits**

This repository contains the experiment code, configurations, and compact result
artifacts for *Silent Escapes*. The study examines a cross-layer failure mode in
combined C2PA provenance and localized invisible-watermark pipelines: a crop can
remove the only contradicting watermark signal while a newly signed, misleading
image-level manifest remains cryptographically valid.

## Main findings

- The localized WAM signal degrades as the crop removes more of its marked region.
- The escape curve is reproduced with SDXL and PixArt-alpha generated images.
- A fixed-area, position-aware stress test shows that crop placement matters: at
  40% removed area, the center policy yields a 17.0% escape rate while the
  mask-aware policy yields 55.0%.
- A signed-region assertion and region-aware audit flag 686 of 752 SDXL silent
  escapes (91.2%) and 870 of 940 PixArt-alpha silent escapes (92.6%) at the
  paper's reference thresholds.
- A size-preserving, exact-mask LaMa inpainting stress test produces 199 silent
  escapes among 200 SDXL assets. The region-aware signal check flags all 199 as
  `WATERMARK_SUPPRESSED`, while every output retains valid C2PA and region data.

Precomputed machine-readable results are committed under `results/`, including
the 10,800 row-level C1 attack and audit records used to derive the position-aware
statistics and the 200 row-level LaMa audit records. Large model weights,
generated image corpora, contact sheets, and intermediate images are
intentionally not stored in Git.

## Repository layout

```text
configs/                         fixed experiment configurations
experiments/                     regional, position-aware, defense, and inpainting experiments
legacy_pixel_seal/               PixelSeal replication drivers
manifests/                       C2PA manifest templates
requirements/                    direct dependencies and reference lock file
results/                         curated JSON/CSV summaries and paper plots
scripts/                         global-watermark and four-state audit pipeline
third_party/watermark-anything/  vendored WAM inference code and licenses
tools/                           release verification and credential helpers
ckpts/README.md                  model-weight acquisition guide
data/README.md                   dataset generation and directory guide
REPRODUCIBILITY.md               claim-to-command-to-artifact mapping
DESIGN.md                        inpainting threat model and security boundaries
```

## Quick verification (CPU, no model weights)

The following command checks the release layout and recomputes the headline
numbers from the committed JSON artifacts:

```bash
python tools/verify_release.py
```

It does not run neural inference and normally completes in under a second.

## Environment

Reference environment:

- Python 3.10
- PyTorch 2.6.0 with CUDA 12.4
- `c2pa-python==0.37.10`
- WAM (`wam_mit.pth`), SAM ViT-B, and PixelSeal

On the tested Linux/CUDA environment:

```bash
bash setup_env.sh
source venvs/c2pa/bin/activate
```

For another CUDA version or for Windows, install the matching PyTorch build
first, then install `requirements/requirements.txt`. See
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the full commands and
[`ckpts/README.md`](ckpts/README.md) for weight sources.

## Demo signing credentials

No private key is committed. Generate a disposable P-256 demo CA and leaf
certificate locally:

```bash
python tools/generate_demo_credentials.py
```

This writes `certs/ec_key.pem` and `certs/ec_chain.pem`, both ignored by Git.
They are only for reproducing the research pipeline and must not be used as a
real identity or production signing credential.

## Reproduction levels

1. **Artifact verification:** run `python tools/verify_release.py`.
2. **Single-image smoke test:** download WAM and SAM weights, prepare one image
   and mask, then run the corresponding embedding/detection commands in
   `REPRODUCIBILITY.md`.
3. **Full paper reproduction:** generate the fixed-seed SDXL and PixArt-alpha
   datasets and run the staged experiment commands. GPU and storage requirements
   are documented in `data/README.md`.

## Scope and responsible use

The code evaluates authentication failure modes and defenses in a controlled
research setting. It does not contain production credentials. The mask-aware
crop policy reads the signed region geometry but does not query watermark
detections or audit verdicts. The inpainting condition uses the exact signed SAM
mask, zero dilation, and byte-identical pixels outside that mask; it is a
controlled stress test rather than a defense-adaptive attack.

## License

Project code is released under the MIT License. Vendored or separately installed
third-party components retain their own licenses; see `LICENSE` and the license
files under `third_party/`.
