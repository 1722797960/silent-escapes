# Reproducibility guide

This document maps each quantitative claim to its command and committed output.
Run all commands from the repository root.

## 1. Verify the released artifacts

```bash
python tools/verify_release.py
```

This validates the curated release structure and recomputes the paper's headline
rates from the committed JSON files. It requires only the Python standard library.

## 2. Result map

| Paper component | Experiment entry point | Committed artifact |
|---|---|---|
| Global WAM/PixelSeal baselines | `scripts/run_a_stage2.sh` and `legacy_pixel_seal/` | `results/summary_500.json` |
| SDXL regional escape curve | `experiments/regional_v2.py` | `results/regional_v2/summary.json` |
| PixArt-alpha regional escape curve | `experiments/regional_v2.py` | `results/regional_v2_pixart/summary.json` |
| Fixed-area position-aware stress test | `experiments/c1_attack_suite_v1.py`, `c1_audit_v1.py`, `c1_summarize_v1.py` | `results/c1_v1/full/summary.json` |
| End-to-end region-aware defense, SDXL | `experiments/defense_e2e_run.py`, then `recompute_region_retention.py` | `results/defense_e2e/audit_retention.json` |
| End-to-end region-aware defense, PixArt-alpha | same pipeline | `results/defense_e2e_pixart/audit_retention.json` |
| Defense threshold sensitivity | `experiments/threshold_sensitivity_v1.py` | `results/threshold_sensitivity_v1/threshold_sensitivity.json` |
| Independent C2PA SDK verification | `experiments/independent_verify.py` | `results/defense_e2e/independent_verification.json` |

## 3. Fixed experimental settings

- WAM decoding threshold: `t = 0.75`
- WAM payload: 32 bits
- SAM model: ViT-B
- Position-aware crop areas: `rho = 0.0, 0.1, ..., 0.8`
- Crop policies: center, four edge anchors, random, and mask-aware
- Random seed: `20260928`
- Bootstrap resamples: 2,000
- Region-retention floor: `0.05`
- Region logit-mean threshold: `0.5`
- Resize filter: LANCZOS

The effective configuration written by each run is retained beside the result
summary. Historical absolute machine paths have been replaced with repository-
relative paths in the public artifacts; metric values are unchanged.

## 4. Dataset preparation

Generate the fixed-seed SDXL set:

```bash
python image_gen.py --num-samples 500 --output-dir data/originals_500 --device cuda
```

Generate SAM masks:

```bash
python experiments/sam_segment.py \
  --input-dir data/originals_500 \
  --output-dir data/masks_200 \
  --num-images 200 \
  --ckpt ckpts/sam/sam_vit_b_01ec64.pth
```

PixArt-alpha generation is split to reduce peak memory use:

```bash
python experiments/encode_pixart_prompts.py
python experiments/generate_pixart.py
```

See `data/README.md` for expected directories and `ckpts/README.md` for model
sources. Generated corpora are not committed because they are deterministic and
large.

## 5. Regional escape curves

```bash
python experiments/regional_v2.py \
  --input-dir data/originals_500 \
  --mask-dir data/masks_200 \
  --output-dir results/regional_v2 \
  --ckpt ckpts/wam_mit.pth \
  --params-json ckpts/params.json
```

Use `data/pixart_200`, `data/masks_pixart_200`, and
`results/regional_v2_pixart` for the cross-generator arm.

## 6. End-to-end region-aware audit

First generate disposable credentials:

```bash
python tools/generate_demo_credentials.py
```

Then run the staged defense pipeline:

```bash
python experiments/defense_e2e_run.py --stage all --num-images 200
python experiments/recompute_region_retention.py --e2e-dir results/defense_e2e
python experiments/independent_verify.py \
  --output-json results/defense_e2e/independent_verification.json
```

The full run requires the image corpus, masks, checkpoints, a CUDA-capable GPU,
and enough storage for signed intermediate assets. It does not require any
private service or API credential.

## 7. Position-aware stress test

The public configuration is `configs/c1_v1.json`. The three stages are:

```bash
python experiments/c1_attack_suite_v1.py --config configs/c1_v1.json
python experiments/c1_audit_v1.py --config configs/c1_v1.json
python experiments/c1_summarize_v1.py --config configs/c1_v1.json
```

The mask-aware policy selects the lowest-retention option among four preset edge
anchors using the signed region mask. It never observes watermark detector output
or audit verdicts.
