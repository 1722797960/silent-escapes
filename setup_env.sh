#!/usr/bin/env bash
# Create the single Python environment needed by every experiment in this repo.
#
# One venv (venvs/c2pa) covers all stages: WAM embedding/detection, SAM
# segmentation, C2PA signing/verification/audit, SDXL & PixArt generation.
# Scripts that need the c2pa interpreter either read $C2PA_PY or default to
# venvs/c2pa/bin/python.
#
# GPU note: pick the PyTorch index URL that matches your CUDA driver.
#   CUDA 12.4 driver (>= 550): keep the cu124 lines below (tested here)
#   CPU-only:                  swap the torch install for plain `pip install torch --index-url https://download.pytorch.org/whl/cpu`
#   Windows: install torch from pytorch.org, then `pip install -r requirements/requirements.txt`
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d venvs/c2pa ]; then
  python3 -m venv venvs/c2pa
fi
VPIP=venvs/c2pa/bin/pip
$VPIP install --upgrade pip

# PyTorch — MUST use the cu124 index on the reference container (driver 550.120,
# CUDA 12.4). Default PyPI torch is a CUDA 13 build and requires driver >= 580.
$VPIP install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

# Diffusion + HF stack, C2PA, plotting, crypto, science deps
$VPIP install diffusers==0.40.0 transformers accelerate datasets safetensors \
  matplotlib cryptography "c2pa-python==0.37.10" omegaconf scikit-learn \
  scikit-image "segment-anything==1.0" jupyterlab notebook

# videoseal (PixelSeal) — Meta's package, installed from its GitHub tarball.
# VIDEOSEAL_REF may be set to a commit SHA for a frozen rerun; it defaults to
# main because the historical environment did not preserve the source commit.
VIDEOSEAL_REF=${VIDEOSEAL_REF:-main}
VIDEOSEAL_TMP=$(mktemp -d)
trap 'rm -rf "$VIDEOSEAL_TMP"' EXIT
curl --fail --location --silent --show-error \
  "https://github.com/facebookresearch/videoseal/archive/${VIDEOSEAL_REF}.tar.gz" \
  --output "$VIDEOSEAL_TMP/videoseal.tar.gz"
tar xzf "$VIDEOSEAL_TMP/videoseal.tar.gz" -C "$VIDEOSEAL_TMP" --no-same-owner
VIDEOSEAL_SRC=$(find "$VIDEOSEAL_TMP" -mindepth 1 -maxdepth 1 -type d -name 'videoseal-*' -print -quit)
test -n "$VIDEOSEAL_SRC"
$VPIP install "$VIDEOSEAL_SRC"

# Sanity check
venvs/c2pa/bin/python - << 'EOF'
import torch, torchvision, diffusers, transformers, c2pa, videoseal, omegaconf, sklearn, skimage, segment_anything
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
print("diffusers", diffusers.__version__, "| c2pa-python", c2pa.__version__ if hasattr(c2pa, "__version__") else "ok")
print("videoseal OK — all deps installed")
EOF
echo "setup_env.sh DONE — activate with: source venvs/c2pa/bin/activate"
