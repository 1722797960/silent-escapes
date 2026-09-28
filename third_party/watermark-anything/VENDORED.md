# Vendored code: Watermark Anything (WAM)

This is a **read-only vendored subset** of Meta's
[watermark-anything](https://github.com/facebookresearch/watermark-anything) repository
(license: MIT + LICENSE-COCO for COCO-derived assets — see LICENSE files here).

What was kept (everything the experiments import):

* `watermark_anything/` — the model package (embedder/extractor/metrics/augmentation …)
* `configs/` — YAML configs read by `load_model_from_checkpoint` **via relative paths**
  (embedding scripts `os.chdir` into this directory before constructing the model — do
  not rename/move these files relative to the package)
* `notebooks/inference_utils.py` — checkpoint loading helper
* `requirements.txt` — upstream dependency list (informational; see ../../requirements/)

What was NOT vendored: `checkpoints/` (the paper uses `wam_mit.pth`, shipped at
`../../ckpts/wam_mit.pth`, byte-identical to upstream), notebook assets, training scripts
beyond the package.

The weights were not modified; the code is unmodified except that nothing here is imported
from the network at runtime.
