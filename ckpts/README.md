# Model checkpoints

Model weights are not distributed in this Git repository. Obtain them from the
original projects and review their licenses before use.

| Local path | Upstream source |
|---|---|
| `ckpts/wam_mit.pth` and `ckpts/params.json` | [Watermark Anything](https://github.com/facebookresearch/watermark-anything) |
| `ckpts/sam/sam_vit_b_01ec64.pth` | [Segment Anything ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |
| `ckpts/pixelseal_checkpoint.pth` | [VideoSeal / PixelSeal](https://github.com/facebookresearch/videoseal) |
| PixArt-alpha components | [PixArt-XL-2-1024-MS](https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS) |

PixArt-alpha components were converted to fp16 for the reported run. The
conversion helper is `experiments/convert_t5_fp16.py`.
