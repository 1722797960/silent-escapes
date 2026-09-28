# legacy_pixel_seal — PixelSeal arm of the 500-image replication (§4)

These three scripts drive Meta's **videoseal** package (which implements PixelSeal)
the same way the original Integrity Clash reproduction pipeline does. They are kept
separate from `scripts/` because they talk to a different watermarking stack.

* `watermark_embed.py` — embed PixelSeal 256-bit watermark into a directory of images
* `watermark_detect.py` — decode + report bit accuracy per image (same JSON contract as WAM's)
* `watermark_attack.py` — the three attack settings: `crop10` (centered 10% crop),
  `jpeg_q80` (JPEG round-trip at quality 80), `social` (resize+blur+re-compress social-media pipeline)

*vendored from the original integrity-clash reproduction workspace; no behavioral changes —
only the `image_gen` import path was made self-contained in `scripts/image_gen_extend.py`.*
