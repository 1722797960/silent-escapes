# Generated data

The full generated image corpus is not stored in Git. Recreate it with the fixed
commands in `REPRODUCIBILITY.md`.

Expected directories:

- `data/originals_500/`: 500 SDXL images.
- `data/masks_200/`: SAM masks for the 200-image regional subset.
- `data/pixart_200/`: 200 PixArt-alpha images.
- `data/masks_pixart_200/`: corresponding SAM masks.
- `data/derived/`: local embedding, signing, attack, and verification products.

Generated data and masks are ignored by Git; this README remains tracked.
