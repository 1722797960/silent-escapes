"""Generate the incremental slice of the deterministic 500-sample set.

Reuses image_gen.py's stratified sampling logic (seed=42), generates ONLY
indices [start..end), saves as {global_index:04d}_...png into output-dir,
and writes sampled_prompts.tsv for the generated range. Repeatable/resumable:
already-existing PNGs are skipped.
"""
import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_parti_split(device_hint: str = ""):
    from datasets import load_dataset

    mirror_cfg = os.environ.get("HF_DATASETS_OFFLINE")
    try:
        return load_dataset("nateraw/parti-prompts", split="train")
    except Exception as e:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        return load_dataset("nateraw/parti-prompts", split="train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=500)
    ap.add_argument("--start", type=int, default=50)
    ap.add_argument("--end", type=int, default=None, help="exclusive; default=total")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()
    end = args.end if args.end is not None else args.total
    assert 0 <= args.start < end <= args.total

    # Reuse image_gen.py's helpers (vendored at the repo root of this project).
    sys.path.insert(0, _REPO_ROOT)
    import image_gen as ig

    split = get_parti_split()
    categories = sorted({row["Category"] for row in split})
    challenges = sorted({row["Challenge"] for row in split})
    indices = ig.stratified_sample_indices(
        num_samples=args.total,
        categories=list(categories),
        challenges=list(challenges),
        category_column="Category",
        challenge_column="Challenge",
        seed=args.seed,
    )
    slice_indices = indices[args.start:end]

    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, "sampled_prompts.tsv")
    meta_exists = os.path.isfile(meta_path)
    meta_f = open(meta_path, "a", encoding="utf-8")
    if not meta_exists:
        meta_f.write("index\tprompt\tcategory\tchallenge\n")

    pipe = ig.build_pipeline(args.device)

    generated = 0
    skipped = 0
    for gi, ds_idx in enumerate(slice_indices, start=args.start):
        row = split[ds_idx]
        prompt = row["Prompt"]
        cat = row["Category"]
        chal = row["Challenge"]
        safe = ig.sanitize_filename(prompt)
        fname = f"{gi:04d}_{safe}_cat-{cat}_chal-{chal}.png"
        fpath = os.path.join(args.output_dir, fname)
        if os.path.isfile(fpath):
            skipped += 1
            continue
        img_seed = args.seed + gi
        generator = torch.Generator(device=args.device).manual_seed(img_seed)
        image = pipe(prompt=prompt, generator=generator).images[0]
        image.save(fpath)
        generated += 1
        meta_f.write(f"{gi}\t{prompt.replace(chr(9),' ')}\t{cat}\t{chal}\n")
        if generated % 10 == 0:
            meta_f.flush()
            print(f"generated {generated} (last index {gi})", flush=True)
    meta_f.close()
    print(f"done. generated={generated} skipped={skipped}")


if __name__ == "__main__":
    main()
