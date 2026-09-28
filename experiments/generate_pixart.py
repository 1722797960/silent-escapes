"""Generate the second-generator corpus with PixArt-alpha (Road B add-on).

Mirrors the original integrity-clash generation protocol: 200 prompts sampled
from the same stratified Parti-Prompts corpus (categories x challenge
dimensions, seed 42), 1024x1024 PNGs. Uses the SAME prompts as the SDXL
corpus (data/originals_500 sampled_prompts tsvs, first 200 indices) so the
only changed variable is the generator -- this is the point of the
cross-generator experiment.

Run under the integrity-clash venv (diffusers 0.40.0).
"""
import argparse
import glob
import os

import torch
from PIL import Image

E2E_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=os.path.join(E2E_ROOT, "data/pixart_200"))
    ap.add_argument("--num-images", type=int, default=200)
    ap.add_argument("--guidance-scale", type=float, default=4.5,
                    help="PixArt-alpha recommended CG (paper default 4.5)")
    ap.add_argument("--num-inference-steps", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # same prompt corpus as the SDXL set: first 200 indices of the merged tsvs
    prompts = []
    for tsv in [os.path.join(E2E_ROOT, "data/originals_500/sampled_prompts.tsv"),
                os.path.join(E2E_ROOT, "data/originals_500/sampled_prompts_50_499.tsv")]:
        with open(tsv, encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    prompts.append((int(parts[0]), parts[1]))
    prompts.sort(key=lambda t: t[0])
    prompts = prompts[: args.num_images]
    print(f"{len(prompts)} prompts, indices {prompts[0][0]}..{prompts[-1][0]}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- stage B: synthesis only. Prompt embeddings were pre-computed by
    # encode_pixart_prompts.py (T5-XL on a dedicated GPU) because the container's
    # 12GB cgroup RAM limit cannot hold T5 + a full pipeline at once.
    from diffusers import PixArtAlphaPipeline, PixArtTransformer2DModel, AutoencoderKL
    transformer = PixArtTransformer2DModel.from_pretrained(
        os.path.join(E2E_ROOT, "ckpts/pixart_transformer_fp16"),
        torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)
    vae = AutoencoderKL.from_pretrained(
        os.path.join(E2E_ROOT, "ckpts/pixart_vae_fp16"),
        torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)
    # vendored pipeline config: model_index.json + scheduler (tokenizer comes
    # pre-encoded embeddings; text encoder is not loaded at all)
    snap = os.path.join(E2E_ROOT, "ckpts/pixart_pipeline_cfg")
    pipe = PixArtAlphaPipeline.from_pretrained(
        snap, text_encoder=None, tokenizer=None, transformer=transformer, vae=vae,
        torch_dtype=torch.float16, use_safetensors=True,
    )
    pipe.to(args.device)
    pipe.vae.enable_tiling()  # decode-time VRAM guard, same as the SDXL pipeline

    gen = torch.Generator(device=args.device)
    emb_dir = os.path.join(E2E_ROOT, "data/pixart_200/text_embeddings")
    for idx, prompt in prompts:
        dest = os.path.join(args.output_dir, f"pixart_{idx:04d}.png")
        if os.path.exists(dest):
            continue
        emb = torch.load(os.path.join(emb_dir, f"{idx:04d}.pt"), map_location=args.device)
        uncond = torch.load(os.path.join(emb_dir, "null.pt"), map_location=args.device)
        gen.manual_seed(42 + idx)  # per-index deterministic seed, SDXL-style
        with torch.no_grad():
            img = pipe(negative_prompt=None,
                       prompt_embeds=emb["prompt_embeds"],
                       prompt_attention_mask=emb["prompt_attention_mask"],
                       negative_prompt_embeds=uncond["prompt_embeds"],
                       negative_prompt_attention_mask=uncond["prompt_attention_mask"],
                       num_inference_steps=args.num_inference_steps,
                       guidance_scale=args.guidance_scale, height=1024, width=1024,
                       generator=gen).images[0]
        img.save(dest, format="PNG")
        print(f"[{idx:04d}] {prompt[:50]} -> {os.path.basename(dest)}", flush=True)

    print("Done:", args.output_dir)


if __name__ == "__main__":
    main()
