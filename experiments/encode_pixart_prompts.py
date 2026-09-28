"""Stage A: encode all prompts with T5-XL on a dedicated GPU, save embeddings.

The container has a 12GB cgroup RAM limit; T5-XL fp16 (9.6GB) barely fits and
forward-pass overhead pushes the full pipeline over. Splitting text encoding
(generator A) from image synthesis (generate_pixart.py, stage B) lets each
process stay within limits: A keeps T5 on its own GPU, B never loads a text
encoder at all.

Writes data/pixart_200/text_embeddings/{idx:04d}.pt containing the prompt
embeddings dict PixArtAlphaPipeline expects from encode_prompt.
"""
import glob
import os

import torch
from transformers import T5EncoderModel, T5Tokenizer

E2E_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(E2E_ROOT, "data/pixart_200/text_embeddings")
DEVICE = "cuda:1"  # dedicated GPU for the text encoder


def main():
    prompts = []
    for tsv in [os.path.join(E2E_ROOT, "data/originals_500/sampled_prompts.tsv"),
                os.path.join(E2E_ROOT, "data/originals_500/sampled_prompts_50_499.tsv")]:
        with open(tsv, encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    prompts.append((int(parts[0]), parts[1]))
    prompts.sort(key=lambda t: t[0])
    prompts = prompts[:200]
    os.makedirs(OUT, exist_ok=True)

    te = T5EncoderModel.from_pretrained(
        os.path.join(E2E_ROOT, "ckpts/pixart_t5_fp16"),
        torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True)
    te.to(DEVICE).eval()

    tok = T5Tokenizer.from_pretrained(
        os.path.join(E2E_ROOT, "ckpts/pixart_tokenizer"), maxlen=120)

    with torch.inference_mode():
        for idx, prompt in prompts:
            out_path = os.path.join(OUT, f"{idx:04d}.pt")
            if os.path.exists(out_path):
                continue
            inp = tok([prompt], return_tensors="pt", max_length=120,
                      padding="max_length", truncation=True)
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            out = te(**inp)
            emb = out[0] if isinstance(out, tuple) else out.last_hidden_state
            # PixArt attention_mask for the transformer cross-attention
            torch.save({"prompt_embeds": emb.cpu(),
                        "prompt_attention_mask": inp["attention_mask"].cpu()},
                       out_path)
            if (len(os.listdir(OUT)) % 20) == 0:
                print(f"{len(os.listdir(OUT))}/200 encoded", flush=True)
    # classifier-free guidance needs the unconditional embedding too
    null_path = os.path.join(OUT, "null.pt")
    if not os.path.exists(null_path):
        with torch.inference_mode():
            inp = tok([""], return_tensors="pt", max_length=120,
                      padding="max_length", truncation=True)
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            out = te(**inp)
            emb = out[0] if isinstance(out, tuple) else out.last_hidden_state
            torch.save({"prompt_embeds": emb.cpu(),
                        "prompt_attention_mask": inp["attention_mask"].cpu()},
                       null_path)
    print("done:", OUT)


if __name__ == "__main__":
    main()
