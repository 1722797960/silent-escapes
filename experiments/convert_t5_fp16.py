"""One-time conversion: PixArt's T5-XL text encoder fp32 -> fp16 shards.

The fp32 originals (19GB) peak past a 12GB cgroup RAM limit during
from_pretrained even with low_cpu_mem_usage. This script streams each shard
tensor-by-tensor (mmap, convert, re-pack into ~1GB fp16 parts), keeping the
peak at roughly one shard read + one fp16 shard write.

Source: download the fp32 PixArt-XL-2-1024-MS snapshot from HuggingFace first
(e.g. `huggingface-cli download PixArt-alpha/PixArt-XL-2-1024-MS`), then point
SRC at its text_encoder directory. The fp16 output here already ships as
ckpts/pixart_t5_fp16, so this script only needs to be re-run when regenerating
from scratch.
"""
import glob
import json
import os

from safetensors import safe_open
from safetensors.torch import save_file

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Override with -DSRC=... / -DDST=... style env vars if regenerating.
SRC = os.environ.get(
    'T5_FP32_SRC',
    os.path.join(_REPO_ROOT, 'ckpts/pixart_t5_fp32_src/text_encoder'))
DST = os.environ.get('T5_FP16_DST', os.path.join(_REPO_ROOT, 'ckpts/pixart_t5_fp16'))


def main():
    os.makedirs(DST, exist_ok=True)
    for f in glob.glob(SRC + '/*.json'):
        json.dump(json.load(open(f)), open(DST + '/' + os.path.basename(f), 'w'))

    total_parts = 0
    for shard in sorted(glob.glob(SRC + '/model-*.safetensors')):
        base = os.path.basename(shard)
        out_idx, cur, cur_bytes = 0, {}, 0
        with safe_open(shard, framework='pt') as f:
            for k in f.keys():
                t = f.get_tensor(k).half()
                cur[k] = t
                cur_bytes += t.numel() * 2
                if cur_bytes > 1_000_000_000:
                    save_file(cur, f'{DST}/{base}.fp16.part{out_idx}')
                    cur, cur_bytes = {}, 0
                    out_idx += 1
                del t
        if cur:
            save_file(cur, f'{DST}/{base}.fp16.part{out_idx}')
            out_idx += 1
        print(f'{base} -> {out_idx} fp16 parts', flush=True)
        total_parts += out_idx
    print(f'done: {total_parts} parts in {DST}')


if __name__ == '__main__':
    main()
