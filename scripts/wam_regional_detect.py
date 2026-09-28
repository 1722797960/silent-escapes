"""Quantify regional-watermark survival vs crop severity."""
import argparse, json, os, sys
import torch, torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WAM_REPO = os.path.join(_REPO_ROOT, "third_party", "watermark-anything")
sys.path.insert(0, WAM_REPO)
from notebooks.inference_utils import load_model_from_checkpoint
from watermark_anything.data.metrics import msg_predict_inference
from watermark_anything.data.transforms import default_transform

def crop_and_back(img, frac):
    w, h = img.size
    m = int(round(w * frac))
    img = img.crop((m, m, w - m, h - m))
    return img.resize((w, h), Image.LANCZOS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--metadata-path", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--params-json", required=True)
    ap.add_argument("--crop-fracs", default="0.0,0.1,0.2,0.3,0.4")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    abs_params, abs_ckpt = os.path.abspath(args.params_json), os.path.abspath(args.ckpt)
    prev = os.getcwd(); os.chdir(WAM_REPO)
    try:
        model = load_model_from_checkpoint(abs_params, abs_ckpt)
    finally:
        os.chdir(prev)
    model.to(args.device).eval()

    metadata = json.load(open(args.metadata_path))["images"]
    fracs = [float(x) for x in args.crop_fracs.split(",")]
    rows = []
    for entry in metadata:
        msg = torch.tensor(entry["message_bits"], dtype=torch.float32)
        path = os.path.join(args.input_dir, entry["watermarked_image"])
        for frac in fracs:
            img = Image.open(path).convert("RGB")
            if frac > 0:
                img = crop_and_back(img, frac)
            t = tensor = default_transform(img).unsqueeze(0).to(args.device)
            with torch.no_grad():
                preds = model.detect(t)["preds"]
            mask_preds = torch.sigmoid(preds[:, 0])
            bit_preds = preds[:, 1:]
            pred = msg_predict_inference(bit_preds, mask_preds).cpu().float()[0]
            acc = (pred == msg).float().mean().item()
            coverage = (mask_preds > 0.5).float().mean().item()
            rows.append({"image": entry["input_image"], "crop_frac": frac,
                         "bit_accuracy": acc, "mask_coverage": coverage})
            print(f'{entry["input_image"][:40]} crop={frac:.2f} acc={acc:.3f} cov={coverage:.2f}', flush=True)
    with open(args.output_json, "w") as f:
        json.dump({"rows": rows}, f, indent=1)
    # summary table
    for frac in fracs:
        vals = [r["bit_accuracy"] for r in rows if r["crop_frac"] == frac]
        above = sum(1 for v in vals if v >= 0.75)
        print(f"crop={frac:.2f}: mean={sum(vals)/len(vals):.3f} min={min(vals):.3f} detected(>=0.75)={above}/{len(vals)}")

if __name__ == "__main__":
    main()
