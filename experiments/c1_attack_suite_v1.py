"""Generate C1 crop attacks and perform compliant C2PA re-signing.

This script reuses the already embedded and signed SDXL assets from the
end-to-end defense experiment. It never regenerates images, SAM masks, or WAM
messages. Each attacked PNG is re-signed with the misleading human-edited
manifest while carrying the original region assertion into the new manifest.

Severity is the removed image-area fraction rho. All methods therefore have a
comparable x-axis even though their crop geometry differs.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_E2E = ROOT / "results" / "defense_e2e"
DEFAULT_OUT = ROOT / "results" / "c1_v1"
DEFAULT_CONFIG = ROOT / "configs" / "c1_v1.json"
DEFAULT_C2PA_PY = ROOT / "venvs" / "c2pa" / "bin" / "python"
RESIGN_HELPER = ROOT / "experiments" / "resign_with_region_assertion.py"
CERT = ROOT / "certs" / "ec_chain.pem"
KEY = ROOT / "certs" / "ec_key.pem"
VALID_METHODS = ("center", "left", "right", "top", "bottom", "random")


def parse_csv(value: str, cast):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def stable_rng(seed: int, image_name: str, rho: float) -> random.Random:
    material = f"{seed}|{image_name}|{rho:.6f}".encode("utf-8")
    derived = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return random.Random(derived)


def crop_box(method: str, rho: float, width: int, height: int,
             rng: random.Random) -> tuple[int, int, int, int]:
    """Return a PIL crop box with approximately rho of image area removed."""
    if method not in VALID_METHODS:
        raise ValueError(f"unsupported method: {method}")
    if not 0.0 <= rho < 1.0:
        raise ValueError(f"rho must be in [0, 1), got {rho}")
    if rho == 0.0:
        return 0, 0, width, height

    if method in ("center", "random"):
        scale = math.sqrt(1.0 - rho)
        crop_w = max(1, min(width, int(round(width * scale))))
        crop_h = max(1, min(height, int(round(height * scale))))
        if method == "center":
            x0 = (width - crop_w) // 2
            y0 = (height - crop_h) // 2
        else:
            x0 = rng.randint(0, width - crop_w)
            y0 = rng.randint(0, height - crop_h)
        return x0, y0, x0 + crop_w, y0 + crop_h

    if method in ("left", "right"):
        removed = min(width - 1, max(0, int(round(width * rho))))
        return ((removed, 0, width, height) if method == "left"
                else (0, 0, width - removed, height))

    removed = min(height - 1, max(0, int(round(height * rho))))
    return ((0, removed, width, height) if method == "top"
            else (0, 0, width, height - removed))


def unpack_mask(bitmap64: str) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(base64.b64decode(bitmap64), dtype=np.uint8))
    if bits.size != 64 * 64:
        raise ValueError(f"expected 4096 mask bits, got {bits.size}")
    return bits.reshape(64, 64).astype(np.float32)


def mask_retention(mask: np.ndarray, box: tuple[int, int, int, int],
                   width: int, height: int) -> float:
    x0, y0, x1, y1 = box
    gh, gw = mask.shape
    gx0 = max(0, min(gw, int(math.floor(x0 * gw / width))))
    gy0 = max(0, min(gh, int(math.floor(y0 * gh / height))))
    gx1 = max(gx0 + 1, min(gw, int(math.ceil(x1 * gw / width))))
    gy1 = max(gy0 + 1, min(gh, int(math.ceil(y1 * gh / height))))
    kept = mask[gy0:gy1, gx0:gx1]
    return float(kept.sum() / max(mask.sum(), 1e-6))


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            completed.add(record["attack_id"])
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--e2e-dir", default=str(DEFAULT_E2E))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--num-images", type=int)
    parser.add_argument("--rhos", help="comma-separated removed-area fractions")
    parser.add_argument("--methods", help="comma-separated attack methods")
    parser.add_argument("--c2pa-python", default=os.environ.get("C2PA_PY", str(DEFAULT_C2PA_PY)))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    num_images = args.num_images or int(config["num_images"])
    rhos = (parse_csv(args.rhos, float) if args.rhos
            else [float(x) for x in config["removed_area_fractions"]])
    methods = (parse_csv(args.methods, str) if args.methods
               else list(config["methods"]))
    unknown = sorted(set(methods) - set(VALID_METHODS))
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}")

    e2e_dir = Path(args.e2e_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    attacks_dir = output_dir / "attacks"
    attacks_dir.mkdir(exist_ok=True)
    records_path = output_dir / "attack_manifest.jsonl"
    completed = load_completed(records_path) if args.resume else set()
    if records_path.exists() and not args.resume:
        raise SystemExit(f"{records_path} already exists; use --resume or a new output directory")

    # Do not call Path.resolve() here: the venv's python is a symlink, and
    # resolving it would bypass pyvenv.cfg and lose the c2pa installation.
    c2pa_python = Path(args.c2pa_python).expanduser()
    if not c2pa_python.is_absolute():
        c2pa_python = Path.cwd() / c2pa_python
    required = [c2pa_python, RESIGN_HELPER, CERT, KEY,
                e2e_dir / "embed_meta.json",
                e2e_dir / "watermarked_signed_ra"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required paths:\n" + "\n".join(missing))

    meta = json.loads((e2e_dir / "embed_meta.json").read_text(encoding="utf-8"))["images"]
    meta = meta[:num_images]
    run_config = {
        **config,
        "effective_num_images": len(meta),
        "effective_removed_area_fractions": rhos,
        "effective_methods": methods,
        "source_e2e_dir": str(e2e_dir),
        "output_dir": str(output_dir),
        "python": sys.version,
        "platform": platform.platform(),
        "started_unix": time.time(),
    }
    (output_dir / "config_effective.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    total = len(meta) * len(rhos) * len(methods)
    done = 0
    for image_index, entry in enumerate(meta):
        source = e2e_dir / "watermarked_signed_ra" / entry["signed_image"]
        if not source.exists():
            raise FileNotFoundError(source)
        original = Image.open(source).convert("RGB")
        width, height = original.size
        mask = unpack_mask(entry["bitmap64"])
        region = {
            "bbox_csv": entry["bbox_csv"],
            "bitmap64": entry["bitmap64"],
            "payload_hash": entry["payload_hash"],
            "alg": "wam_mit_regional",
        }

        for rho in rhos:
            for method in methods:
                attack_id = f"{image_index:04d}|{method}|{rho:.3f}"
                if attack_id in completed:
                    done += 1
                    continue
                rng = stable_rng(int(config["random_seed"]), entry["input_image"], rho)
                box = crop_box(method, rho, width, height, rng)
                x0, y0, x1, y1 = box
                attacked = original.crop(box).resize((width, height), Image.Resampling.LANCZOS)
                rho_actual = 1.0 - ((x1 - x0) * (y1 - y0) / (width * height))

                severity_dir = attacks_dir / method / f"rho_{int(round(rho * 100)):03d}"
                severity_dir.mkdir(parents=True, exist_ok=True)
                destination = severity_dir / entry["signed_image"]
                temporary = severity_dir / f"._c1_tmp_{os.getpid()}_{image_index:04d}.png"
                attacked.save(temporary, format="PNG")
                try:
                    result = subprocess.run(
                        [str(c2pa_python), str(RESIGN_HELPER),
                         "--input", str(temporary), "--output", str(destination),
                         "--region-json", "-", "--cert", str(CERT), "--key", str(KEY)],
                        input=json.dumps(region), text=True, capture_output=True)
                    if result.returncode != 0:
                        raise RuntimeError(
                            f"re-sign failed for {attack_id}:\nstdout={result.stdout}\nstderr={result.stderr}")
                finally:
                    temporary.unlink(missing_ok=True)

                record = {
                    "attack_id": attack_id,
                    "image_index": image_index,
                    "input_image": entry["input_image"],
                    "signed_image": entry["signed_image"],
                    "method": method,
                    "rho_requested": rho,
                    "rho_actual": round(rho_actual, 8),
                    "crop_box_px": [x0, y0, x1, y1],
                    "original_size": [width, height],
                    "mask_retention_geometry": round(
                        mask_retention(mask, box, width, height), 8),
                    "output_relpath": str(destination.relative_to(output_dir)).replace("\\", "/"),
                }
                append_jsonl(records_path, record)
                completed.add(attack_id)
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"generated {done}/{total}", flush=True)

    print(f"complete: {done}/{total}; records={records_path}")


if __name__ == "__main__":
    main()
