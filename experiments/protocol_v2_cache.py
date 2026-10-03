"""Two-layer cache for protocol-v2 detector and manifest results."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from protocol_v2_common import normalize_bits


PIXEL_CACHE_SCHEMA = "protocol-v2-pixel-cache-1"
MANIFEST_CACHE_SCHEMA = "protocol-v2-manifest-cache-1"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def decoded_rgb_key(path: str | Path) -> tuple[str, dict[str, Any]]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        prefix = f"{rgb.width}x{rgb.height}:RGB:".encode("ascii")
        digest = hashlib.sha256(prefix + rgb.tobytes()).hexdigest()
        return digest, {"width": rgb.width, "height": rgb.height, "mode": "RGB"}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class PixelCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _paths(self, key: str) -> tuple[Path, Path]:
        directory = self.root / key[:2]
        return directory / f"{key}.json", directory / f"{key}.npz"

    def contains(self, key: str) -> bool:
        metadata_path, arrays_path = self._paths(key)
        return metadata_path.is_file() and arrays_path.is_file()

    def put(
        self,
        *,
        key: str,
        decoded_bits: str | Sequence[int | bool],
        mask_logits: np.ndarray,
        image: dict[str, Any],
        model_sha256: str,
        checkpoint_sha256: str,
        preprocessing_version: str,
    ) -> None:
        metadata_path, arrays_path = self._paths(key)
        if self.contains(key):
            raise FileExistsError(f"pixel cache entry already exists: {key}")
        bits = normalize_bits(decoded_bits)
        logits = np.asarray(mask_logits, dtype=np.float16)
        if logits.ndim != 2:
            raise ValueError(f"mask_logits must be 2D, got shape {logits.shape}")
        metadata = {
            "schema": PIXEL_CACHE_SCHEMA,
            "key": key,
            "decoded_bits": bits,
            "mask_logits_shape": list(logits.shape),
            "mask_logits_dtype": "float16",
            "image": image,
            "model_sha256": model_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "preprocessing_version": preprocessing_version,
        }
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = arrays_path.with_suffix(".tmp.npz")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, mask_logits=logits)
        os.replace(temporary, arrays_path)
        _atomic_json(metadata_path, metadata)

    def get(self, key: str) -> dict[str, Any] | None:
        metadata_path, arrays_path = self._paths(key)
        if not (metadata_path.is_file() and arrays_path.is_file()):
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema") != PIXEL_CACHE_SCHEMA or metadata.get("key") != key:
            raise ValueError(f"invalid pixel cache metadata for {key}")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            logits = arrays["mask_logits"]
        expected_shape = tuple(metadata["mask_logits_shape"])
        if logits.dtype != np.float16 or logits.shape != expected_shape:
            raise ValueError(f"pixel cache array mismatch for {key}")
        return {**metadata, "mask_logits": logits}


class ManifestCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path_for(self, manifest_sha256: str) -> Path:
        return self.root / manifest_sha256[:2] / f"{manifest_sha256}.json"

    def get(self, manifest_sha256: str) -> dict[str, Any] | None:
        path = self.path_for(manifest_sha256)
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != MANIFEST_CACHE_SCHEMA:
            raise ValueError(f"invalid manifest cache schema: {path}")
        if payload.get("manifest_sha256") != manifest_sha256:
            raise ValueError(f"manifest cache key mismatch: {path}")
        return payload

    def put(self, *, manifest_sha256: str, payload: dict[str, Any]) -> None:
        path = self.path_for(manifest_sha256)
        if path.exists():
            raise FileExistsError(f"manifest cache entry already exists: {manifest_sha256}")
        _atomic_json(path, {
            "schema": MANIFEST_CACHE_SCHEMA,
            "manifest_sha256": manifest_sha256,
            **payload,
        })
