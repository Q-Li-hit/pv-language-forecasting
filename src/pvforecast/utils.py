"""File identities and JSON output shared by the command-line tools."""

import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def model_digest(folder):
    """Identify a local model snapshot by file names and contents."""
    folder = Path(folder)
    files = sorted(path for path in folder.rglob("*") if path.is_file()
                   and path.suffix in (".json", ".safetensors", ".bin", ".model", ".txt", ".tiktoken"))
    if not files or not (folder / "config.json").is_file():
        raise ValueError("A local Transformers model directory with config.json is required.")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(folder).as_posix().encode())
        digest.update(file_sha256(path).encode())
    return digest.hexdigest()


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, np.ndarray):
            return clean(item.tolist())
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
