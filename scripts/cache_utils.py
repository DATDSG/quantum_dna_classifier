from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

# Cache directory for precomputed Gram matrices (QSVM)
CACHE_DIR = Path("results/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def kernel_cache_key(X: np.ndarray, fmap, backend: str, shots: int, noise_cfg: dict) -> str:
    """Stable key from meta + tiny data sketch. Avoids giant hashes."""
    meta = {
        "backend": backend, "shots": int(shots), "noise": noise_cfg,
        "n": int(X.shape[0]), "d": int(X.shape[1]),
        "fmap_depth": int(fmap.depth()), "fmap_qubits": int(fmap.num_qubits),
    }
    sketch = np.round(X[: min(32, len(X)), :], 6)  # small head sample
    payload = json.dumps(meta, sort_keys=True) + sketch.tobytes().hex()
    return hashlib.sha1(payload.encode()).hexdigest()

def _path(key: str) -> Path:
    return CACHE_DIR / f"qkernel_{key}.npy"

def save_kernel(key: str, K: np.ndarray) -> None:
    np.save(_path(key), K)

def load_kernel(key: str) -> np.ndarray | None:
    p = _path(key)
    return np.load(p) if p.exists() else None
