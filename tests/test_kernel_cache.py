from pathlib import Path
import importlib.util
import numpy as np
from qiskit import QuantumCircuit


def _load_module(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {mod_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_kernel_cache_roundtrip(tmp_path: Path, monkeypatch):
    cu_path = Path("scripts") / "cache_utils.py"
    cu = _load_module("cache_utils", cu_path)

    # Redirect cache dir to tmp
    monkeypatch.setattr(cu, "CACHE_DIR", tmp_path, raising=True)

    # Tiny dataset & feature map
    X = np.random.RandomState(0).randn(6, 4).astype("float32")
    qc = QuantumCircuit(4)
    qc.h(range(4))

    key = cu.kernel_cache_key(X, qc, "aer_simulator_statevector", 512, {"enabled": False})
    assert isinstance(key, str) and len(key) >= 8

    K = np.eye(len(X), dtype="float32")
    cu.save_kernel(key, K)
    K2 = cu.load_kernel(key)
    assert K2 is not None
    assert np.allclose(K, K2)
    assert K2.dtype == np.float32