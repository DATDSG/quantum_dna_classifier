from __future__ import annotations
import json, os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import yaml
import numpy as np
import pandas as pd

from .cache import memoize

def _safe_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

@memoize(maxsize=8)
def load_dashboard_cfg(cfg_path: str = "configs/dashboard.yaml") -> Dict[str, Any]:
    p = Path(cfg_path)
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}

@memoize(maxsize=8)
def load_local_metrics(metrics_dir: str) -> Dict[str, Dict[str, Any]]:
    """Read results/metrics/*.json into a {stem: payload} dict."""
    d: Dict[str, Dict[str, Any]] = {}
    mdir = Path(metrics_dir)
    if not mdir.exists(): return d
    for f in sorted(mdir.glob("*.json")):
        d[f.stem] = _safe_json(f)
    return d

def metrics_to_frame(metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, payload in metrics.items():
        row = {"name": name, **{k: payload.get(k) for k in [
            "f1","roc_auc","accuracy","precision","recall","brier","ece",
            "wall_time_s","best_threshold",
            "circuit_depth","two_qubit_gates","shots","kernel_cond_train","n_features"
        ]}}
        rows.append(row)
    return pd.DataFrame(rows)

@memoize(maxsize=8)
def load_windows_meta() -> Dict[str, Any]:
    p = Path("data/metadata/windows_meta.json")
    return _safe_json(p) if p.exists() else {}

@memoize(maxsize=8)
def load_labels() -> np.ndarray:
    p = Path("data/interim/labels.npy")
    return np.load(p) if p.exists() else np.array([])

@memoize(maxsize=8)
def load_lengths() -> List[int]:
    wpath = Path("data/interim/windows.npy")
    if not wpath.exists(): return []
    wins = np.load(wpath, allow_pickle=True)
    return [len(w) for w in wins]

def export_dataframe(df, path: str) -> str:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return str(p)
