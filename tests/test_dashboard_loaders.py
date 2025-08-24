from __future__ import annotations
from pathlib import Path
import json
import numpy as np

from dashboard.services.results_loader import load_local_metrics, metrics_to_frame

def test_load_local_metrics(tmp_path: Path):
    mdir = tmp_path / "results" / "metrics"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "cnn.json").write_text(json.dumps({"f1":0.8,"roc_auc":0.9}), encoding="utf-8")
    (mdir / "qsvm.json").write_text(json.dumps({"f1":0.7,"roc_auc":0.85,"circuit_depth":12}), encoding="utf-8")

    d = load_local_metrics(str(mdir))
    assert "cnn" in d and "qsvm" in d
    df = metrics_to_frame(d)
    assert {"name","f1","roc_auc"}.issubset(set(df.columns))
