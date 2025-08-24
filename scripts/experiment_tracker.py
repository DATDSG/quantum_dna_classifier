# Minimal MLflow helpers; CPU-only friendly; safe if MLflow missing.
from __future__ import annotations
import contextlib, json, yaml
from pathlib import Path
from typing import Dict

try:
    import mlflow
except Exception:
    mlflow = None  # type: ignore

def _load_tracking(cfg_path: str) -> dict:
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
    return cfg.get("mlflow", {"experiment_name": "default", "tracking_uri": "file:results/logs/mlruns", "tags": {}})

@contextlib.contextmanager
def start_run(run_name: str, tracking_cfg_path: str, tags: Dict | None = None):
    if mlflow is None:
        yield
        return
    t = _load_tracking(tracking_cfg_path)
    mlflow.set_tracking_uri(t["tracking_uri"])
    mlflow.set_experiment(t["experiment_name"])
    with mlflow.start_run(run_name=run_name):
        all_tags = dict(t.get("tags", {}))
        if tags: all_tags.update(tags)
        if all_tags:
            mlflow.set_tags(all_tags)
        yield

def log_params(d: Dict):
    if mlflow is None: return
    mlflow.log_params(d)

def log_metrics(d: Dict):
    if mlflow is None: return
    mlflow.log_metrics(d)

def log_dict(d: Dict, artifact_path: str):
    if mlflow is None: return
    mlflow.log_dict(d, artifact_file=artifact_path)
