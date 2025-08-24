# dashboard/services/mlflow_client.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from pathlib import Path
import os

import pandas as pd

from .cache import memoize
from .results_loader import load_dashboard_cfg

# ---- Internals ---------------------------------------------------------------

def _default_tracking_uri() -> str:
    # Repo-local file store by default
    return f"file:{Path('results/logs/mlruns').resolve().as_posix()}"

@memoize(maxsize=4)
def _client():
    import mlflow
    cfg = load_dashboard_cfg()
    uri = cfg.get("mlflow", {}).get("tracking_uri") or os.getenv("MLFLOW_TRACKING_URI") or _default_tracking_uri()
    mlflow.set_tracking_uri(uri)
    from mlflow.tracking import MlflowClient
    return MlflowClient()

def _get_experiment_id(name: str) -> Optional[str]:
    """Robust experiment lookup for MLflow 2.x (no list_experiments dependency)."""
    if not name:
        return None
    c = _client()
    # Fast path
    exp = c.get_experiment_by_name(name)
    if exp:
        return exp.experiment_id
    # Fallback search across active experiments (if API available)
    try:
        from mlflow.entities import ViewType  # type: ignore
        exps = c.search_experiments(view_type=ViewType.ACTIVE_ONLY)
    except Exception:
        try:
            exps = c.search_experiments()  # type: ignore[call-arg]
        except Exception:
            exps = []
    for e in exps:
        if e.name == name or e.name.lower() == name.lower():
            return e.experiment_id
    return None

# ---- Public API --------------------------------------------------------------

def list_runs(filters: Dict[str, Any] | None = None, max_results: int = 5000) -> pd.DataFrame:
    """
    Return MLflow runs as a flattened DataFrame (params/metrics/tags prefixed).
    Safe on MLflow 2.x; uses the experiment name from configs/dashboard.yaml.
    """
    try:
        cfg = load_dashboard_cfg()
        exp_name = cfg.get("mlflow", {}).get("experiment_name", "Neem-DNA-QML")
        eid = _get_experiment_id(exp_name)
        if not eid:
            return pd.DataFrame()

        c = _client()
        runs = c.search_runs(
            [eid],
            max_results=max_results,
            order_by=["attributes.start_time DESC"],
        )
        rows: List[Dict[str, Any]] = []
        for r in runs:
            params = dict(getattr(r.data, "params", {}) or {})
            metrics = dict(getattr(r.data, "metrics", {}) or {})
            tags = dict(getattr(r.data, "tags", {}) or {})
            rows.append(
                {
                    "run_id": r.info.run_id,
                    "experiment_id": r.info.experiment_id,
                    "status": r.info.status,
                    "start_time": r.info.start_time,
                    "end_time": r.info.end_time,
                    **{f"param.{k}": v for k, v in params.items()},
                    **{f"metric.{k}": v for k, v in metrics.items()},
                    **{f"tag.{k}": v for k, v in tags.items()},
                }
            )
        df = pd.DataFrame(rows)
        if not filters or df.empty:
            return df

        # Simple equality filtering; support scalar or list/tuple values
        for k, v in filters.items():
            if k not in df.columns:
                continue
            if isinstance(v, (list, tuple, set)):
                df = df[df[k].isin(list(v))]
            else:
                df = df[df[k] == v]
        return df
    except Exception:
        # Never crash the dashboard; return empty frame on client errors
        return pd.DataFrame()

def mlflow_run_link(run_id: str) -> str:
    """
    Return a link to the run in the MLflow UI if we can infer a base URL.
    - If MLFLOW_UI_BASE is set (e.g., http://127.0.0.1:5000), compose HTTP URL.
    - Else return the tracking URI fragment (not always directly clickable).
    """
    base = os.getenv("MLFLOW_UI_BASE")
    if base:
        base = base.rstrip("/")
        return f"{base}/#/experiments/0/runs/{run_id}"
    cfg = load_dashboard_cfg()
    uri = cfg.get("mlflow", {}).get("tracking_uri") or _default_tracking_uri()
    return f"{uri}#{run_id}"
