#!/usr/bin/env python
# Aggregate metrics; compute optional bootstrap CIs if per-sample preds exist.
from __future__ import annotations
import argparse, json, yaml
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from sklearn import metrics

def load_yaml(p: str) -> dict:
    return yaml.safe_load(open(p, "r", encoding="utf-8"))

def bootstrap_ci(y_true: np.ndarray, y_proba: np.ndarray, n: int, seed: int, fn) -> Tuple[float, Tuple[float,float]]:
    rng = np.random.RandomState(seed)
    base = float(fn(y_true, y_proba))
    vals = []
    for _ in range(n):
        idx = rng.randint(0, len(y_true), size=len(y_true))
        vals.append(float(fn(y_true[idx], y_proba[idx])))
    lo, hi = np.percentile(vals, [2.5, 97.5]).tolist()
    return base, (float(lo), float(hi))

def try_load_preds(name: str, base_metrics_dir: Path, labels_path: Path) -> Tuple[np.ndarray | None, np.ndarray | None]:
    p = base_metrics_dir / f"proba_{name}.npy"
    if p.exists() and labels_path.exists():
        return np.load(labels_path), np.load(p)
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-config", required=True)
    ap.add_argument("--tracking", required=True)
    args = ap.parse_args()

    ev = load_yaml(args.eval_config)
    save_metrics = Path(ev["save_dir"]["metrics"]); save_metrics.mkdir(parents=True, exist_ok=True)

    # Collect model-level metrics if present
    out: Dict = {"results": {}, "ci": {}}
    for name in ("classical_summary", "qsvm", "vqc"):
        p = save_metrics / f"{name}.json"
        if p.exists():
            out["results"][name] = json.loads(p.read_text(encoding="utf-8"))

    # Optional CIs if predictions are available
    n_boot = int(ev["confidence_intervals"]["n_bootstrap"])
    seed = int(ev["confidence_intervals"]["seed"])
    labels_path = Path("data/interim/labels.npy")

    # CNN val predictions (if saved as proba_cnn.npy)
    y, proba = try_load_preds("cnn", save_metrics, labels_path)
    if y is not None:
        auc, auc_ci = bootstrap_ci(y, proba, n_boot, seed, metrics.roc_auc_score)
        brier = float(metrics.brier_score_loss(y, proba))
        out["ci"]["cnn"] = {"roc_auc": auc, "roc_auc_ci": [auc_ci[0], auc_ci[1]], "brier": brier}

    # QSVM val predictions (if saved as proba_qsvm.npy)
    y, proba = try_load_preds("qsvm", save_metrics, labels_path)
    if y is not None:
        auc, auc_ci = bootstrap_ci(y, proba, n_boot, seed, metrics.roc_auc_score)
        out["ci"]["qsvm"] = {"roc_auc": auc, "roc_auc_ci": [auc_ci[0], auc_ci[1]]}

    # VQC val predictions (if saved as proba_vqc.npy)
    y, proba = try_load_preds("vqc", save_metrics, labels_path)
    if y is not None:
        auc, auc_ci = bootstrap_ci(y, proba, n_boot, seed, metrics.roc_auc_score)
        out["ci"]["vqc"] = {"roc_auc": auc, "roc_auc_ci": [auc_ci[0], auc_ci[1]]}

    (save_metrics / "aggregate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[OK] Wrote aggregate metrics with CIs.")

if __name__ == "__main__":
    main()
