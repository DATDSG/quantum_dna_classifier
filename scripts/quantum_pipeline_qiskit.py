from __future__ import annotations
import argparse, yaml, json, os
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from datetime import datetime

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score
from joblib import dump

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit.circuit import ParameterVector

from noise_models import simple_noise_model
from quantum_stats import summarize_circuit_stats
from cache_utils import kernel_cache_key, save_kernel, load_kernel
from experiment_tracker import start_run, log_params, log_metrics, log_dict


# -----------------------
# Helpers
# -----------------------

def load_yaml(p: str) -> dict:
    return yaml.safe_load(open(p, "r", encoding="utf-8"))

def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU-only
    import random
    random.seed(seed); np.random.seed(seed)
    try:
        from qiskit.utils import algorithm_globals
        algorithm_globals.random_seed = seed
    except Exception:
        pass

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def maybe_pca(X: np.ndarray, qcfg: Dict, seed: int) -> Tuple[np.ndarray, Optional[Any]]:
    """Optionally apply PCA to reduce feature dimension (stabilizes kernels)."""
    pca_cfg = qcfg.get("pca", {})
    if not pca_cfg or not pca_cfg.get("enabled", False):
        return X, None
    n = int(pca_cfg.get("n_components", min(4, X.shape[1])))
    from sklearn.decomposition import PCA
    model = PCA(n_components=n, random_state=seed)
    return model.fit_transform(X), model

def build_feature_map(n_features: int, cfg: Dict) -> QuantumCircuit:
    """Prefer ZZFeatureMap; fallback to simple RY + CX entangling template."""
    reps = cfg["feature_map"].get("reps", 2)
    ent = cfg["feature_map"].get("entanglement", "linear")
    try:
        from qiskit.circuit.library import ZZFeatureMap
        return ZZFeatureMap(feature_dimension=n_features, reps=reps, entanglement=ent)
    except Exception:
        qc = QuantumCircuit(n_features)
        x = ParameterVector("x", n_features)
        for _ in range(reps):
            for i in range(n_features):
                qc.ry(x[i], i)
            for i in range(n_features - 1):
                qc.cx(i, i + 1)
        return qc

def _statevector_kernel(X: np.ndarray, fmap: QuantumCircuit) -> np.ndarray:
    """Exact kernel via statevector inner products."""
    states = []
    for x in X:
        qc = QuantumCircuit(fmap.num_qubits)
        qc.compose(fmap.assign_parameters(x), inplace=True)
        states.append(Statevector.from_instruction(qc))
    n = len(states)
    K = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            fid = state_fidelity(states[i], states[j])
            K[i, j] = K[j, i] = fid
    return K

def _sampling_kernel(
    X: np.ndarray,
    fmap: QuantumCircuit,
    backend_name: str,
    shots: int,
    noise_cfg: Dict,
    seed: int,
    t_opts: Dict,
) -> np.ndarray:
    """Approximate kernel by histogram intersection of sampled bitstrings."""
    backend = Aer.get_backend(backend_name)
    backend.set_options(seed_simulator=seed)
    noise_model = None
    if noise_cfg.get("enabled", False):
        noise_model = simple_noise_model(
            p1=noise_cfg["depolarizing"]["one_qubit"],
            p2=noise_cfg["depolarizing"]["two_qubit"],
            ro01=noise_cfg["readout_error"]["prob0to1"],
            ro10=noise_cfg["readout_error"]["prob1to0"],
        )
    n = len(X)
    K = np.zeros((n, n), dtype=float)
    # Pre-transpile circuits for each x
    circs = []
    for x in X:
        c = fmap.assign_parameters(x)
        circs.append(transpile(c, backend, **t_opts))
    counts = []
    for c in circs:
        res = backend.run(
            c, shots=shots, noise_model=noise_model, seed_simulator=seed
        ).result().get_counts()
        counts.append(res)
    for i in range(n):
        K[i, i] = 1.0
        for j in range(i + 1, n):
            keys = set(counts[i]) | set(counts[j])
            inter = sum(min(counts[i].get(k, 0), counts[j].get(k, 0)) for k in keys)
            K[i, j] = K[j, i] = inter / float(shots)
    return K

def center_kernel(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    one = np.ones((n, n)) / n
    return K - one @ K - K @ one + one @ K @ one


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qconfig", required=True)
    ap.add_argument("--encoding-config", required=True)
    ap.add_argument("--eval-config", required=True)
    ap.add_argument("--tracking", required=True)
    args = ap.parse_args()

    qcfg = load_yaml(args.qconfig)
    ecfg = load_yaml(args.encoding_config)
    evcfg = load_yaml(args.eval_config)
    dcfg = load_yaml("configs/data.yaml")

    seed = int(evcfg["random_seed"])
    set_seeds(seed)

    processed = Path(dcfg["paths"]["processed_dir"])
    interim   = Path(dcfg["paths"]["interim_dir"])
    meta_dir  = Path(dcfg["paths"]["metadata_dir"])
    metrics_dir = ensure_dir(Path(evcfg["save_dir"]["metrics"]))
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ---- Features: prefer CNN embeddings; else angles; optional PCA ----
    emb = processed / "cnn_embeddings.npy"
    if emb.exists():
        X = np.load(emb)
    else:
        X = np.load(processed / "angles.npy")
    X, pca_model = maybe_pca(X, qcfg, seed)
    y = np.load(interim / "labels.npy")

    # ---- Splits (immutable if present) ----
    sp = meta_dir / "splits.json"
    if sp.exists():
        j = json.loads(sp.read_text(encoding="utf-8"))
        idx = j.get("indices", {})
        tr_idx = np.array(idx.get("train", []), dtype=int)
        va_idx = np.array(idx.get("val", []), dtype=int)
    else:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        (tr_idx, va_idx), = list(sss.split(np.zeros_like(y), y))

    # ---- Feature map + circuit stats ----
    fmap = build_feature_map(X.shape[1], qcfg)
    stats = summarize_circuit_stats(fmap)

    # ---- Backend/transpile knobs ----
    backend_name = qcfg["backend"]["name"]
    shots = int(qcfg["backend"]["shots"]) if qcfg["backend"]["shots"] else 2048
    t_opts: Dict[str, Any] = {}
    for k in ("coupling_map", "basis_gates", "optimization_level", "layout_method", "routing_method"):
        v = qcfg["backend"].get(k, None)
        if v not in (None, [], ""):
            t_opts[k] = v

    # ---- Cache key for kernel ----
    key = kernel_cache_key(X, fmap, backend_name, shots, qcfg.get("noise", {}))
    K = load_kernel(key) if qcfg.get("kernel", {}).get("cache", True) else None

    with start_run("quantum_qsvm", args.tracking, tags={"branch": "quantum", "algo": "qsvm", "run_ts": run_ts}):
        # Log high-level params
        log_params({
            "backend": backend_name,
            "shots": shots,
            "pca_enabled": bool(qcfg.get("pca", {}).get("enabled", False)),
            "pca_n_components": int(qcfg.get("pca", {}).get("n_components", 0)) if qcfg.get("pca", {}).get("enabled", False) else 0,
            "fmap_reps": int(qcfg["feature_map"].get("reps", 2)),
            "fmap_entanglement": qcfg["feature_map"].get("entanglement", "linear"),
            "kernel_center": bool(qcfg["kernel"].get("center", True)),
            "kernel_reg_eps": float(qcfg["kernel"].get("regularization_eps", 1e-10)),
            "noise_enabled": bool(qcfg.get("noise", {}).get("enabled", False)),
        })

        # Build kernel if no cache
        if K is None:
            if "statevector" in backend_name:
                K = _statevector_kernel(X, fmap)
            else:
                K = _sampling_kernel(X, fmap, backend_name, shots, qcfg.get("noise", {}), seed, t_opts)
            if qcfg.get("kernel", {}).get("cache", True):
                save_kernel(key, K)

        # Center + regularize
        if qcfg["kernel"].get("center", True):
            K = center_kernel(K)
        reg_eps = float(qcfg["kernel"].get("regularization_eps", 1e-10))
        K = K + np.eye(K.shape[0]) * reg_eps

        # Slice train/val
        K_train = K[np.ix_(tr_idx, tr_idx)]
        K_val   = K[np.ix_(va_idx, tr_idx)]
        ytr, yva = y[tr_idx], y[va_idx]

        # Optional small C-search if list provided
        Cs = qcfg["svm"]["C"]
        Cs = Cs if isinstance(Cs, (list, tuple)) else [Cs]
        best_clf, best_f1, best_proba, best_C = None, -1.0, None, None
        for Cval in Cs:
            clf = SVC(
                kernel="precomputed",
                C=float(Cval),
                class_weight=qcfg["svm"]["class_weight"],
                probability=bool(qcfg["svm"].get("probability", True)),
                random_state=seed,
            )
            clf.fit(K_train, ytr)
            proba = clf.predict_proba(K_val)[:, 1]
            yhat = (proba >= 0.5).astype(int)
            f1 = f1_score(yva, yhat)
            if f1 > best_f1:
                best_f1, best_clf, best_proba, best_C = f1, clf, proba, float(Cval)

        # Metrics
        yhat = (best_proba >= 0.5).astype(int)
        f1 = float(best_f1)
        roc = float(roc_auc_score(yva, best_proba))
        try:
            cond = float(np.linalg.cond(K_train))
        except Exception:
            cond = float("nan")

        out = {
            "f1": f1,
            "roc_auc": roc,
            "n_features": int(X.shape[1]),
            "circuit_depth": int(stats.get("depth", 0)),
            "two_qubit_gates": int(stats.get("two_qubit_gates", 0)),
            "shots": int(shots),
            "kernel_cond_train": cond,
            "best_C": best_C,
        }
        log_metrics({"qsvm_f1": f1, "qsvm_roc": roc, "kernel_cond_train": cond})

        # Persist metrics + artifacts
        ensure_dir(metrics_dir).joinpath("qsvm.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        log_dict(out, "qsvm_metrics.json")

        io_cfg = qcfg.get("io", {})
        outdir = ensure_dir(Path(io_cfg.get("save_dir", "models/qsvm")))
        if io_cfg.get("save_kernel", True):
            np.save(outdir / f"K_train_{run_ts}.npy", K_train)
            np.save(outdir / f"K_val_{run_ts}.npy", K_val)
        if io_cfg.get("save_svc", True) and best_clf is not None:
            dump(best_clf, outdir / f"svc_{run_ts}.joblib")
        if io_cfg.get("save_pca", True) and pca_model is not None:
            dump(pca_model, outdir / f"pca_{run_ts}.joblib")
        # Save validation predictions & indices for reproducibility
        np.save(outdir / f"val_proba_{run_ts}.npy", best_proba)
        (outdir / f"val_idx_{run_ts}.json").write_text(
            json.dumps({"val": list(map(int, va_idx))}, indent=2), encoding="utf-8"
        )
        # Save run config snapshot
        (outdir / f"run_config_{run_ts}.json").write_text(
            json.dumps(
                {
                    "backend": backend_name,
                    "shots": shots,
                    "feature_dim": int(X.shape[1]),
                    "pca_enabled": bool(qcfg.get("pca", {}).get("enabled", False)),
                    "pca_n_components": int(qcfg.get("pca", {}).get("n_components", 0)) if qcfg.get("pca", {}).get("enabled", False) else 0,
                    "fmap_reps": int(qcfg["feature_map"].get("reps", 2)),
                    "fmap_entanglement": qcfg["feature_map"].get("entanglement", "linear"),
                    "kernel_center": bool(qcfg["kernel"].get("center", True)),
                    "kernel_reg_eps": reg_eps,
                    "noise_enabled": bool(qcfg.get("noise", {}).get("enabled", False)),
                    "best_C": best_C,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print("[OK] QSVM results:", out)


if __name__ == "__main__":
    main()
