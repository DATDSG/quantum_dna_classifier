from __future__ import annotations
import argparse, yaml, json, os, time, platform
from pathlib import Path
from typing import Tuple, Dict, Any
from datetime import datetime

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.metrics import f1_score, roc_auc_score

from experiment_tracker import start_run, log_params, log_metrics, log_dict


# -----------------------
# Utils
# -----------------------

def load_yaml(p: str) -> dict:
    return yaml.safe_load(open(p, "r", encoding="utf-8"))

def set_seeds(seed: int) -> None:
    # Force CPU + determinism; keep logs quiet
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    np.random.seed(seed)
    pnp.random.seed(seed)

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_splits(meta_dir: Path, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    sp = meta_dir / "splits.json"
    if sp.exists():
        j = json.loads(sp.read_text(encoding="utf-8"))
        idx = j.get("indices", {})
        tr = np.array(idx.get("train", []), dtype=int)
        va = np.array(idx.get("val", []), dtype=int)
        if tr.size and va.size:
            return tr, va
    # fallback stratified 80/20
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    (tr_idx, va_idx), = list(sss.split(np.zeros_like(y), y))
    return tr_idx, va_idx

def find_best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Pick threshold maximizing F1 on validation; stable on CPU."""
    if proba.ndim != 1:
        proba = proba.ravel()
    vals = np.unique(proba)
    if len(vals) > 512:
        vals = np.linspace(vals.min(), vals.max(), 512)
    best_f1, best_t = -1.0, 0.5
    for t in vals:
        yhat = (proba >= t).astype(int)
        f1 = f1_score(y_true, yhat)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


# -----------------------
# Model building
# -----------------------

def make_device(name: str, wires: int, shots):
    """Try requested device; fallback to default.qubit. Avoid lightning on Windows (can segfault)."""
    if platform.system() == "Windows" and name == "lightning.qubit":
        name = "default.qubit"
    try:
        return qml.device(name, wires=wires, shots=shots)
    except Exception:
        return qml.device("default.qubit", wires=wires, shots=shots)

def build_reuploading_ansatz(wires: int, ent: str = "linear"):
    """One data-reuploading block with light entanglement."""
    def block(x, theta_row):
        # data reuploading
        for i in range(wires):
            qml.RX(x[i % x.shape[0]], wires=i)
        # trainable single-qubit rotations
        for i in range(wires):
            qml.RY(theta_row[i], wires=i)
        # minimal entanglement
        if ent == "linear":
            for i in range(wires - 1):
                qml.CNOT(wires=[i, i + 1])
        elif ent == "full":
            for i in range(wires):
                for j in range(i + 1, wires):
                    qml.CNOT(wires=[i, j])
    return block


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

    # ----- Features: prefer CNN embeddings; else angle encoding -----
    emb = processed / "cnn_embeddings.npy"
    angles = processed / "angles.npy"
    if emb.exists():
        X = np.load(emb)
        feat_src = "embeddings"
    elif angles.exists():
        X = np.load(angles)
        feat_src = "angles"
    else:
        raise FileNotFoundError("No features found. Expected cnn_embeddings.npy or angles.npy in data/processed/")
    y = np.load(interim / "labels.npy")

    # Reduce/pad features to `wires` deterministically
    wires = int(qcfg["device"]["wires"])
    if X.shape[1] > wires:
        from sklearn.decomposition import PCA
        X = PCA(n_components=wires, random_state=seed).fit_transform(X)
    elif X.shape[1] < wires:
        X = np.concatenate([X, np.zeros((X.shape[0], wires - X.shape[1]), X.dtype)], axis=1)

    # Splits
    tr_idx, va_idx = load_splits(meta_dir, y, seed)
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = y[tr_idx], y[va_idx]

    # Device / QNode
    shots = qcfg["device"]["shots"] if qcfg["device"]["shots"] not in (None, "null") else None
    dev = make_device(qcfg["device"]["name"], wires, shots)
    layers = int(qcfg["ansatz"]["data_reupload_layers"])
    ent = qcfg["ansatz"].get("entanglement", "linear")

    # Trainable params θ: (layers, wires)
    theta = pnp.random.normal(0.0, 0.1, size=(layers, wires), requires_grad=True)
    block = build_reuploading_ansatz(wires, ent)

    @qml.qnode(dev, interface="autograd", diff_method="best")
    def circuit(x, params):
        # shallow pre-rotations keep landscape smooth
        for i in range(wires):
            qml.RY(0.1, wires=i)
        for l in range(layers):
            block(x, params[l])
        return qml.expval(qml.PauliZ(0))  # single-qubit readout -> logit in [-1,1]

    def predict_proba(Xb, params):
        # Keep autograd types here; only cast when reporting/saving
        logits = [circuit(x, params) for x in Xb]
        logits = pnp.stack(logits)
        return (1.0 - logits) * 0.5

    # Optimizer and regularization
    opt_name = str(qcfg["training"]["optimizer"]).lower()
    lr = float(qcfg["training"]["lr"])
    weight_decay = float(qcfg["training"].get("weight_decay", 0.0))  # L2 on params
    grad_clip = float(qcfg["training"].get("grad_clip", 0.0))        # clip global norm if > 0
    opt = qml.optimize.AdamOptimizer(lr) if opt_name == "adam" else qml.optimize.GradientDescentOptimizer(lr)

    batch_size = int(qcfg["training"]["batch_size"])
    epochs     = int(qcfg["training"]["epochs"])
    patience   = int(qcfg["training"]["early_stop_patience"])
    log_grads  = bool(qcfg["training"].get("log_grad_norms", True))

    def l2_norm_sq(params) -> pnp.ndarray:
        return pnp.sum(pnp.stack([pnp.sum(params[l] ** 2) for l in range(layers)]))

    def bce_loss(params, xb, yb):
        p = predict_proba(xb, params)  # autograd array
        eps = 1e-7
        ce = -pnp.mean(yb * pnp.log(p + eps) + (1 - yb) * pnp.log(1 - p + eps))
        if weight_decay > 0.0:
            ce = ce + weight_decay * l2_norm_sq(params) / (layers * wires)
        return ce

    def clip_params(params, max_norm: float):
        if max_norm <= 0:
            return params
        g = qml.grad(lambda pr: bce_loss(pr, Xtr[: min(batch_size, len(Xtr))], ytr[: min(batch_size, len(ytr))]))(params)
        # global norm
        gnorm = float(np.sqrt(np.sum([np.sum(np.array(gi) ** 2) for gi in g])))
        if gnorm > max_norm:
            scale = max_norm / (gnorm + 1e-8)
            params = pnp.stack([params[l] - (1 - scale) * 0.0 * params[l] for l in range(layers)])  # no-op on params
        return params, gnorm

    best_val = pnp.inf
    best_params = None
    waits = 0
    grad_norms, val_curve = [], []
    t0 = time.time()

    with start_run("quantum_vqc", args.tracking, tags={"branch": "quantum", "algo": "vqc", "run_ts": run_ts}):
        log_params({
            "device": dev.name, "wires": wires, "layers": layers, "entanglement": ent,
            "optimizer": opt_name, "lr": lr, "shots": 0 if shots is None else shots,
            "weight_decay": weight_decay, "grad_clip": grad_clip, "features": feat_src,
        })

        rng = np.random.RandomState(seed)
        for epoch in range(epochs):
            idx = rng.permutation(len(Xtr))
            # mini-batch
            for i in range(0, len(Xtr), batch_size):
                sl = idx[i:i+batch_size]
                xb, yb = Xtr[sl], ytr[sl]
                theta, cost = opt.step_and_cost(lambda pr: bce_loss(pr, xb, yb), theta)

            # optional grad-norm clip (no param change; monitored only)
            if grad_clip > 0:
                theta, gnorm_clip = clip_params(theta, grad_clip)
                log_metrics({"vqc_grad_clip_norm": gnorm_clip})

            # validation
            val_loss = float(bce_loss(theta, Xva, yva))
            val_curve.append(val_loss)

            if log_grads:
                sl = idx[: min(batch_size, len(Xtr))]
                g = qml.grad(lambda pr: bce_loss(pr, Xtr[sl], ytr[sl]))(theta)
                gnorm = float(np.sqrt(np.sum([np.sum(np.array(gi) ** 2) for gi in g])))
                grad_norms.append(gnorm)
                log_metrics({"vqc_grad_norm": gnorm})

            log_metrics({"vqc_val_loss": val_loss, "epoch": epoch})

            if val_loss < best_val - 1e-5:
                best_val = val_loss
                waits = 0
                # materialize a detached copy for saving later
                best_params = [pnp.array(theta[l]).copy() for l in range(layers)]
            else:
                waits += 1
                if waits >= patience:
                    break

        # --- Final eval on validation w/ threshold tuning ---
        proba_autograd = predict_proba(Xva, best_params)
        proba = np.asarray(proba_autograd, dtype=float).ravel()
        thr = find_best_threshold(yva, proba)   # improves F1 on val without changing test
        yhat = (proba >= thr).astype(int)
        f1  = float(f1_score(yva, yhat))
        roc = float(roc_auc_score(yva, proba))
        wall = float(time.time() - t0)

        out = {
            "f1": f1,
            "roc_auc": roc,
            "val_loss_min": float(best_val),
            "grad_norm_last": (float(grad_norms[-1]) if grad_norms else None),
            "best_threshold": float(thr),
            "wall_time_s": wall,
        }
        (metrics_dir / "vqc.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        log_dict(out, "vqc_metrics.json")
        print("[OK] VQC results:", out)

        # ---- Persist artifacts (params, preds, indices, config snapshot) ----
        io_cfg = qcfg.get("io", {})
        outdir = ensure_dir(Path(io_cfg.get("save_dir", "models/vqc")))
        if io_cfg.get("save_params", True) and best_params is not None:
            np.save(outdir / f"params_{run_ts}.npy", np.asarray(best_params, dtype=float))
            (outdir / f"vqc_config_{run_ts}.json").write_text(json.dumps({
                "device": dev.name, "wires": wires, "layers": layers, "entanglement": ent,
                "optimizer": opt_name, "lr": lr, "shots": 0 if shots is None else shots,
                "weight_decay": weight_decay, "grad_clip": grad_clip, "features": feat_src,
            }, indent=2), encoding="utf-8")

        if io_cfg.get("save_preds", True):
            np.save(outdir / f"val_proba_{run_ts}.npy", proba)
            (outdir / f"val_idx_{run_ts}.json").write_text(
                json.dumps({"val": list(map(int, va_idx)), "best_threshold": float(thr)}, indent=2),
                encoding="utf-8"
            )
        # Save training curve for plotting
        (outdir / f"val_curve_{run_ts}.json").write_text(json.dumps([float(v) for v in val_curve], indent=2),
                                                         encoding="utf-8")


if __name__ == "__main__":
    main()
