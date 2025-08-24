from __future__ import annotations
import argparse, json, yaml, os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn import svm, metrics, model_selection
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from tf_input import make_dataset
from experiment_tracker import start_run, log_params, log_metrics, log_dict


# ------------------------
# Utilities
# ------------------------

def load_yaml(p: str) -> dict:
    return yaml.safe_load(open(p, "r", encoding="utf-8"))

def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # force CPU for TF
    os.environ["TF_FORCE_CPU"] = "1"
    import random
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    # convince TF to not see any GPU if present
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")

def load_splits(meta_dir: Path, y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load immutable splits if present; else build a simple stratified 80/20."""
    sp = meta_dir / "splits.json"
    if sp.exists():
        j = json.loads(sp.read_text(encoding="utf-8"))
        idx = j.get("indices", {})
        tr = np.array(idx.get("train", []), dtype=int)
        va = np.array(idx.get("val", []), dtype=int)
        if tr.size and va.size:
            return tr, va
    # Fallback: stratified split
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    (tr_idx, va_idx), = list(sss.split(np.zeros_like(y), y))
    return tr_idx, va_idx


# ------------------------
# Models
# ------------------------

def build_cnn(cfg: dict) -> keras.Model:
    """Compact 1D CNN; names 'embedding' layer for downstream use."""
    try:
        lr = float(cfg.get("lr", 1e-3))
    except Exception:
        raise ValueError(f"cnn.lr must be float; got {repr(cfg.get('lr'))}")

    seq_len  = int(cfg.get("max_len", 300))
    channels = int(cfg.get("channels", 4))

    inputs = keras.Input(shape=(seq_len, channels), name="input_layer")
    x = keras.layers.Conv1D(filters=cfg.get("filters", [32, 64])[0],
                            kernel_size=cfg.get("kernel_sizes", [7, 5])[0],
                            padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPool1D()(x)
    x = keras.layers.Conv1D(filters=cfg.get("filters", [32, 64])[1],
                            kernel_size=cfg.get("kernel_sizes", [7, 5])[1],
                            padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(cfg.get("dropout", 0.3))(x)
    emb = keras.layers.Dense(int(cfg.get("embeddings_output_dim", 16)),
                             activation="relu", name="embedding")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(emb)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ------------------------
# Pipeline
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-config", required=True)
    ap.add_argument("--encoding-config", required=True)
    ap.add_argument("--eval-config", required=True)
    ap.add_argument("--tracking", required=True)
    args = ap.parse_args()

    mcfg = load_yaml(args.models_config)
    ecfg = load_yaml(args.encoding_config)
    evcfg = load_yaml(args.eval_config)
    dcfg = load_yaml("configs/data.yaml")

    set_seeds(int(evcfg["random_seed"]))

    processed = Path(dcfg["paths"]["processed_dir"])
    interim   = Path(dcfg["paths"]["interim_dir"])
    meta_dir  = Path(dcfg["paths"]["metadata_dir"])
    metrics_out = ensure_dir(Path(evcfg["save_dir"]["metrics"]))
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    with start_run("classical", args.tracking, tags={"branch": "classical", "run_ts": run_ts}):
        results: Dict[str, dict] = {}

        # ------------------------------------------------------------
        # SVM on k-mer (train on train split; evaluate on val)
        # ------------------------------------------------------------
        if mcfg.get("svm_rbf", {}).get("enabled", False):
            k = int(ecfg["kmer"]["k"])
            kmer_path = processed / f"kmer_k{k}.npy"
            if not kmer_path.exists():
                print(f"[WARN] Missing {kmer_path}; skipping SVM.")
            else:
                X_kmer = np.load(kmer_path)
                y = np.load(interim / "labels.npy")
                tr_idx, va_idx = load_splits(meta_dir, y, int(evcfg["random_seed"]))
                Xtr, Xva = X_kmer[tr_idx], X_kmer[va_idx]
                ytr, yva = y[tr_idx], y[va_idx]

                params = {"C": mcfg["svm_rbf"]["C"], "gamma": mcfg["svm_rbf"]["gamma"]}
                log_params({"svm_grid": json.dumps(params)})
                clf = svm.SVC(
                    kernel="rbf",
                    class_weight=mcfg["svm_rbf"]["class_weight"],
                    probability=True,
                    random_state=int(evcfg["random_seed"]),
                )
                grid = model_selection.GridSearchCV(
                    clf, params, cv=int(mcfg["svm_rbf"]["cv_folds"]),
                    n_jobs=-1, scoring="f1"
                )
                grid.fit(Xtr, ytr)
                best = grid.best_estimator_
                proba = best.predict_proba(Xva)[:, 1]
                yhat = (proba >= 0.5).astype(int)
                res = {
                    "f1": float(metrics.f1_score(yva, yhat)),
                    "roc_auc": float(metrics.roc_auc_score(yva, proba)),
                    "best_params": grid.best_params_,
                }
                results["svm_rbf"] = res
                log_metrics({"svm_rbf_f1": res["f1"], "svm_rbf_roc": res["roc_auc"]})

                # persist
                if mcfg["svm_rbf"].get("save", True):
                    outdir = ensure_dir(Path(mcfg["svm_rbf"].get("save_dir", "models/svm")))
                    dump(best, outdir / f"svm_rbf_{run_ts}.joblib")
                    np.save(outdir / f"val_proba_{run_ts}.npy", proba)
                    save_json({"val": list(map(int, va_idx))}, outdir / f"val_idx_{run_ts}.json")

        # ------------------------------------------------------------
        # Random Forest on k-mer (optional baseline)
        # ------------------------------------------------------------
        if mcfg.get("random_forest", {}).get("enabled", False):
            k = int(ecfg["kmer"]["k"])
            kmer_path = processed / f"kmer_k{k}.npy"
            if not kmer_path.exists():
                print(f"[WARN] Missing {kmer_path}; skipping RandomForest.")
            else:
                X_kmer = np.load(kmer_path)
                y = np.load(interim / "labels.npy")
                tr_idx, va_idx = load_splits(meta_dir, y, int(evcfg["random_seed"]))
                Xtr, Xva = X_kmer[tr_idx], X_kmer[va_idx]
                ytr, yva = y[tr_idx], y[va_idx]

                rf_cfg = mcfg["random_forest"]
                param_grid = {
                    "n_estimators": rf_cfg["n_estimators"],
                    "max_depth": rf_cfg["max_depth"],
                    "min_samples_leaf": rf_cfg["min_samples_leaf"],
                    "max_features": rf_cfg.get("max_features", ["sqrt"]),
                    "bootstrap": [rf_cfg.get("bootstrap", True)],
                    "class_weight": [rf_cfg.get("class_weight", "balanced")],
                }
                log_params({"rf_grid": json.dumps(param_grid)})

                base = RandomForestClassifier(random_state=int(evcfg["random_seed"]), n_jobs=-1)
                grid = model_selection.GridSearchCV(
                    base, param_grid, cv=int(rf_cfg.get("cv_folds", 5)),
                    n_jobs=-1, scoring="f1"
                )
                grid.fit(Xtr, ytr)
                best = grid.best_estimator_
                proba = best.predict_proba(Xva)[:, 1]
                yhat = (proba >= 0.5).astype(int)
                rf_f1 = float(metrics.f1_score(yva, yhat))
                rf_roc = float(metrics.roc_auc_score(yva, proba))
                results["random_forest"] = {"f1": rf_f1, "roc_auc": rf_roc}
                log_metrics({"rf_f1": rf_f1, "rf_roc": rf_roc})

                if rf_cfg.get("save", True):
                    outdir = ensure_dir(Path(rf_cfg.get("save_dir", "models/rf")))
                    dump(best, outdir / f"rf_{run_ts}.joblib")
                    np.save(outdir / f"val_proba_{run_ts}.npy", proba)
                    save_json({"val": list(map(int, va_idx))}, outdir / f"val_idx_{run_ts}.json")
                    # feature importances
                    try:
                        fi = best.feature_importances_.ravel().tolist()
                        with open(outdir / f"feature_importances_{run_ts}.csv", "w", encoding="utf-8") as fh:
                            fh.write("feature_index,importance\n")
                            for i, imp in enumerate(fi):
                                fh.write(f"{i},{imp}\n")
                    except Exception:
                        pass

        # ------------------------------------------------------------
        # Compact CNN (train on train; validate on val) + artifacts
        # ------------------------------------------------------------
        if mcfg.get("cnn", {}).get("enabled", False):
            target_L = int(mcfg["cnn"]["max_len"])
            onehot_path = processed / f"onehot_L{target_L}.npy"
            if not onehot_path.exists():
                cands = sorted(processed.glob("onehot_L*.npy"))
                if not cands:
                    print(f"[WARN] No one-hot encodings found in {processed}; skipping CNN.")
                else:
                    onehot_path = cands[0]
            if onehot_path.exists():
                X = np.load(onehot_path)
                y = np.load(interim / "labels.npy")
                tr_idx, va_idx = load_splits(meta_dir, y, int(evcfg["random_seed"]))
                Xtr, Xva = X[tr_idx], X[va_idx]
                ytr, yva = y[tr_idx], y[va_idx]

                # build model with inferred input from data
                cfg_cnn = dict(mcfg["cnn"])
                cfg_cnn["max_len"] = int(X.shape[1])
                cfg_cnn["channels"] = int(X.shape[2])
                model = build_cnn(cfg_cnn)

                outdir = ensure_dir(Path(mcfg["cnn"].get("save_dir", "models/cnn")))
                ckpt = keras.callbacks.ModelCheckpoint(
                    filepath=str(outdir / f"cnn_{run_ts}.keras"),
                    monitor="val_loss", save_best_only=True, save_weights_only=False
                )
                tb_cb = keras.callbacks.TensorBoard(log_dir=mcfg["cnn"]["tensorboard_dir"])
                es = keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=int(mcfg["cnn"]["early_stop_patience"]),
                    restore_best_weights=True,
                )

                ds_tr = make_dataset(Xtr, ytr, batch=int(mcfg["cnn"]["batch_size"]))
                ds_va = make_dataset(Xva, yva, batch=int(mcfg["cnn"]["batch_size"]), shuffle=False)

                hist = model.fit(
                    ds_tr, validation_data=ds_va,
                    epochs=int(mcfg["cnn"]["epochs"]),
                    callbacks=[tb_cb, es, ckpt], verbose=0
                )

                # reload best (safety)
                try:
                    model = keras.models.load_model(outdir / f"cnn_{run_ts}.keras")
                except Exception:
                    pass

                proba = model.predict(Xva, verbose=0).ravel()
                yhat = (proba >= 0.5).astype(int)
                res_cnn = {
                    "f1": float(metrics.f1_score(yva, yhat)),
                    "roc_auc": float(metrics.roc_auc_score(yva, proba)),
                }
                results["cnn"] = res_cnn
                log_metrics({"cnn_f1": res_cnn["f1"], "cnn_roc": res_cnn["roc_auc"]})

                # Persist artifacts
                save_json(hist.history, outdir / f"history_{run_ts}.json")
                np.save(outdir / f"val_proba_{run_ts}.npy", proba)
                save_json({"val": list(map(int, va_idx))}, outdir / f"val_idx_{run_ts}.json")

                # Save embeddings for ALL samples (downstream QSVM/VQC)
                if mcfg["cnn"].get("save_embeddings", True):
                    try:
                        get_emb = keras.Model(model.input, model.get_layer("embedding").output)
                    except Exception:
                        # fallback: penultimate layer
                        get_emb = keras.Model(model.input, model.layers[-2].output)
                    emb = get_emb.predict(X, verbose=0)
                    np.save(processed / "cnn_embeddings.npy", emb)

        # write consolidated metrics
        save_json(results, metrics_out / "classical_summary.json")
        log_dict(results, "classical_metrics.json")
        print("[OK] Classical results:", results)


if __name__ == "__main__":
    main()
