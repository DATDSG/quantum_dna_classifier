#!/usr/bin/env python
# CNN saliency (if a saved model exists); robust no-op otherwise.
from __future__ import annotations
import argparse, yaml, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
import plotly.express as px

def load_yaml(p: str) -> dict:
    return yaml.safe_load(open(p, "r", encoding="utf-8"))

def find_model() -> Path | None:
    # Try common paths
    for pat in ["models/checkpoints/*.keras", "models/checkpoints/*.h5", "models/*.keras", "models/*.h5"]:
        for f in Path(".").glob(pat):
            return f
    return None

def saliency(model: keras.Model, X: np.ndarray, top_n: int = 8) -> list:
    """Grad of logit wrt input → |grad| as saliency."""
    out = []
    Xs = X[:top_n]
    x = tf.convert_to_tensor(Xs)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = model(x, training=False)
    grads = tape.gradient(y, x).numpy()
    sal = np.mean(np.abs(grads), axis=2)  # (N, L)
    for i in range(len(Xs)):
        out.append(sal[i])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-config", default="configs/classical_models.yaml")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--eval-config", default="configs/evaluation.yaml")
    args = ap.parse_args()

    mcfg = load_yaml(args.models_config)
    dcfg = load_yaml(args.data_config)
    ev   = load_yaml(args.eval_config)

    L = int(mcfg["cnn"]["max_len"])
    onehot = Path(dcfg["paths"]["processed_dir"]) / f"onehot_L{L}.npy"
    if not onehot.exists():
        print("[interpretability] onehot not found; run make encode")
        return
    X = np.load(onehot)
    model_path = find_model()
    if model_path is None:
        print("[interpretability] no saved CNN model found; skipping.")
        (Path(ev["save_dir"]["plots"]) / "interpretability_README.txt").write_text(
            "No model checkpoint found; run training with model checkpoint saving to enable saliency.",
            encoding="utf-8",
        )
        return
    model = keras.models.load_model(model_path)
    curves = saliency(model, X, top_n=min(8, len(X)))
    outdir = Path(ev["save_dir"]["plots"]); outdir.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(curves):
        fig = px.line(y=c, title=f"Saliency window {i}")
        fig.write_html(str(outdir / f"saliency_{i}.html"), include_plotlyjs="cdn")
    print(f"[OK] wrote {len(curves)} saliency plots to {outdir}")

if __name__ == "__main__":
    main()
