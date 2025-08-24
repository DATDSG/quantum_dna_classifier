#!/usr/bin/env python
# Build one-hot, k-mer, and angle encodings. Idempotent; CPU-only.
from __future__ import annotations
import argparse, yaml, json, os, re
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

def load_yaml(p: str) -> dict:
    return yaml.safe_load(open(p, "r", encoding="utf-8"))

def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random
    random.seed(seed); np.random.seed(seed)

def load_windows_labels(dcfg: dict) -> tuple[np.ndarray, np.ndarray]:
    w = np.load(dcfg["output_files"]["windows_npy"], allow_pickle=True)
    y = np.load(dcfg["output_files"]["labels_npy"])
    return w, y

def one_hot_encode(wins: np.ndarray, alphabet: list[str], L: int, pad_value: int = 0) -> np.ndarray:
    a2i = {ch: i for i, ch in enumerate(alphabet)}
    X = np.full((len(wins), L, len(alphabet)), pad_value, dtype=np.float32)
    for i, w in enumerate(wins):
        s = w[:L]
        for j, ch in enumerate(s):
            idx = a2i.get(ch, None)
            if idx is not None:
                X[i, j, idx] = 1.0
    return X

def kmer_counts(w: str, k: int, alphabet=("A","C","G","T")) -> dict:
    cnt = {}
    for i in range(0, len(w)-k+1):
        s = w[i:i+k]
        if re.search(rf"[^{''.join(alphabet)}]", s): continue
        cnt[s] = cnt.get(s, 0) + 1
    return cnt

def build_kmer_matrix(wins: np.ndarray, k: int) -> tuple[np.ndarray, list[str]]:
    # Build vocabulary across corpus
    vocab = {}
    rows = []
    for w in wins:
        cnt = kmer_counts(w, k)
        rows.append(cnt)
        for kmer in cnt.keys():
            if kmer not in vocab: vocab[kmer] = len(vocab)
    X = np.zeros((len(wins), len(vocab)), dtype=np.float32)
    for i, cnt in enumerate(rows):
        for kmer, c in cnt.items():
            X[i, vocab[kmer]] = float(c)
    vocab_list = [None]*len(vocab)
    for kmer, idx in vocab.items():
        vocab_list[idx] = kmer
    return X, vocab_list

def scale_features(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "tfidf":
        tfidf = TfidfTransformer(norm=None, use_idf=True, smooth_idf=True)
        X = tfidf.fit_transform(X).astype(np.float32).toarray()
        X = normalize(X, norm="l2").astype(np.float32)
    elif mode == "l2":
        X = normalize(X, norm="l2").astype(np.float32)
    return X

def angles_from_features(X: np.ndarray, clip: tuple[float,float], scale: str) -> np.ndarray:
    lo, hi = clip
    Xc = np.clip(X, lo, hi)
    if scale == "pi":
        # map to [0, π]
        Xn = (Xc - lo) / (hi - lo + 1e-9)
        return (np.pi * Xn).astype(np.float32)
    return Xc.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoding-config", required=True)
    ap.add_argument("--data-config", required=True)
    args = ap.parse_args()

    ecfg = load_yaml(args.encoding_config)
    dcfg = load_yaml(args.data_config)
    set_seeds(int(ecfg["random_seed"]))

    Path(dcfg["paths"]["processed_dir"]).mkdir(parents=True, exist_ok=True)
    wins, y = load_windows_labels(dcfg)

    # one-hot
    if ecfg["one_hot"]["enabled"]:
        L = int(ecfg["one_hot"]["max_len"])
        X_oh = one_hot_encode(wins, ecfg["one_hot"]["alphabet"], L, ecfg["one_hot"]["pad_value"])
        np.save(Path(dcfg["paths"]["processed_dir"]) / f"onehot_L{L}.npy", X_oh)

    # k-mer
    if ecfg["kmer"]["enabled"]:
        k = int(ecfg["kmer"]["k"])
        X_kmer, vocab = build_kmer_matrix(wins, k)
        X_kmer = scale_features(X_kmer, str(ecfg["kmer"]["normalize"]))
        np.save(Path(dcfg["paths"]["processed_dir"]) / f"kmer_k{k}.npy", X_kmer)
        (Path(dcfg["paths"]["metadata_dir"]) / f"kmer_k{k}_vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    # angle encoding (from k-mer by default)
    src = ecfg["angle_encoding"].get("source", "kmer")
    if src == "kmer":
        k = int(ecfg["kmer"]["k"])
        X = np.load(Path(dcfg["paths"]["processed_dir"]) / f"kmer_k{k}.npy")
    else:
        # fallback to flattened one-hot average
        L = int(ecfg["one_hot"]["max_len"])
        X = np.load(Path(dcfg["paths"]["processed_dir"]) / f"onehot_L{L}.npy")
        X = X.mean(axis=1)  # (N,4)
    clip = (float(ecfg["angle_encoding"]["feature_clip"]["min"]),
            float(ecfg["angle_encoding"]["feature_clip"]["max"]))
    Xang = angles_from_features(X, clip, str(ecfg["angle_encoding"]["scale"]))
    np.save(Path(dcfg["paths"]["processed_dir"]) / "angles.npy", Xang)

    # labels are already in interim; no change
    print("[OK] Encodings written to", dcfg["paths"]["processed_dir"])

if __name__ == "__main__":
    main()
