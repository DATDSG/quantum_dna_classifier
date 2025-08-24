#!/usr/bin/env python
# Parse Neem chloroplast (and optional related accessions), window sequence, label CDS vs rRNA/tRNA.
from __future__ import annotations
import argparse, json, os, re, yaml
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from Bio import Entrez, SeqIO


def load_yaml(p: str) -> dict:
    return yaml.safe_load(open(p, "r", encoding="utf-8"))


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    import random

    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(cfg: dict) -> None:
    for k in ("raw_dir", "interim_dir", "processed_dir", "embeddings_dir", "metadata_dir"):
        Path(cfg["paths"][k]).mkdir(parents=True, exist_ok=True)


def fetch_genbank(accession: str, raw_dir: str, email_env: str, api_key_env: str) -> Path:
    """Download GB file if missing; otherwise reuse disk. Requires NCBI_EMAIL env when fetching."""
    out = Path(raw_dir) / f"{accession}.gb"
    if out.exists():
        return out

    email = os.getenv(email_env)
    if not email:
        raise RuntimeError(
            f"Env {email_env} not set; cannot fetch {accession}. Place file at {out} or set email."
        )

    Entrez.email = email
    api_key = os.getenv(api_key_env)
    if api_key:
        Entrez.api_key = api_key

    with Entrez.efetch(db="nuccore", id=accession, rettype="gb", retmode="text") as h:
        data = h.read()  # returns str
        out.write_bytes(data.encode("utf-8"))  # fix: encode before writing
    return out


def _feature_intervals(record, types: List[str]) -> List[Tuple[int, int]]:
    iv = []
    for f in record.features:
        if f.type in types:
            try:
                s = int(f.location.start)
                e = int(f.location.end)
                iv.append((s, e))
            except Exception:
                pass
    return iv


def _overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def window_and_label(
    seq: str,
    cds_iv: List[Tuple[int, int]],
    rrna_iv: List[Tuple[int, int]],
    length: int,
    stride: int,
    uppercase=True,
    drop_amb=True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice genome into fixed windows. Label 1 if overlaps CDS; 0 if overlaps rRNA/tRNA; else drop."""
    if uppercase:
        seq = seq.upper()
    n = len(seq)
    wins, labs = [], []
    for start in range(0, n - length + 1, stride):
        end = start + length
        w_iv = (start, end)
        lab = None
        if any(_overlaps(w_iv, iv) for iv in cds_iv):
            lab = 1
        elif any(_overlaps(w_iv, iv) for iv in rrna_iv):
            lab = 0
        if lab is None:
            continue
        w = seq[start:end]
        if drop_amb and re.search(r"[^ACGT]", w):
            if sum(ch not in "ACGT" for ch in w) / len(w) > 0.05:
                continue
        wins.append(w)
        labs.append(lab)
    return np.array(wins, dtype=object), np.array(labs, dtype=np.int8)


def deduplicate(wins: np.ndarray, labs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    seen, keep_idx = set(), []
    for i, w in enumerate(wins):
        h = hash(w)
        if h in seen:
            continue
        seen.add(h)
        keep_idx.append(i)
    return wins[keep_idx], labs[keep_idx]


def process_accession(acc: str, cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
    gbp = fetch_genbank(
        acc, cfg["paths"]["raw_dir"], cfg["fetch"]["email_env"], cfg["fetch"]["api_key_env"]
    )
    rec = SeqIO.read(str(gbp), "genbank")
    cds_iv = _feature_intervals(rec, ["CDS"])
    rrna_iv = _feature_intervals(rec, ["rRNA", "tRNA"])
    wins, labs = window_and_label(
        str(rec.seq),
        cds_iv,
        rrna_iv,
        length=int(cfg["windowing"]["length"]),
        stride=int(cfg["windowing"]["stride"]),
        uppercase=bool(cfg["windowing"].get("uppercase", True)),
        drop_amb=bool(cfg["windowing"].get("drop_ambiguous", True)),
    )
    if bool(cfg["windowing"].get("deduplicate", True)):
        wins, labs = deduplicate(wins, labs)
    return wins, labs


def maybe_augment(
    cfg: dict, base_wins: np.ndarray, base_labs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Optionally pull related genomes if dataset too small."""
    if not cfg["augmentation"]["enabled"]:
        return base_wins, base_labs, [cfg["primary_accession"]]
    if len(base_wins) >= int(cfg["augmentation"]["min_windows"]):
        return base_wins, base_labs, [cfg["primary_accession"]]

    email = os.getenv(cfg["fetch"]["email_env"])
    if not email:
        return base_wins, base_labs, [cfg["primary_accession"]]

    Entrez.email = email
    fam = cfg["augmentation"]["taxonomy"]["name"]
    limit = int(cfg["augmentation"].get("limit", 5))
    forced = cfg["augmentation"].get("accessions", []) or []
    accs = list(forced)

    if not forced:
        query = f'{fam}[Organism] AND chloroplast[Title]'
        try:
            sr = Entrez.esearch(db="nuccore", term=query, retmax=limit + 1)
            ids = Entrez.read(sr)["IdList"]
            if ids:
                summ = Entrez.esummary(db="nuccore", id=",".join(ids))
                for d in Entrez.read(summ):
                    accs.append(d.get("Caption"))
        except Exception:
            pass

    accs = [a for a in accs if a and a != cfg["primary_accession"]][:limit]
    all_w, all_y, srcs = [base_wins], [base_labs], [cfg["primary_accession"]]

    for acc in accs:
        try:
            w, y = process_accession(acc, cfg)
            if len(w) == 0:
                continue
            all_w.append(w)
            all_y.append(y)
            srcs.append(acc)
        except Exception:
            continue

    wins = np.concatenate(all_w, axis=0)
    labs = np.concatenate(all_y, axis=0)
    return wins, labs, srcs


def save_splits(y: np.ndarray, sources: List[str], cfg: dict) -> dict:
    """Save stratified splits; reuse if already saved."""
    outp = Path(cfg["splits"]["saved_path"])
    if outp.exists():
        return json.loads(outp.read_text(encoding="utf-8"))

    from sklearn.model_selection import StratifiedShuffleSplit

    rng = int(cfg["splits"]["random_seed"])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - cfg["splits"]["train"], random_state=rng)
    (tr_all, te_all), = list(sss.split(np.zeros_like(y), y))
    val_ratio = cfg["splits"]["val"] / (cfg["splits"]["val"] + cfg["splits"]["test"])
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_ratio, random_state=rng)
    (va, _), = list(sss2.split(np.zeros_like(y[te_all]), y[te_all]))
    va = te_all[va].tolist()
    te = list(set(te_all.tolist()) - set(va))
    tr = tr_all.tolist()

    out = {"indices": {"train": tr, "val": va, "test": te}, "seed": rng}
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    set_seeds(int(cfg["splits"]["random_seed"]))
    ensure_dirs(cfg)

    wins, labs = process_accession(cfg["primary_accession"], cfg)
    wins, labs, srcs = maybe_augment(cfg, wins, labs)

    np.save(cfg["output_files"]["windows_npy"], wins)
    np.save(cfg["output_files"]["labels_npy"], labs)
    meta = {"sources": srcs, "n": int(len(wins))}
    (Path(cfg["paths"]["metadata_dir"]) / "windows_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    save_splits(labs, srcs, cfg)
    print(f"[OK] windows={len(wins)} labels saved → {cfg['output_files']['windows_npy']}, {cfg['output_files']['labels_npy']}")


if __name__ == "__main__":
    main()
