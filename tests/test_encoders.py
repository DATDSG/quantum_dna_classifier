import sys
from pathlib import Path
import importlib.util
import numpy as np
import pytest


def _load_module(mod_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {mod_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_one_hot_kmer_angles_roundtrip(tmp_path: Path):
    """Unit-test the pure functions in scripts/encode_dna.py (no files, pure CPU)."""
    enc_path = Path("scripts") / "encode_dna.py"
    if not enc_path.exists():
        pytest.skip("scripts/encode_dna.py not found")

    enc = _load_module("encode_dna", enc_path)

    # Tiny windows corpus
    wins = np.array(["ACGTAC", "AAAACC", "GGGGTT"], dtype=object)

    # one-hot (L=6, alphabet=ACGT)
    X_oh = enc.one_hot_encode(wins, ["A", "C", "G", "T"], L=6, pad_value=0)
    assert X_oh.shape == (3, 6, 4)
    # Check channel sums equal counts
    assert int(X_oh[0].sum()) == 6  # all positions one-hot set

    # k-mer counts and matrix
    X_kmer, vocab = enc.build_kmer_matrix(wins, k=2)
    assert X_kmer.shape[0] == 3
    assert len(vocab) == X_kmer.shape[1]
    assert (X_kmer >= 0).all()

    # scaling (tfidf then l2)
    X_scaled = enc.scale_features(X_kmer, mode="tfidf")
    assert np.isfinite(X_scaled).all()

    # angle mapping with clipping to [-1,1] -> [0, pi]
    Xang = enc.angles_from_features(X_scaled, clip=(-1.0, 1.0), scale="pi")
    assert Xang.shape == X_scaled.shape
    assert (Xang >= 0).all() and (Xang <= np.pi + 1e-6).all()
