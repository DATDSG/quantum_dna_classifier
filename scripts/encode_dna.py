from Bio import SeqIO
import os
import numpy as np
from collections import Counter
from sklearn.preprocessing import OneHotEncoder

def read_fasta(file_path):
    """Reads a FASTA file and returns a list of sequences."""
    return [str(record.seq) for record in SeqIO.parse(file_path, "fasta")]

def filter_short_sequences(sequences, min_length=30):
    """Removes sequences shorter than min_length."""
    return [seq for seq in sequences if len(seq) >= min_length]

def kmer_encoding(sequence, k=3):
    """Converts a DNA sequence into k-mer frequency vector."""
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    count = Counter(kmers)
    total = sum(count.values())
    return {kmer: count[kmer] / total for kmer in count}

def encode_kmer_batch(sequences, k=3):
    """Encodes a batch of sequences into k-mer frequency vectors."""
    all_kmers = set()
    encoded = []
    for seq in sequences:
        vec = kmer_encoding(seq, k)
        all_kmers.update(vec.keys())
        encoded.append(vec)

    all_kmers = sorted(all_kmers)
    features = []
    for vec in encoded:
        features.append([vec.get(kmer, 0.0) for kmer in all_kmers])
    return np.array(features), all_kmers

def one_hot_encode(sequences):
    """Converts sequences to one-hot encoding (for CNNs)."""
    encoder = OneHotEncoder(handle_unknown='ignore')
    flattened = [list(seq) for seq in sequences]
    encoded = encoder.fit_transform(flattened).toarray()
    return encoded

def save_numpy_array(data, filepath):
    """Saves encoded data as .npy"""
    np.save(filepath, data)
