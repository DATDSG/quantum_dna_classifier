# scripts/encode_dna.py

from Bio import SeqIO
import os
import numpy as np
from collections import Counter

def read_fasta(file_path):
    """
    Reads a FASTA file and returns a list of DNA sequences as strings.
    """
    return [str(record.seq) for record in SeqIO.parse(file_path, "fasta")]

def filter_short_sequences(sequences, min_length=30):
    """
    Removes sequences shorter than min_length.
    Useful to clean up raw or partial entries.
    """
    return [seq for seq in sequences if len(seq) >= min_length]

def kmer_encoding(sequence, k=3):
    """
    Converts a DNA sequence into a dictionary of k-mer frequency values.
    """
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    count = Counter(kmers)
    total = sum(count.values())
    return {kmer: count[kmer] / total for kmer in count}

def encode_kmer_batch(sequences, k=3):
    """
    Encodes a batch of sequences into k-mer frequency vectors.
    Returns a NumPy array and the ordered k-mer vocabulary.
    """
    all_kmers = set()
    encoded = []

    # First pass: get all k-mers in all sequences
    for seq in sequences:
        vec = kmer_encoding(seq, k)
        all_kmers.update(vec.keys())
        encoded.append(vec)

    all_kmers = sorted(all_kmers)  # consistent feature order

    # Second pass: fill vectors
    features = []
    for vec in encoded:
        features.append([vec.get(kmer, 0.0) for kmer in all_kmers])

    return np.array(features), all_kmers

def one_hot_encode(sequences, pad_char="N", pad_to="max"):
    """
    One-hot encodes sequences into NumPy array shape (N, L, 5),
    where L is length of longest or median sequence (for padding).
    
    Parameters:
        sequences: list of DNA strings
        pad_char: character to pad with (usually 'N')
        pad_to: either "max" or "median" to determine padding length
    """
    vocab = ['A', 'C', 'G', 'T', pad_char]
    vocab_dict = {ch: i for i, ch in enumerate(vocab)}

    # Ensure all uppercase strings
    sequences = [str(seq).upper() for seq in sequences]

    if pad_to == "median":
        lengths = [len(seq) for seq in sequences]
        target_len = int(np.median(lengths))
    else:
        target_len = max(len(seq) for seq in sequences)

    # Pad or truncate sequences
    padded = []
    for seq in sequences:
        if len(seq) > target_len:
            padded.append(seq[:target_len])
        else:
            padded.append(seq.ljust(target_len, pad_char))

    # Convert to one-hot encoded array
    N = len(padded)
    one_hot = np.zeros((N, target_len, len(vocab)), dtype=np.uint8)

    for i, seq in enumerate(padded):
        for j, ch in enumerate(seq):
            one_hot[i, j, vocab_dict.get(ch, vocab_dict[pad_char])] = 1

    return one_hot

def save_numpy_array(data, filepath):
    """
    Saves any NumPy array to .npy format.
    """
    np.save(filepath, data)
