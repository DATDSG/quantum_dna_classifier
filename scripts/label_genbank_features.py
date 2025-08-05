# scripts/label_genbank_features.py

from Bio import SeqIO
import os
import numpy as np

def extract_labeled_sequences(gb_path, min_length=30):
    records = list(SeqIO.parse(gb_path, "genbank"))
    assert len(records) == 1, "Expected a single GenBank record"

    record = records[0]
    dna_seq = record.seq
    sequences = []
    labels = []

    for feature in record.features:
        if feature.type in ["CDS", "tRNA", "rRNA"]:
            start = int(feature.location.start)
            end = int(feature.location.end)
            seq = str(dna_seq[start:end])

            if len(seq) < min_length:
                continue

            label = 1 if feature.type == "CDS" else 0
            sequences.append(seq)
            labels.append(label)

    return sequences, labels

if __name__ == "__main__":
    gb_path = "data/raw/azadirachta_indica.gb"
    if not os.path.exists(gb_path):
        raise FileNotFoundError("GenBank file not found at path: " + gb_path)

    sequences, labels = extract_labeled_sequences(gb_path)
    print(f"✅ Extracted {len(sequences)} labeled sequences")

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/X_filtered.npy", sequences)
    np.save("data/processed/y_labels.npy", labels)
    print("✅ Saved: data/processed/X_filtered.npy and y_labels.npy")
