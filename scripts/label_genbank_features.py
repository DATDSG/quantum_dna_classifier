# scripts/label_genbank_features.py

from Bio import SeqIO
import os
import numpy as np

def extract_labeled_sequences(gb_path, min_length=30):
    """
    Extract CDS, tRNA, and rRNA sequences from a single GenBank file.
    Label CDS = 1 (protein coding), tRNA/rRNA = 0 (non-coding).
    """
    records = list(SeqIO.parse(gb_path, "genbank"))
    assert len(records) == 1, f"Expected a single GenBank record in {gb_path}"

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
                continue  # skip short sequences

            label = 1 if feature.type == "CDS" else 0
            sequences.append(seq)
            labels.append(label)

    return sequences, labels

def load_all_labeled_genomes(species_file_dict, min_length=30):
    """
    Extract labeled sequences from multiple GenBank files.
    Returns all sequences and labels combined.
    """
    all_sequences = []
    all_labels = []

    for species, gb_path in species_file_dict.items():
        if not os.path.exists(gb_path):
            print(f"⚠️ Skipping {species} — GenBank file not found: {gb_path}")
            continue

        print(f"📥 Processing {species}: {gb_path}")
        seqs, labels = extract_labeled_sequences(gb_path, min_length=min_length)
        all_sequences.extend(seqs)
        all_labels.extend(labels)
        print(f"✅ {species}: {len(seqs)} sequences extracted")

    return all_sequences, all_labels

if __name__ == "__main__":
    # Define all GenBank sources
    species_to_files = {
        "azadirachta_indica": "data/raw/azadirachta_indica.gb",
        "arabidopsis_thaliana": "data/raw/arabidopsis_thaliana.gb",
        "oryza_sativa": "data/raw/oryza_sativa.gb",
        "glycine_max": "data/raw/glycine_max.gb",
        "nicotiana_tabacum": "data/raw/nicotiana_tabacum.gb",
        "spinacia_oleracea": "data/raw/spinacia_oleracea.gb"
    }

    # Load sequences and labels from all genomes
    all_seqs, all_labels = load_all_labeled_genomes(species_to_files, min_length=30)

    print(f"\n✅ Total sequences extracted: {len(all_seqs)}")
    os.makedirs("data/processed", exist_ok=True)

    # Save combined output
    np.save("data/processed/X_filtered.npy", np.array(all_seqs, dtype=object))
    np.save("data/processed/y_labels.npy", np.array(all_labels))
    print("✅ Saved to data/processed/X_filtered.npy and y_labels.npy")
