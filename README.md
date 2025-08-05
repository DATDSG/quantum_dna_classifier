# 🧬 Quantum DNA Classifier

This repository contains the full research project for **"Comparative Analysis of Quantum and Classical Machine Learning for DNA Sequence Classification: A Case Study on _Azadirachta indica_"**, developed as part of the final year BSc (Honours) in Computer Science at NSBM Green University.

## 📌 Project Overview

This project aims to benchmark classical and quantum machine learning (QML) models on real genomic data. The chloroplast genome of *Azadirachta indica* (Neem) is used as the dataset to evaluate classification performance between traditional models (SVM, CNN, RF) and quantum models (VQC, QSVM).

## 🚀 Features

- DNA preprocessing and sequence encoding (k-mer, one-hot, hybrid)
- Classical ML models: SVM, CNN, Random Forest (Scikit-learn, TensorFlow)
- Quantum models: QSVM (Qiskit), VQC (PennyLane)
- Model performance benchmarking: Accuracy, F1, AUC
- Bloch sphere and gate-depth visualizations
- MLflow/TensorBoard integration for tracking

## 📁 Project Structure

quantum_dna_classifier/
├── data/
├── notebooks/
├── scripts/
├── models/
├── results/
├── configs/
├── docs/
├── env/
├── tests/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── Makefile


## 🛠️ Tech Stack

- Python 3.11
- Scikit-learn, TensorFlow
- Qiskit, PennyLane
- Biopython, NumPy, Pandas
- MLflow, TensorBoard

## 📚 Dataset

- Source: NCBI GenBank
- Accession: [KF986530.1](https://www.ncbi.nlm.nih.gov/nuccore/KF986530.1)
- Format: FASTA + GenBank

## 📦 Installation

```bash
git clone https://github.com/yourusername/quantum_dna_classifier.git
cd quantum_dna_classifier
conda env create -f env/environment.yml
conda activate quantum-dna-env

make preprocess    # Encode and clean DNA data
make train         # Train classical and quantum models
make evaluate      # Benchmark & visualize results
