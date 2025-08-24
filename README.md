# Hybrid Quantum–Classical DNA Classification (Neem Chloroplast)

Binary classification of chloroplast DNA windows (CDS vs rRNA/tRNA) centered on *Azadirachta indica* (GenBank KF986530.1). Reproducible, GPU-accelerated, and research-grade: classical (RBF-SVM, compact 1D-CNN) and quantum (QSVM, VQC) with device-realistic noise, MLflow/TensorBoard tracking, unit tests, and a modern Plotly Dash dashboard.

## Why this repo
- **Reproducibility**: pinned env (Conda/Docker), deterministic seeds, saved splits, MLflow/TensorBoard.
- **Dual pipelines**: Classical baseline + quantum heads on identical folds.
- **Robustness**: QASM Aer noise (readout/depolarizing), mitigation stubs, CIs via bootstrap/k-fold.
- **Interpretability**: CNN saliency; single-qubit Bloch projections; circuit depth/gate/shot stats.

## Quickstart

### 0) Clone & prepare
```bash
git clone https://github.com/DATDSG/quantum_dna_classifier.git
cd quantum_dna_classifier
