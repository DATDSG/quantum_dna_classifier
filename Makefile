SHELL := /bin/bash
PY := python -u

.PHONY: help setup data encode train-classical train-quantum-qsvm train-quantum-vqc evaluate report dashboard test clean deep-clean

help:
	@echo "Targets:"
	@echo "  setup                  -> bootstrap folders & quick env sanity"
	@echo "  data                   -> parse GenBank & build windows/labels"
	@echo "  encode                 -> build one-hot / k-mer / angle encodings"
	@echo "  train-classical        -> run classical SVM + CNN pipeline"
	@echo "  train-quantum-qsvm     -> run Qiskit QSVM pipeline"
	@echo "  train-quantum-vqc      -> run PennyLane VQC pipeline"
	@echo "  evaluate, report       -> aggregate metrics & export plots"
	@echo "  dashboard              -> launch Plotly Dash app"
	@echo "  test                   -> run unit tests"
	@echo "  clean                  -> remove intermediates (safe)"
	@echo "  deep-clean             -> remove all caches/metrics/logs (full reset)"

setup:
	bash env/setup.sh

data:
	$(PY) scripts/preprocess_genbank.py --config configs/data.yaml

encode:
	$(PY) scripts/encode_dna.py --encoding-config configs/encoding.yaml --data-config configs/data.yaml

train-classical:
	$(PY) scripts/classical_pipeline.py --models-config configs/classical_models.yaml --encoding-config configs/encoding.yaml --eval-config configs/evaluation.yaml --tracking configs/tracking.yaml

train-quantum-qsvm:
	$(PY) scripts/quantum_pipeline_qiskit.py --qconfig configs/quantum_qiskit.yaml --encoding-config configs/encoding.yaml --eval-config configs/evaluation.yaml --tracking configs/tracking.yaml

train-quantum-vqc:
	$(PY) scripts/quantum_pipeline_pennylane.py --qconfig configs/quantum_pennylane.yaml --encoding-config configs/encoding.yaml --eval-config configs/evaluation.yaml --tracking configs/tracking.yaml

evaluate:
	$(PY) scripts/evaluate.py --eval-config configs/evaluation.yaml --tracking configs/tracking.yaml

report:
	$(PY) scripts/visualize.py --eval-config configs/evaluation.yaml --html-only
	@echo "Figures → results/plots ; reports → results/reports"

dashboard:
	$(PY) -m dashboard.app

test:
	PYTHONPATH=. pytest -q

clean:
	rm -rf data/interim/* data/processed/* data/embeddings/* data/metadata/*
	rm -rf models/checkpoints/*
	find results -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} \; || true

deep-clean: clean
	rm -rf results/cache results/logs results/metrics
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .pytest_cache .ruff_cache .mypy_cache

