from __future__ import annotations
from pathlib import Path
import json
import dash
import dash_bootstrap_components as dbc
from dash import html
from dashboard.services.results_loader import load_dashboard_cfg, load_local_metrics

dash.register_page(__name__, path="/quantum", name="Quantum Stats")
CFG = load_dashboard_cfg()
MET = load_local_metrics(CFG["paths"]["metrics_dir"])

def metric_card(label: str, value):
    return dbc.Col(
        dbc.Card(dbc.CardBody([html.Div(label, className="metric-label"),
                               html.H4(str(value if value is not None else "—"), className="metric-value")])))

def pick_quantum_payload(metrics: dict) -> dict:
    for key in ("qsvm", "vqc"):
        p = metrics.get(key)
        if p: return p
    for _, p in metrics.items():
        if any(k in p for k in ("circuit_depth","two_qubit_gates","shots","n_features","kernel_cond_train")):
            return p
    return {}

def layout_content():
    p = pick_quantum_payload(MET)
    cards = dbc.Row(
        [
            metric_card("Circuit depth", p.get("circuit_depth")),
            metric_card("Two-qubit gates", p.get("two_qubit_gates")),
            metric_card("Shots", p.get("shots")),
            metric_card("Feature dim", p.get("n_features")),
            metric_card("Kernel cond (train)", p.get("kernel_cond_train")),
        ],
        className="g-3",
    )
    info = dbc.Alert(
        "QSVM/VQC scripts log quantum stats and kernel condition numbers. "
        "Run `make train-quantum-qsvm` / `make train-quantum-vqc` to populate.",
        color="info",
    )
    return dbc.Container([html.H3("Quantum Stats"), cards, html.Hr(), info], fluid=True, className="py-3")

layout = layout_content()
