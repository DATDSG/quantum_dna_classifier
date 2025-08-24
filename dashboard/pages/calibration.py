from __future__ import annotations
import numpy as np
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

from dashboard.services.results_loader import load_dashboard_cfg, load_local_metrics
from dashboard.components.charts import reliability_curve

dash.register_page(__name__, path="/calibration", name="Calibration & Thresholding")
CFG = load_dashboard_cfg()
METRICS = load_local_metrics(CFG["paths"]["metrics_dir"])

def layout_content():
    # heuristic: pick any available preds saved by pipelines
    # users may extend to choose a specific run
    try:
        import glob
        import json
        import numpy as np
        # attempt to find CNN / QSVM / VQC val preds
        cand = []
        cand += glob.glob("models/cnn/val_proba_*.npy")
        cand += glob.glob("models/qsvm/val_proba_*.npy")
        cand += glob.glob("models/vqc/val_proba_*.npy")
        cand = sorted(cand)[-1:]
        proba = np.load(cand[0]) if cand else np.array([])
        # map labels using stored val indices
        idx_path = cand[0].replace("val_proba_", "val_idx_").replace(".npy",".json") if cand else ""
        if idx_path:
            j = json.loads(open(idx_path, "r", encoding="utf-8").read())
            import numpy as np
            y = np.load("data/interim/labels.npy")[np.array(j["val"], dtype=int)]
        else:
            y = np.array([])
    except Exception:
        proba, y = np.array([]), np.array([])

    if proba.size and y.size:
        tab, brier, ece = reliability_curve(proba, y, bins=10)
        fig = px.bar(tab, x="p_mean", y="y_mean", hover_data=["n"], labels={"p_mean":"Mean predicted p","y_mean":"Empirical"})
        head = dbc.Alert(f"Brier={brier:.4f} | ECE={ece:.4f}", color="secondary")
        return dbc.Container([html.H3("Calibration & Thresholding"), head, dcc.Graph(figure=fig)], fluid=True, className="py-3")

    msg = dbc.Alert("No saved validation predictions found. Train models to populate this page.", color="warning")
    return dbc.Container([html.H3("Calibration & Thresholding"), msg], fluid=True, className="py-3")

layout = layout_content()
