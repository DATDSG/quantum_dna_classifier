from __future__ import annotations
import numpy as np
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output

from dashboard.services.results_loader import load_dashboard_cfg, load_local_metrics, metrics_to_frame, export_dataframe

dash.register_page(__name__, path="/outcomes", name="Outcomes")

CFG = load_dashboard_cfg()
METRICS = load_local_metrics(CFG["paths"]["metrics_dir"])
DF = metrics_to_frame(METRICS)

def _bootstrap_ci(arr: np.ndarray, iters: int = 1000, seed: int = 42, alpha: float = 0.95):
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(iters):
        s = rng.choice(arr, size=arr.size, replace=True)
        stats.append(np.nanmean(s))
    lo = np.percentile(stats, (1-alpha)/2*100)
    hi = np.percentile(stats, (1+(alpha))/2*100)
    return (float(np.nanmean(arr)), float(lo), float(hi))

def layout_content():
    if DF.empty:
        return dbc.Container([html.H3("Outcomes"), dbc.Alert("No metrics yet.", color="warning")], fluid=True, className="py-3")
    # basic “Top N by ROC-AUC”
    top = DF.sort_values("roc_auc", ascending=False).head(5)
    mean_auc, lo, hi = _bootstrap_ci(DF["roc_auc"].dropna().values, iters=CFG["limits"].get("bootstrap_iters", 1000), seed=CFG["limits"].get("seed",42))
    head = dbc.Alert(f"Overall ROC-AUC mean={mean_auc:.3f} [{lo:.3f},{hi:.3f}] (bootstrap CI)", color="secondary")
    table = dash.dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in top.columns],
        data=top.to_dict("records"),
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": 13, "padding": "6px"},
    )
    btn = dbc.Button("Export Top-5 (CSV)", id="out-dl", color="secondary", size="sm", className="mt-2")
    msg = html.Div(id="out-dl-path", className="small text-muted mt-1")
    return dbc.Container([html.H3("Outcomes"), head, table, btn, msg], fluid=True, className="py-3")

@callback(Output("out-dl-path","children"), Input("out-dl","n_clicks"), prevent_initial_call=True)
def _dl(_):
    top = DF.sort_values("roc_auc", ascending=False).head(5)
    path = export_dataframe(top, "results/reports/top5.csv")
    return f"Saved → {path}"

layout = layout_content()
