from __future__ import annotations
from pathlib import Path
import numpy as np
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dashboard.services.results_loader import load_windows_meta, load_labels, load_lengths

dash.register_page(__name__, path="/data", name="Data Overview")

def layout_content():
    meta = load_windows_meta()
    y = load_labels()
    lengths = load_lengths()
    n_pos = int((y == 1).sum()) if y.size else 0
    n_neg = int((y == 0).sum()) if y.size else 0
    pie = px.pie(names=["CDS (1)","r/tRNA (0)"], values=[n_pos, n_neg], title="Class balance")
    hist = px.histogram(x=lengths, nbins=30, title="Window length distribution (bp)") if lengths else None

    cards = dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.Div("Total windows", className="metric-label"),
                                       html.H4(str(meta.get("n","—")), className="metric-value")]))),
        dbc.Col(dbc.Card(dbc.CardBody([html.Div("Unique accessions", className="metric-label"),
                                       html.H4(str(len(meta.get("sources",[])) or "1"), className="metric-value")]))),
    ], className="g-3")

    graphs = []
    graphs.append(dcc.Graph(figure=pie))
    if hist is not None:
        graphs.append(dcc.Graph(figure=hist))

    return dbc.Container([html.H3("Data Overview"), cards, *graphs], fluid=True, className="py-3")

layout = layout_content()
