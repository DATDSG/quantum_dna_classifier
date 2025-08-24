from __future__ import annotations
import glob, json
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

dash.register_page(__name__, path="/train", name="Trainability")

def _load_cnn_history():
    cand = sorted(glob.glob("models/cnn/history_*.json"))
    if not cand: return None
    h = json.loads(open(cand[-1], "r", encoding="utf-8").read())
    return h

def _load_vqc_curve():
    cand = sorted(glob.glob("models/vqc/val_curve_*.json"))
    if not cand: return None
    return json.loads(open(cand[-1], "r", encoding="utf-8").read())

def layout_content():
    h = _load_cnn_history()
    vqc = _load_vqc_curve()
    figs = []
    if h:
        import pandas as pd
        df = pd.DataFrame(h)
        figs.append(dcc.Graph(figure=px.line(df, y=["loss","val_loss"], title="CNN training curves")))
    if vqc:
        import pandas as pd
        df = pd.DataFrame({"val_loss": vqc})
        figs.append(dcc.Graph(figure=px.line(df, y="val_loss", title="VQC validation loss")))
    if not figs:
        return dbc.Container([html.H3("Trainability"), dbc.Alert("No histories found. Train models to populate.", color="warning")], fluid=True, className="py-3")
    return dbc.Container([html.H3("Trainability"), *figs], fluid=True, className="py-3")

layout = layout_content()
