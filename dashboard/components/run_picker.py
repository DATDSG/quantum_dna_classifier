from __future__ import annotations
from typing import List
import dash_bootstrap_components as dbc
from dash import html, dcc
import pandas as pd

def multi_select(label: str, id_: str, options: List[str]):
    return dbc.Col(
        [
            html.Small(label),
            dcc.Dropdown(id=id_, options=[{"label": o, "value": o} for o in options], multi=True, clearable=True),
        ],
        md=3,
    )

def build_run_filters(df: pd.DataFrame):
    # derive simple facets from MLflow tags/params if present
    families = sorted(set(df.get("tag.branch", []))) or []
    encs = sorted(set(df.get("param.encoding", []))) or []
    datasets = sorted(set(df.get("tag.dataset", []))) or []
    seeds = sorted(set(df.get("param.seed", []))) or []
    return dbc.Row(
        [
            multi_select("Model family (tag.branch)", "pick-family", families),
            multi_select("Encoding (param.encoding)", "pick-enc", encs),
            multi_select("Dataset (tag.dataset)", "pick-ds", datasets),
            multi_select("Seed (param.seed)", "pick-seed", seeds),
        ],
        className="g-2",
    )
