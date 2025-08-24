from __future__ import annotations
import dash_bootstrap_components as dbc
from dash import html, dcc

def date_filter(id_prefix: str = "flt"):
    return dbc.Row(
        [
            dbc.Col([html.Small("Date range"), dcc.DatePickerRange(id=f"{id_prefix}-date")], md=4),
            dbc.Col([html.Small("Search"), dbc.Input(id=f"{id_prefix}-search", placeholder="tag:model, encoding, seed...")], md=4),
        ],
        className="g-2",
    )
