# dashboard/pages/model_comparison.py
from __future__ import annotations
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table, callback, Input, Output

from dashboard.services.results_loader import (
    load_dashboard_cfg,
    load_local_metrics,
    metrics_to_frame,
    export_dataframe,
)
from dashboard.services.mlflow_client import list_runs
from dashboard.components.run_picker import build_run_filters

dash.register_page(__name__, path="/models", name="Model Comparison")


# ---- helpers ----------------------------------------------------------------

def _combined_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load local metrics + MLflow runs and return (combined_df, mlflow_df)."""
    cfg = load_dashboard_cfg()

    # Local results -> tidy frame
    local_metrics = load_local_metrics(cfg["paths"]["metrics_dir"])
    local_df = metrics_to_frame(local_metrics)
    if local_df is None:
        local_df = pd.DataFrame()
    local_df = local_df.copy()
    if not local_df.empty:
        local_df["source"] = "local"

    # MLflow runs -> light frame; may be empty if no experiment yet
    mlf_df = list_runs()
    if not mlf_df.empty:
        mlf_df = mlf_df.copy()
        # Normalize common columns
        if "name" not in mlf_df.columns:
            mlf_df["name"] = mlf_df.get("run_id", "")
        # Promote key metrics if present
        rename_cols = {}
        if "metric.f1" in mlf_df.columns:
            rename_cols["metric.f1"] = "f1"
        if "metric.roc_auc" in mlf_df.columns:
            rename_cols["metric.roc_auc"] = "roc_auc"
        if rename_cols:
            mlf_df = mlf_df.rename(columns=rename_cols)
        mlf_df["source"] = "mlflow"

    # Combine on shared columns only (keeps table sane)
    if not local_df.empty and not mlf_df.empty:
        common = list(set(local_df.columns).intersection(mlf_df.columns))
        if not common:
            combined = local_df
        else:
            combined = pd.concat([local_df[common], mlf_df[common]], ignore_index=True)
    else:
        combined = local_df if not local_df.empty else mlf_df

    return combined.fillna(""), mlf_df.fillna("")


def _table_columns(df: pd.DataFrame) -> list[dict]:
    """Prefer a useful subset if available; else fall back to all columns."""
    preferred = [
        "name", "source",
        "f1", "roc_auc", "accuracy", "precision", "recall",
        "brier", "ece", "wall_time_s",
        "param.model", "param.encoding", "param.cnn_max_len",
        "tag.algo", "tag.branch",
        "start_time", "end_time",
    ]
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        cols = sorted(df.columns)
    return [{"name": c, "id": c} for c in cols]


# ---- page layout -------------------------------------------------------------

def layout_content():
    df, mlf = _combined_frame()
    filters = build_run_filters(mlf)  # handles empty MLflow gracefully

    table = dash_table.DataTable(
        id="cmp-table",
        columns=_table_columns(df),
        data=df.to_dict("records"),
        sort_action="native",
        filter_action="native",
        page_size=12,
        style_table={"overflowX": "auto"},
        style_cell={"fontSize": 13, "padding": "6px"},
    )

    dl_btn = dbc.Button("Download table (CSV)", id="cmp-dl", color="secondary", size="sm", className="mt-2")
    dl_path = html.Div(id="cmp-dl-path", className="small text-muted mt-1")

    empty_state = dbc.Alert(
        "No results found yet. Train a model (e.g., `make train-classical`) or ensure MLflow points to your run store.",
        color="info",
        className="mt-3",
    ) if df.empty else None

    return dbc.Container(
        [
            html.H3("Model Comparison"),
            filters,
            html.Hr(),
            table,
            dl_btn,
            dl_path,
            empty_state,
        ],
        fluid=True,
        className="py-3",
    )

layout = layout_content()


# ---- callbacks ---------------------------------------------------------------

@callback(Output("cmp-dl-path", "children"), Input("cmp-dl", "n_clicks"), prevent_initial_call=True)
def _download_table(_):
    df, _ = _combined_frame()
    if df.empty:
        return "Nothing to export yet."
    path = export_dataframe(df, "results/reports/model_comparison.csv")
    return f"Saved → {path}"
