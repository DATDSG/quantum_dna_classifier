from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def roc_overlay(curves: list[Tuple[np.ndarray, np.ndarray]], labels: list[str]) -> go.Figure:
    fig = go.Figure()
    for (fpr, tpr), name in zip(curves, labels):
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=name))
    fig.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash", width=1))
    fig.update_layout(title="ROC overlay", xaxis_title="FPR", yaxis_title="TPR")
    return fig

def pr_overlay(curves: list[Tuple[np.ndarray, np.ndarray]], labels: list[str]) -> go.Figure:
    fig = go.Figure()
    for (rec, prec), name in zip(curves, labels):
        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=name))
    fig.update_layout(title="PR overlay", xaxis_title="Recall", yaxis_title="Precision")
    return fig

def reliability_curve(proba: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[pd.DataFrame, float, float]:
    """Return calibration table + Brier score + ECE."""
    proba = proba.ravel()
    y = y.ravel()
    bins = np.clip(bins, 5, 50)
    edges = np.linspace(0, 1, bins + 1)
    inds = np.digitize(proba, edges) - 1
    df = pd.DataFrame({"p": proba, "y": y, "bin": inds})
    tab = df.groupby("bin").agg(p_mean=("p","mean"), y_mean=("y","mean"), n=("y","count")).reset_index()
    tab = tab.replace([np.inf, -np.inf], np.nan).dropna()
    brier = float(((proba - y) ** 2).mean())
    # ECE (equal width bins)
    ece = float((tab["n"] * (tab["p_mean"] - tab["y_mean"]).abs()).sum() / tab["n"].sum())
    return tab, brier, ece
