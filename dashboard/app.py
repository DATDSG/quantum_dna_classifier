from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import dash
import dash_bootstrap_components as dbc
import yaml
from dash import Dash, dcc, html

# --- Ensure project root on sys.path so `dashboard.*` resolves on Windows/Linux ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- Load optional dashboard config ---
CFG_PATH = ROOT / "configs" / "dashboard.yaml"
if CFG_PATH.exists():
    DASHCFG = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
else:
    DASHCFG = {
        "host": "127.0.0.1",
        "port": 8050,
        "debug": False,
        "paths": {
            "results_dir": "results",
            "metrics_dir": "results/metrics",
            "plots_dir": "results/plots",
            "reports_dir": "results/reports",
            "mlruns_dir": "results/logs/mlruns",
        },
    }

# Environment overrides (handy on Windows PowerShell/CMD)
HOST = os.environ.get("DASH_HOST", DASHCFG.get("host", "127.0.0.1"))
PORT = int(os.environ.get("DASH_PORT", str(DASHCFG.get("port", 8050))))
DEBUG = bool(DASHCFG.get("debug", False))

# --- Dash app (dark theme is nice; keep Bootstrap if you prefer light) ---
external_stylesheets = [dbc.themes.BOOTSTRAP]  # swap to dbc.themes.CYBORG for dark
app: Dash = dash.Dash(
    __name__,
    use_pages=True,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True,  # allow page-specific callbacks to register later
    pages_folder=str((Path(__file__).parent / "pages").resolve()),
    assets_folder=str((Path(__file__).parent / "assets").resolve()),
    title="Neem DNA — QML Dashboard",
)

def _make_sidebar() -> dbc.Col:
    """
    Build a vertical nav from whatever pages are registered.
    We try a preferred order; anything else appears at the end.
    """
    # Force import of pages by touching page registry (Dash loads them at init)
    reg = dash.page_registry  # {module: {path, name, ...}}
    # Preferred path order if present
    preferred: List[str] = ["/", "/data", "/models", "/calibration", "/quantum", "/train", "/outcomes"]
    existing_paths = {info["path"] for info in reg.values()}
    ordered = [p for p in preferred if p in existing_paths]
    # Append any remaining pages (excluding the 404)
    remaining = [info["path"] for info in reg.values()
                 if info["path"] not in ordered and info["path"] != "/_dash-pages-status"]
    ordered += sorted(remaining)

    nav_links = [
        dbc.NavLink(next((info["name"] for info in reg.values() if info["path"] == path), path),
                    href=path, active="exact")
        for path in ordered
    ]
    return dbc.Col(
        dbc.Nav(nav_links, vertical=True, pills=True),
        md=2,
    )

app.layout = dbc.Container(
    [
        dbc.NavbarSimple(
            brand="Neem DNA — QML Dashboard",
            brand_href="/",
            color="dark",
            dark=True,
            className="mb-3",
        ),
        dbc.Row(
            [
                _make_sidebar(),
                dbc.Col(dash.page_container, md=10),
            ],
            className="g-3",
        ),
    ],
    fluid=True,
)

def main() -> None:
    # Dash 2.x: app.run or run_server; both fine. Keep run_server for clarity.
    app.run_server(host=HOST, port=PORT, debug=DEBUG)

if __name__ == "__main__":
    main()
