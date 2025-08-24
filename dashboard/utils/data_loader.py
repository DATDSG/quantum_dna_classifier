from __future__ import annotations
from pathlib import Path
import json

def load_metrics(base: str = "results/metrics") -> dict:
    """Load all JSON metrics under results/metrics into a dict keyed by filename stem."""
    p = Path(base); p.mkdir(parents=True, exist_ok=True)
    out = {}
    for f in p.glob("*.json"):
        try:
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Also expose aggregate if nested under 'aggregate.json'
    if "aggregate" in out and isinstance(out["aggregate"], dict):
        out.update(out["aggregate"])
    return out