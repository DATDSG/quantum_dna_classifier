from __future__ import annotations
import argparse, json, yaml, os
from pathlib import Path
import plotly.express as px

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-config", required=True, help="Path to configs/evaluation.yaml")
    ap.add_argument("--html-only", action="store_true",
                    help="Skip PNG export to avoid kaleido hangs on some Windows setups.")
    ap.add_argument("--precision", type=int, default=3, help="Decimal places for F1.")
    args = ap.parse_args()

    ev = yaml.safe_load(open(args.eval_config, "r", encoding="utf-8"))
    plots_dir = Path(ev["save_dir"]["plots"]); plots_dir.mkdir(parents=True, exist_ok=True)

    agg_path = Path(ev["save_dir"]["metrics"]) / "aggregate.json"
    if not agg_path.exists():
        print("No aggregate.json yet. Run evaluate first.")
        return

    data = json.loads(agg_path.read_text(encoding="utf-8"))
    results = data.get("results", {})

    # Collect F1s from various models
    bars = []
    for k, v in results.items():
        if isinstance(v, dict) and "f1" in v:
            bars.append((k, v["f1"]))
        elif isinstance(v, dict) and "svm_rbf" in v and "f1" in v["svm_rbf"]:
            bars.append(("svm_rbf", v["svm_rbf"]["f1"]))
        elif isinstance(v, dict) and "cnn" in v and "f1" in v["cnn"]:
            bars.append(("cnn", v["cnn"]["f1"]))

    if not bars:
        print("[WARN] No F1 scores found in aggregate.json.")
        return

    bars.sort(key=lambda t: t[1], reverse=True)
    fig = px.bar(
        x=[b[0] for b in bars],
        y=[round(float(b[1]), args.precision) for b in bars],
        title="F1 by model",
        labels={"x": "Model", "y": f"F1 (rounded to {args.precision}dp)"},
    )
    html_path = plots_dir / "f1_bar.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"[OK] Wrote {html_path}")

    # PNG export (best-effort)
    if not args.html_only and os.getenv("REPORT_HTML_ONLY", "").strip().lower() not in {"1", "true", "yes"}:
        try:
            import plotly.io as pio  # ensure kaleido reachable
            png_path = plots_dir / "f1_bar.png"
            fig.write_image(str(png_path), scale=2)
            print(f"[OK] Wrote {png_path}")
        except Exception as e:
            print(f"[WARN] PNG export failed ({e}). Re-run with --html-only to skip PNG.")

    print("[OK] Plots exported to", plots_dir)

if __name__ == "__main__":
    main()
