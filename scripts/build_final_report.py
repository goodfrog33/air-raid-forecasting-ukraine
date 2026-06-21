"""Assemble reports/final_report.md from the artifacts produced by the pipeline.

Reproducible: re-run after training to refresh the report with the latest
numbers. Also patches the README's results placeholder with the headline table.

    python scripts/build_final_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _load(name: str) -> dict | None:
    p = REPORTS / name
    return json.loads(p.read_text()) if p.exists() else None


def _ranking_table(ranking: list[dict], metrics: list[str]) -> str:
    cols = ["rank", "model", "family"] + metrics
    head = "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for r in ranking:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return head + "\n" + "\n".join(rows)


def main() -> None:
    eda = _load("eda_summary.json") or {}
    count = _load("count_national_ranking.json")
    proba = _load("proba_region_ranking.json")
    metrics = _load("metrics_summary.json") or {}
    explain = _load("production_count_explainability.json") or {}

    L: list[str] = []
    L.append("# Final Report — Air Raid Alert Forecasting in Ukraine\n")
    L.append("_A miniature defense-analytics platform. Auto-generated from pipeline artifacts._\n")

    # 1. Data
    L.append("## 1. Data\n")
    if eda:
        d = eda
        L.append(f"- **Events:** {d.get('n_events'):,} consolidated oblast-level alerts "
                 f"across **{d.get('n_regions')}** regions.")
        L.append(f"- **Span:** {d.get('date_min')} → {d.get('date_max')}.")
        L.append(f"- **Average alerts/day:** {d.get('avg_alerts_per_day')}.")
        dur = d.get("duration_minutes", {})
        L.append(f"- **Duration (min):** mean {dur.get('mean')}, median {dur.get('median')}, "
                 f"p90 {dur.get('p90')}, p99 {dur.get('p99')}.")
        top = d.get("top_regions", {})
        if top:
            L.append("- **Most-affected regions:** "
                     + ", ".join(f"{k} ({v:,})" for k, v in list(top.items())[:6]) + ".")
    L.append("\nSource: Vadimkin *ukrainian-air-raid-sirens-dataset* (live download, UTC). "
             "Provenance recorded in `data/raw/_manifest.json`.\n")

    # 2. EDA highlights
    L.append("## 2. EDA highlights\n")
    if eda.get("stationarity"):
        s = eda["stationarity"]
        L.append(f"- **Stationarity:** ADF p={s.get('adf', {}).get('pvalue')}, "
                 f"KPSS p={s.get('kpss', {}).get('pvalue')} (classic trend-stationary signature).")
    if eda.get("decomposition"):
        dc = eda["decomposition"]
        L.append(f"- **STL strengths:** seasonal {dc.get('seasonal_strength'):.3f}, "
                 f"trend {dc.get('trend_strength'):.3f}.")
    L.append("- Figures in `reports/figures/` (daily series, hour×weekday heatmap, regional "
             "intensity, duration distribution, STL decomposition, ACF/PACF).\n")

    # 3. Forecasting targets
    L.append("## 3. Forecasting targets\n")
    L.append("A: alert count (next 1/6/24h) · B: alert probability (next 1/6/24h) · "
             "C: expected duration · D: severity (Low/Medium/High/Critical, derived from "
             "duration quantiles).\n")

    # 4. Model comparison (count)
    L.append("## 4. Model comparison — national hourly count (1-step-ahead)\n")
    if count:
        L.append(f"**Best model: `{count['best_model']}`** (by {count['primary_metric']}, "
                 "expanding-window backtest).\n")
        L.append(_ranking_table(count["ranking"], ["MAE", "RMSE", "SMAPE"]))
    else:
        L.append("_Not available — run training._")
    L.append("")

    # 5. Classification comparison
    L.append("## 5. Model comparison — per-region P(alert within 6h)\n")
    if proba:
        L.append(f"**Best model: `{proba['best_model']}`** (by {proba['primary_metric']}).\n")
        L.append(_ranking_table(proba["ranking"], ["ROC_AUC", "F1", "Accuracy", "LogLoss"]))
    else:
        L.append("_Not available — run training._")
    L.append("")

    # 6. Optimization
    L.append("## 6. Iterative optimization\n")
    if metrics.get("tuning"):
        t = metrics["tuning"]
        L.append(f"- Tracked LightGBM search → best CV MAE **{t.get('best_mae'):.4f}** "
                 f"(naive baseline {t.get('naive_baseline_mae')}).")
        L.append(f"- Best params: `{t.get('best_params')}`")
        L.append("- Full history in `reports/experiments.jsonl`.\n")

    # 7. Production model + backtest
    L.append("## 7. Production model\n")
    L.append("A global, region-aware gradient-boosted model serves any region & horizon for "
             "Targets A & B; auxiliary models cover duration (C) and severity (D).")
    if metrics.get("production_count_backtest"):
        L.append("\n**Production region count (H=1) backtest:**\n```")
        L.append(json.dumps(metrics["production_count_backtest"], indent=2))
        L.append("```")
    if metrics.get("severity_metrics"):
        L.append(f"\n**Severity classifier:** {json.dumps(metrics['severity_metrics'])}\n")

    # 8. Explainability
    L.append("## 8. Explainability\n")
    top = explain.get("shap_top") or explain.get("feature_importance_top") or {}
    if top:
        L.append("Top drivers (mean |SHAP| / importance):")
        for k, v in list(top.items())[:10]:
            L.append(f"- `{k}`: {v}")
    L.append("\nSee `reports/figures/production_count_shap_summary.png`.\n")

    # 9. Deliverables
    L.append("## 9. Deliverables\n")
    L.append("Source package, 6 notebooks, multi-model pipeline, backtesting & comparison "
             "frameworks, FastAPI service, Streamlit dashboard, pytest suite, Dockerized "
             "deployment, and this report.\n")
    L.append("> **Ethical note:** analytical use on open historical data only — not an "
             "early-warning system.\n")

    out = REPORTS / "final_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("wrote", out)

    # Patch README results placeholder with the headline count table.
    readme = ROOT / "README.md"
    if readme.exists() and count:
        txt = readme.read_text()
        block = (f"**Best national count model: `{count['best_model']}`** "
                 f"(expanding-window backtest, primary metric {count['primary_metric']}).\n\n"
                 + _ranking_table(count["ranking"], ["MAE", "RMSE", "SMAPE"]) + "\n")
        if "<!-- RESULTS_PLACEHOLDER -->" in txt:
            txt = txt.replace("<!-- RESULTS_PLACEHOLDER -->", block)
            readme.write_text(txt)
            print("patched README results")


if __name__ == "__main__":
    main()
