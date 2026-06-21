"""Model comparison, ranking and best-model selection (Phase 8).

Consumes the aggregate metric frame produced by the backtester and turns it into
a ranked comparison table, persisted as CSV + Markdown + JSON so it can be
embedded in the report and rendered by the dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from air_raid_forecasting.logging_utils import get_logger

log = get_logger(__name__)

# (metric, ascending?) — how to rank for each primary criterion.
RANK_DIRECTION = {
    "MAE": True, "RMSE": True, "SMAPE": True, "MAPE": True,
    "ROC_AUC": False, "F1": False, "Accuracy": False, "Precision": False,
    "Recall": False, "LogLoss": True, "Brier": True,
}


def rank_models(aggregate: pd.DataFrame, metric: str = "MAE") -> pd.DataFrame:
    if aggregate.empty:
        return aggregate
    ascending = RANK_DIRECTION.get(metric, True)
    out = aggregate.sort_values(metric, ascending=ascending).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def best_model(aggregate: pd.DataFrame, metric: str = "MAE") -> str:
    ranked = rank_models(aggregate, metric)
    return str(ranked.iloc[0]["model"])


def to_markdown(ranked: pd.DataFrame, metrics: list[str], title: str) -> str:
    cols = ["rank", "model", "family"] + [m for m in metrics if m in ranked.columns]
    view = ranked[cols].copy()
    for m in metrics:
        if m in view.columns:
            view[m] = view[m].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def save_comparison(
    aggregate: pd.DataFrame,
    per_fold: pd.DataFrame,
    out_dir: str | Path,
    name: str,
    primary_metric: str,
    metrics: list[str],
    title: str,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = rank_models(aggregate, primary_metric)

    ranked.to_csv(out_dir / f"{name}_ranking.csv", index=False)
    if not per_fold.empty:
        per_fold.to_csv(out_dir / f"{name}_per_fold.csv", index=False)
    md = to_markdown(ranked, metrics, title)
    (out_dir / f"{name}_ranking.md").write_text(md, encoding="utf-8")

    best = best_model(aggregate, primary_metric) if not aggregate.empty else None
    payload = {
        "title": title,
        "primary_metric": primary_metric,
        "best_model": best,
        "ranking": ranked.to_dict(orient="records"),
    }
    with open(out_dir / f"{name}_ranking.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    log.info("Saved comparison '%s' -> %s (best: %s)", name, out_dir, best)
    return payload
