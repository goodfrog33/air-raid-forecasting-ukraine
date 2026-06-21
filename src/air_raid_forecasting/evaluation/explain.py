"""Explainability for the production model (Phase 10).

Combines three complementary views:

* **Model feature importance** — fast, split/gain-based (tree models).
* **Permutation importance** — model-agnostic, measures the drop in score when a
  feature is shuffled (computed on a held-out sample).
* **SHAP** — per-feature contribution to individual predictions, summarized as
  mean |SHAP| and saved as a beeswarm/bar figure.

All three are persisted to ``reports/`` so the dashboard and report can render
them without recomputation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from air_raid_forecasting.logging_utils import get_logger

log = get_logger(__name__)


def model_feature_importance(forecaster) -> pd.Series | None:
    fi = getattr(forecaster, "feature_importance", None)
    return fi() if callable(fi) else None


def permutation_importance_df(forecaster, X: pd.DataFrame, y: np.ndarray,
                              n_repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    Xp = forecaster._prepare_X(X, fit=False)
    scoring = "roc_auc" if forecaster.task == "proba" else "neg_mean_absolute_error"
    yv = (y > 0).astype(int) if forecaster.task == "proba" else y
    result = permutation_importance(
        forecaster.model, Xp, yv, n_repeats=n_repeats, random_state=seed, scoring=scoring
    )
    return (
        pd.DataFrame({"feature": Xp.columns,
                      "importance": result.importances_mean,
                      "std": result.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def shap_summary(forecaster, X_sample: pd.DataFrame, fig_path: Path,
                 max_display: int = 20) -> pd.Series | None:
    """Compute SHAP values and save a summary bar plot; return mean|SHAP|."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    Xp = forecaster._prepare_X(X_sample, fit=False)
    try:
        explainer = shap.TreeExplainer(forecaster.model)
        values = explainer.shap_values(Xp)
    except Exception as exc:  # pragma: no cover
        log.warning("SHAP computation failed: %s", exc)
        return None
    if isinstance(values, list):  # classifier -> list per class; take positive class
        values = values[-1]
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=Xp.columns).sort_values(ascending=False)

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(values, Xp, plot_type="bar", max_display=max_display, show=False)
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", dpi=120)
    plt.close("all")
    log.info("SHAP summary figure -> %s", fig_path)
    return mean_abs


def run_explain(forecaster, X: pd.DataFrame, y: np.ndarray, reports_dir: str | Path,
                figures_dir: str | Path, tag: str = "production_count", sample: int = 5000,
                seed: int = 42) -> dict:
    reports_dir, figures_dir = Path(reports_dir), Path(figures_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    X_s, y_s = X.iloc[idx], np.asarray(y)[idx]

    out: dict = {"tag": tag}
    fi = model_feature_importance(forecaster)
    if fi is not None:
        out["feature_importance_top"] = fi.head(20).round(4).to_dict()

    try:
        perm = permutation_importance_df(forecaster, X_s, y_s, seed=seed)
        perm.to_csv(reports_dir / f"{tag}_permutation_importance.csv", index=False)
        out["permutation_importance_top"] = perm.head(15).set_index("feature")["importance"].round(5).to_dict()
    except Exception as exc:  # pragma: no cover
        log.warning("Permutation importance failed: %s", exc)

    shap_mean = shap_summary(forecaster, X_s, figures_dir / f"{tag}_shap_summary.png")
    if shap_mean is not None:
        out["shap_top"] = shap_mean.head(20).round(5).to_dict()

    with open(reports_dir / f"{tag}_explainability.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return out
