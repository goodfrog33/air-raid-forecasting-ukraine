"""Pipeline step 5: train, backtest, compare, optimize, explain, persist.

    python -m air_raid_forecasting.pipeline.run_train [--stage all|national|production]
                                                      [--fast] [--no-region-proba]

The work is split into two stages that can run as separate (shorter, lower-
memory) jobs and share state through small JSON handoff files:

* ``national``   — stages A+B: the headline multi-model count backtest +
  LightGBM hyperparameter search (uses torch for the LSTM).
* ``production`` — stages C-H: per-region classification backtest, global
  production models (count/proba/duration/severity), explainability, the
  future forecast and the deployable model bundle.

Outputs land in ``reports/`` and ``models/`` (see the module-level docs of each).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from air_raid_forecasting.config import load_config
from air_raid_forecasting.evaluation.backtest import backtest_model, backtest_models, make_folds
from air_raid_forecasting.evaluation.compare import save_comparison
from air_raid_forecasting.evaluation.experiments import ExperimentTracker
from air_raid_forecasting.evaluation.explain import run_explain
from air_raid_forecasting.logging_utils import get_logger
from air_raid_forecasting.models.advanced import ProphetForecaster
from air_raid_forecasting.models.auxiliary import DurationModel, SeverityModel
from air_raid_forecasting.models.base import ModelContext
from air_raid_forecasting.models.ml import LightGBMForecaster
from air_raid_forecasting.models.persistence import ModelBundle
from air_raid_forecasting.models.registry import build_count_models, build_model, build_proba_models

log = get_logger(__name__)
NATIONAL_HANDOFF = "_national_summary.json"


def _ctx(cfg, meta, value_col, task, target_col, categorical):
    return ModelContext(
        value_col=value_col, feature_cols=meta["feature_cols"], categorical_cols=categorical,
        seasonal_period=cfg.modeling.seasonal_period_hours,
        timeout_s=cfg.modeling.per_model_timeout_seconds, seed=cfg.project.random_seed,
        task=task, target_col=target_col,
    )


def _folds(cfg, timestamps, n_folds):
    return make_folds(
        timestamps, scheme=cfg.backtest.scheme, n_folds=n_folds,
        test_horizon=cfg.backtest.test_horizon, min_train=cfg.backtest.min_train,
        step=cfg.backtest.step, gap_hours=cfg.backtest.gap_hours,
    )


def tune_lightgbm_national(national, ctx_count, folds, tracker) -> dict:
    """Small tracked hyperparameter search for LightGBM on the national series."""
    grids = [
        {"n_estimators": 300, "num_leaves": 31, "learning_rate": 0.05},
        {"n_estimators": 600, "num_leaves": 63, "learning_rate": 0.03},
        {"n_estimators": 800, "num_leaves": 127, "learning_rate": 0.02, "min_child_samples": 50},
        {"n_estimators": 500, "num_leaves": 63, "learning_rate": 0.05, "colsample_bytree": 0.7},
    ]
    best, best_mae = None, np.inf
    for i, params in enumerate(grids):
        model = LightGBMForecaster(params=params)
        model.name = f"lightgbm_tune_{i}"
        res = backtest_model(model, national, ctx_count, folds)
        mae = res["aggregate"].get("MAE", np.inf)
        tracker.log(model.name, params, res["aggregate"], tags={"task": "count", "series": "national"})
        log.info("  tune %d MAE=%.4f params=%s", i, mae, params)
        if mae < best_mae:
            best, best_mae = params, mae
    return {"best_params": best, "best_mae": float(best_mae)}


# Models skipped in --fast mode (slowest, and they don't change the headline).
FAST_SKIP_MODELS = {"lstm", "sarima"}


def train_production_supervised(cfg, region, meta_reg, target_col, task, fast=False):
    model = build_model(cfg.production.model, cfg)
    if fast and isinstance(model, LightGBMForecaster):
        model.params = {"n_estimators": 200}
    model.task = task
    ctx = _ctx(cfg, meta_reg, value_col="any_alert" if task == "proba" else "alerts_started",
               task=task, target_col=target_col, categorical=meta_reg["categorical_cols"])
    sub = region[region[target_col].notna()]
    model.fit(sub, ctx)
    return model


# --------------------------------------------------------------------------- #
# Stage: national (A + B)
# --------------------------------------------------------------------------- #
def run_national_stage(cfg, n_folds, tracker, fast=False) -> dict:
    proc = Path(cfg.paths.processed_dir)
    reports = Path(cfg.paths.reports_dir)
    national = pd.read_parquet(proc / "features_national.parquet")
    national["timestamp"] = pd.to_datetime(national["timestamp"], utc=True)
    meta_nat = json.loads((proc / "features_meta.json").read_text())["national"]

    log.info("=== A) National hourly count — multi-model backtest ===")
    ctx_count = _ctx(cfg, meta_nat, value_col="alerts_started", task="count",
                     target_col=None, categorical=[])
    folds = _folds(cfg, national["timestamp"], n_folds)
    log.info("Backtest folds: %d (scheme=%s, test=%s)%s", len(folds), cfg.backtest.scheme,
             cfg.backtest.test_horizon, " [fast]" if fast else "")
    models = build_count_models(cfg)
    if fast:
        models = [m for m in models if m.name not in FAST_SKIP_MODELS]
    res = backtest_models(models, national, ctx_count, folds)
    for _, row in res["aggregate"].iterrows():
        tracker.log(row["model"], {"default": True},
                    {k: row[k] for k in ("MAE", "RMSE", "SMAPE") if k in row},
                    tags={"task": "count", "series": "national"})
    comp = save_comparison(res["aggregate"], res["per_fold"], reports, "count_national",
                           "MAE", ["MAE", "RMSE", "SMAPE", "MAPE"],
                           "National hourly alert-count — 1-step-ahead model comparison")
    log.info("Best count model (national): %s", comp["best_model"])

    log.info("=== B) Optimization — LightGBM hyperparameter search (tracked) ===")
    tune = tune_lightgbm_national(national, ctx_count, folds, tracker)
    naive_mae = (float(res["aggregate"].set_index("model").loc["naive", "MAE"])
                 if "naive" in set(res["aggregate"]["model"]) else None)

    summary = {"count_comparison": comp,
               "tuning": {**tune, "naive_baseline_mae": naive_mae}}
    (reports / NATIONAL_HANDOFF).write_text(json.dumps(summary, indent=2, default=str))
    return summary


# --------------------------------------------------------------------------- #
# Stage: production (C - H)
# --------------------------------------------------------------------------- #
def run_production_stage(cfg, n_folds_region, args, tracker) -> dict:
    proc = Path(cfg.paths.processed_dir)
    reports = Path(cfg.paths.reports_dir)
    figures = Path(cfg.paths.figures_dir)
    models_dir = Path(cfg.paths.models_dir)

    region = pd.read_parquet(proc / "features_region.parquet")
    region["timestamp"] = pd.to_datetime(region["timestamp"], utc=True)
    events = pd.read_parquet(proc / "alerts_events_labeled.parquet")
    meta = json.loads((proc / "features_meta.json").read_text())
    meta_reg = meta["region"]
    summary: dict = {}

    if not args.no_region_proba:
        log.info("=== C) Per-region P(alert within 6h) — classification backtest ===")
        ctx_proba = _ctx(cfg, meta_reg, value_col="any_alert", task="proba",
                         target_col="target_any_6h", categorical=meta_reg["categorical_cols"])
        folds_reg = _folds(cfg, region["timestamp"], n_folds_region)
        res_proba = backtest_models(build_proba_models(cfg), region, ctx_proba, folds_reg,
                                    target_col="target_any_6h")
        for _, row in res_proba["aggregate"].iterrows():
            tracker.log(row["model"], {"default": True},
                        {k: row[k] for k in ("ROC_AUC", "F1", "Accuracy", "LogLoss") if k in row},
                        tags={"task": "proba", "series": "region", "window": 6})
        summary["proba_comparison"] = save_comparison(
            res_proba["aggregate"], res_proba["per_fold"], reports, "proba_region", "ROC_AUC",
            ["ROC_AUC", "F1", "Accuracy", "Precision", "Recall", "LogLoss"],
            "Per-region P(alert within 6h) — classification model comparison")

    log.info("=== D) Train production models on the full region panel ===")
    fast = getattr(args, "fast", False)
    prod_count = {H: train_production_supervised(cfg, region, meta_reg, f"target_count_{H}h", "count", fast)
                  for H in cfg.targets.count_horizons_hours}
    log.info("  trained %d production count models", len(prod_count))
    prod_proba = {H: train_production_supervised(cfg, region, meta_reg, f"target_any_{H}h", "proba", fast)
                  for H in cfg.targets.proba_windows_hours}
    log.info("  trained %d production proba models", len(prod_proba))

    duration_model = DurationModel(seed=cfg.project.random_seed).fit(events)
    sev_labels = cfg.targets.severity.labels
    ev_sorted = events.sort_values("started_at")
    cut = int(len(ev_sorted) * 0.8)
    sev_model = SeverityModel(sev_labels, seed=cfg.project.random_seed).fit(ev_sorted.iloc[:cut])
    sev_pred = sev_model.predict(ev_sorted.iloc[cut:])
    from sklearn.metrics import accuracy_score, f1_score
    summary["severity_metrics"] = {
        "accuracy": float(accuracy_score(ev_sorted.iloc[cut:]["severity"].astype(str), sev_pred)),
        "f1_macro": float(f1_score(ev_sorted.iloc[cut:]["severity"].astype(str), sev_pred,
                                   average="macro", labels=sev_labels, zero_division=0)),
    }
    sev_model.fit(events)  # refit on all data for production
    log.info("  severity classifier: %s", summary["severity_metrics"])

    log.info("=== E) Production region count (H=1) backtest ===")
    prod_bt = build_model(cfg.production.model, cfg)
    prod_bt.task = "count"
    ctx_prod = _ctx(cfg, meta_reg, value_col="alerts_started", task="count",
                    target_col="target_count_1h", categorical=meta_reg["categorical_cols"])
    res_prod = backtest_model(prod_bt, region, ctx_prod, _folds(cfg, region["timestamp"],
                              min(2, n_folds_region)), target_col="target_count_1h")
    summary["production_count_backtest"] = res_prod["aggregate"]
    tracker.log("production_lightgbm_region_count_1h", {"horizon": 1}, res_prod["aggregate"],
                tags={"task": "count", "series": "region"})

    log.info("=== F) Explainability (SHAP + permutation + importance) ===")
    sub1 = region[region["target_count_1h"].notna()]
    explain = run_explain(prod_count[1], sub1, sub1["target_count_1h"].to_numpy(),
                          reports, figures, tag="production_count", sample=5000,
                          seed=cfg.project.random_seed)
    summary["explainability_top"] = explain.get("shap_top") or explain.get("feature_importance_top")

    log.info("=== G) Future forecast (Prophet) for the dashboard ===")
    try:
        national = pd.read_parquet(proc / "features_national.parquet")
        national["timestamp"] = pd.to_datetime(national["timestamp"], utc=True)
        prophet = ProphetForecaster()
        prophet.fit(national, _ctx(cfg, meta["national"], "alerts_started", "count", None, []))
        prophet.forecast_future(periods=72, freq="h").to_parquet(
            proc / "forecast_national.parquet", index=False)
        log.info("  saved 72h national forecast")
    except Exception as exc:  # pragma: no cover
        log.warning("Future forecast skipped: %s", exc)

    log.info("=== H) Assemble & persist model bundle ===")
    national_summary = {}
    handoff = reports / NATIONAL_HANDOFF
    if handoff.exists():
        national_summary = json.loads(handoff.read_text())
    summary.update(national_summary)
    best_count = (national_summary.get("count_comparison", {}) or {}).get("best_model") \
        or cfg.production.model

    latest = (region.sort_values("timestamp").groupby("region", as_index=False).tail(1)
              .reset_index(drop=True))
    latest_features = latest[["region", "timestamp"] + meta_reg["feature_cols"]].copy()

    bundle = ModelBundle(
        version=cfg.service.model_version,
        created_at=datetime.now(timezone.utc).isoformat(),
        count_models=prod_count, proba_models=prod_proba,
        duration_model=duration_model, severity_model=sev_model,
        feature_meta=meta_reg, severity_thresholds=meta["severity"]["thresholds_minutes"],
        severity_labels=sev_labels, regions=sorted(region["region"].unique().tolist()),
        count_horizons=cfg.targets.count_horizons_hours,
        proba_windows=cfg.targets.proba_windows_hours,
        best_count_model_name=best_count, metrics=summary, latest_features=latest_features,
    )
    bundle.save(models_dir)
    return summary


def _write_training_report(reports: Path, summary: dict) -> None:
    comp = summary.get("count_comparison", {})
    lines = ["# Training Report", "", f"_Generated: {datetime.now(timezone.utc).isoformat()}_", "",
             f"**Best national count model:** `{comp.get('best_model')}` "
             f"(primary metric: {comp.get('primary_metric')})"]
    if "tuning" in summary:
        t = summary["tuning"]
        lines += ["", "## Optimization", f"- Best tuned LightGBM MAE: **{t.get('best_mae')}**",
                  f"- Naive baseline MAE: {t.get('naive_baseline_mae')}",
                  f"- Best params: `{t.get('best_params')}`"]
    if "production_count_backtest" in summary:
        lines += ["", "## Production region model (count, H=1) — backtest", "```",
                  json.dumps(summary["production_count_backtest"], indent=2), "```"]
    if "severity_metrics" in summary:
        lines += ["", "## Severity classifier", "```",
                  json.dumps(summary["severity_metrics"], indent=2), "```"]
    (reports / "training_report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Train, backtest, compare and persist models.")
    parser.add_argument("--stage", choices=["all", "national", "production"], default="all")
    parser.add_argument("--fast", action="store_true", help="Fewer folds for a quick run.")
    parser.add_argument("--no-region-proba", action="store_true",
                        help="Skip the per-region classification backtest.")
    args = parser.parse_args(argv)

    cfg = load_config()
    cfg.ensure_dirs()
    reports = Path(cfg.paths.reports_dir)
    tracker = ExperimentTracker(reports / "experiments.jsonl")
    n_folds = 2 if args.fast else cfg.backtest.n_folds
    n_folds_region = min(3, n_folds)

    summary: dict = {"created_at": datetime.now(timezone.utc).isoformat(), "stage": args.stage}
    if args.stage in ("all", "national"):
        summary.update(run_national_stage(cfg, n_folds, tracker, fast=args.fast))
    if args.stage in ("all", "production"):
        summary.update(run_production_stage(cfg, n_folds_region, args, tracker))

    with open(reports / "metrics_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    _write_training_report(reports, summary)
    log.info("Training stage '%s' complete.", args.stage)
    return summary


if __name__ == "__main__":
    main()
