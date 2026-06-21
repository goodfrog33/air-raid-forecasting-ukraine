"""Time-series backtesting framework (Phase 7).

Rigorous, leakage-free model evaluation via **rolling-origin** cross-validation
with either an *expanding* or *rolling* training window. Never a random split.

Expanding scheme (the default)::

    fold 1:  train [t0 .............. o1) | test [o1, o1+H)
    fold 2:  train [t0 ................... o2) | test [o2, o2+H)
    fold 3:  train [t0 ...................... o3) | test [o3, o3+H)
              (origin advances by `step`; train always starts at t0)

An optional ``gap`` between train end and test start guards against leakage
from features whose windows straddle the boundary. Each (model, fold) fit is
wrapped in a wall-clock timeout so a pathologically slow model is skipped
rather than stalling the whole comparison.
"""

from __future__ import annotations

import signal
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pandas as pd

from air_raid_forecasting.logging_utils import get_logger
from air_raid_forecasting.models.base import Forecaster, ModelContext
from air_raid_forecasting.evaluation.metrics import classification_metrics, regression_metrics

log = get_logger(__name__)


@dataclass
class TimeFold:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # exclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # exclusive


class _Timeout(Exception):
    pass


@contextmanager
def time_limit(seconds: int):
    """Raise :class:`_Timeout` if the block runs longer than *seconds* (Unix)."""
    if seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def make_folds(
    timestamps: pd.Series,
    scheme: str = "expanding",
    n_folds: int = 5,
    test_horizon: str = "30D",
    min_train: str = "270D",
    step: str = "30D",
    gap_hours: int = 0,
) -> list[TimeFold]:
    ts = pd.to_datetime(pd.Series(timestamps).sort_values().unique(), utc=True)
    t_min, t_max = ts[0], ts[-1]
    H = pd.Timedelta(test_horizon)
    MT = pd.Timedelta(min_train)
    S = pd.Timedelta(step)
    GAP = pd.Timedelta(hours=gap_hours)

    # Place the last fold's test window flush against the end of the data, then
    # walk backwards by `step` so folds cover the most recent, relevant period.
    folds: list[TimeFold] = []
    test_start = t_max - H
    for i in range(n_folds):
        ts_start = test_start - i * S
        ts_end = ts_start + H
        train_end = ts_start - GAP
        train_start = t_min if scheme == "expanding" else max(t_min, train_end - MT)
        if train_end - train_start < MT:
            break
        folds.append(TimeFold(0, train_start, train_end, ts_start, ts_end))
    folds = list(reversed(folds))
    for i, f in enumerate(folds):
        f.index = i
    if not folds:
        raise ValueError(
            "No valid folds — the series is too short for the requested "
            f"min_train={min_train}, test_horizon={test_horizon}, n_folds={n_folds}."
        )
    return folds


def _slice(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, ts_col: str) -> pd.DataFrame:
    return df[(df[ts_col] >= start) & (df[ts_col] < end)]


def backtest_model(
    model: Forecaster,
    df: pd.DataFrame,
    ctx: ModelContext,
    folds: list[TimeFold],
    ts_col: str = "timestamp",
    target_col: str | None = None,
    store_predictions: bool = False,
) -> dict:
    """Backtest a single model across all folds; return per-fold + aggregate metrics."""
    is_proba = model.task == "proba"
    rows: list[dict] = []
    preds_store: dict[int, dict] = {}
    for f in folds:
        train = _slice(df, f.train_start, f.train_end, ts_col)
        test = _slice(df, f.test_start, f.test_end, ts_col)
        if target_col is not None:
            test = test[test[target_col].notna()]
            train = train[train[target_col].notna()]
        if len(train) < 50 or len(test) == 0:
            continue
        t0 = time.time()
        try:
            with time_limit(ctx.timeout_s):
                model.fit(train, ctx)
                y_pred = model.predict(train, test, ctx)
        except _Timeout:
            log.warning("  %s timed out (>%ss) on fold %d — skipping model.",
                        model.name, ctx.timeout_s, f.index)
            return {"model": model.name, "family": model.family, "status": "timeout",
                    "per_fold": pd.DataFrame(rows), "aggregate": {}}
        except Exception as exc:  # robust: a failing model shouldn't kill the run
            log.warning("  %s failed on fold %d: %s", model.name, f.index, exc)
            continue
        elapsed = time.time() - t0

        y_true = test[target_col].to_numpy() if target_col else test[ctx.value_col].to_numpy()
        if is_proba:
            metrics = classification_metrics(y_true.astype(int), y_pred)
        else:
            metrics = regression_metrics(y_true.astype(float), y_pred)
        metrics.update({"model": model.name, "family": model.family,
                        "fold": f.index, "n_test": len(test), "fit_seconds": round(elapsed, 2)})
        rows.append(metrics)
        if store_predictions:
            preds_store[f.index] = {
                "timestamp": test[ts_col].to_numpy(),
                "y_true": y_true, "y_pred": y_pred,
            }

    per_fold = pd.DataFrame(rows)
    metric_cols = [c for c in per_fold.columns
                   if c not in {"model", "family", "fold", "n_test"}]
    aggregate = {c: float(per_fold[c].mean()) for c in metric_cols} if not per_fold.empty else {}
    result = {"model": model.name, "family": model.family,
              "status": "ok" if not per_fold.empty else "no_folds",
              "per_fold": per_fold, "aggregate": aggregate}
    if store_predictions:
        result["predictions"] = preds_store
    return result


def backtest_models(
    models: list[Forecaster],
    df: pd.DataFrame,
    ctx: ModelContext,
    folds: list[TimeFold],
    ts_col: str = "timestamp",
    target_col: str | None = None,
    store_predictions: bool = False,
) -> dict:
    """Backtest several models; return tidy per-fold and aggregated frames."""
    per_fold_frames = []
    aggregates = []
    predictions: dict[str, dict] = {}
    for model in models:
        log.info("Backtesting %-16s (%s) ...", model.name, model.family)
        res = backtest_model(model, df, ctx, folds, ts_col, target_col, store_predictions)
        if not res["per_fold"].empty:
            per_fold_frames.append(res["per_fold"])
            agg = {"model": model.name, "family": model.family, "status": res["status"]}
            agg.update(res["aggregate"])
            aggregates.append(agg)
        if store_predictions and "predictions" in res:
            predictions[model.name] = res["predictions"]

    per_fold = pd.concat(per_fold_frames, ignore_index=True) if per_fold_frames else pd.DataFrame()
    aggregate = pd.DataFrame(aggregates)
    return {"per_fold": per_fold, "aggregate": aggregate,
            "folds": folds, "predictions": predictions}
