"""Forecasting target construction (Phase 5).

All targets look strictly into the *future* window ``[t+1, t+H]`` and are built
with vectorized cumulative sums, so they never leak the current observation.

* **Target A** — ``target_count_{H}h``: number of alerts in the next H hours.
* **Target B** — ``target_any_{H}h``: 1 if any alert occurs within the next H
  hours (binary classification).
* **Target C** — alert duration (regression on the event table).
* **Target D** — severity class, derived from the duration distribution.

Rows whose future window extends past the end of the series get ``NaN`` targets
and are dropped before training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _forward_sum(series: pd.Series, horizon: int) -> pd.Series:
    """Sum of values over the inclusive future window ``[t, t+H-1]``.

    Computed as ``C_{t+H-1} - C_{t-1}`` via cumulative sums. The window starts
    at *t* (the hour being forecast) so that H=1 reduces to the contemporaneous
    value — this lets ML targets line up exactly with the 1-step-ahead series
    models in the backtest. Real-time honesty is preserved by the *features*,
    which only use information through ``t-1``.
    """
    c = series.cumsum()
    c_prev = c.shift(1).fillna(0)        # C_{t-1}, with C_{-1} = 0
    return c.shift(-(horizon - 1)) - c_prev


def add_count_targets(
    df: pd.DataFrame,
    value_col: str,
    horizons: list[int],
    group_col: str | None = None,
    ts_col: str = "timestamp",
) -> tuple[pd.DataFrame, list[str]]:
    out = df.sort_values([group_col, ts_col] if group_col else [ts_col]).copy()
    names: list[str] = []
    for h in horizons:
        name = f"target_count_{h}h"
        if group_col:
            out[name] = out.groupby(group_col, sort=False)[value_col].transform(
                lambda s, h=h: _forward_sum(s, h)
            )
        else:
            out[name] = _forward_sum(out[value_col], h)
        names.append(name)
    return out, names


def add_proba_targets(
    df: pd.DataFrame,
    flag_col: str,
    windows: list[int],
    group_col: str | None = None,
    ts_col: str = "timestamp",
) -> tuple[pd.DataFrame, list[str]]:
    out = df.sort_values([group_col, ts_col] if group_col else [ts_col]).copy()
    names: list[str] = []
    for h in windows:
        name = f"target_any_{h}h"
        if group_col:
            cnt = out.groupby(group_col, sort=False)[flag_col].transform(
                lambda s, h=h: _forward_sum(s, h)
            )
        else:
            cnt = _forward_sum(out[flag_col], h)
        # Preserve NaN at the tail (incomplete future window).
        out[name] = np.where(cnt.isna(), np.nan, (cnt > 0).astype(float))
        names.append(name)
    return out, names


def severity_thresholds(events: pd.DataFrame, quantiles: list[float]) -> list[float]:
    """Duration cut-points (minutes) for severity classes."""
    dur = events["duration_minutes"].astype(float)
    return [float(dur.quantile(q)) for q in quantiles]


def severity_from_duration(duration_minutes: float | np.ndarray, thresholds: list[float],
                           labels: list[str]):
    """Map a duration (or array) to a severity label using *thresholds*."""
    edges = [-np.inf, *thresholds, np.inf]
    cats = pd.cut(np.atleast_1d(duration_minutes), bins=edges, labels=labels, include_lowest=True)
    result = pd.Categorical(cats, categories=labels, ordered=True)
    if np.isscalar(duration_minutes):
        return str(result[0])
    return result


def add_severity_label(events: pd.DataFrame, quantiles: list[float], labels: list[str]) -> tuple[pd.DataFrame, list[float]]:
    """Add a ``severity`` ordered-categorical column derived from duration."""
    out = events.copy()
    thr = severity_thresholds(out, quantiles)
    out["severity"] = severity_from_duration(out["duration_minutes"].to_numpy(), thr, labels)
    return out, thr
