"""Lag, rolling and recency features (Phase 4).

**Leakage safety** is the central concern. Every feature at time *t* uses only
information available strictly *before* the target window:

* lag features use ``shift(k)`` for k >= 1,
* rolling features are computed on the already-shifted series, so the window
  ending at *t* covers ``[t-w, t-1]`` — never the current observation,
* ``time_since_last_alert`` counts hours since the previous flagged hour.

All operations are group-aware (per region) so windows never bleed across
region boundaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_lags(
    df: pd.DataFrame,
    value_cols: list[str],
    lags: list[int],
    group_col: str | None = None,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """Add ``{col}_lag_{k}`` columns. Assumes one row per time step."""
    out = df.sort_values([group_col, ts_col] if group_col else [ts_col]).copy()
    grp = out.groupby(group_col, sort=False) if group_col else None
    for col in value_cols:
        series = grp[col] if group_col else out[col]
        for k in lags:
            out[f"{col}_lag_{k}"] = series.shift(k)
    return out


def add_rolling(
    df: pd.DataFrame,
    value_cols: list[str],
    windows: list[int],
    stats: list[str],
    group_col: str | None = None,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """Add leakage-safe rolling ``{col}_roll_{stat}_{w}`` features."""
    out = df.sort_values([group_col, ts_col] if group_col else [ts_col]).copy()
    for col in value_cols:
        # Shift first so the rolling window ending at t covers [t-w, t-1].
        if group_col:
            shifted = out.groupby(group_col, sort=False)[col].shift(1)
        else:
            shifted = out[col].shift(1)
        for w in windows:
            for stat in stats:
                name = f"{col}_roll_{stat}_{w}"
                if group_col:
                    r = shifted.groupby(out[group_col], sort=False).rolling(w, min_periods=1)
                    vals = getattr(r, stat)().reset_index(level=0, drop=True)
                else:
                    vals = getattr(shifted.rolling(w, min_periods=1), stat)()
                out[name] = vals
    return out


def add_time_since_last(
    df: pd.DataFrame,
    flag_col: str,
    group_col: str | None = None,
    ts_col: str = "timestamp",
    cap: int = 1000,
) -> pd.DataFrame:
    """Hours since the previous hour with ``flag_col == 1`` (per group).

    Uses the *previous* flag (shifted) to stay leakage-safe. Capped at *cap*.
    """
    out = df.sort_values([group_col, ts_col] if group_col else [ts_col]).copy()

    def _per_group(flags: pd.Series) -> pd.Series:
        prev = flags.shift(1).fillna(0).to_numpy()
        since = np.empty(len(prev), dtype=float)
        counter = cap
        for i, was_alert in enumerate(prev):
            counter = 0 if was_alert == 1 else min(counter + 1, cap)
            since[i] = counter
        return pd.Series(since, index=flags.index)

    if group_col:
        out["time_since_last_alert"] = (
            out.groupby(group_col, sort=False)[flag_col].transform(_per_group)
        )
    else:
        out["time_since_last_alert"] = _per_group(out[flag_col])
    return out
