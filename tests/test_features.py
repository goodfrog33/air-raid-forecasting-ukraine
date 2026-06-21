"""Tests for feature engineering — especially leakage safety of targets/lags."""

from __future__ import annotations

import numpy as np
import pandas as pd

from air_raid_forecasting.features.targets import (
    _forward_sum,
    add_count_targets,
    add_proba_targets,
    severity_from_duration,
)
from air_raid_forecasting.features.timeseries import add_lags, add_rolling, add_time_since_last


def _series_df(n=50):
    ts = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "y": np.arange(n, dtype=float)})


def test_forward_sum_inclusive_window():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    # H=1 -> the value itself
    assert list(_forward_sum(s, 1).dropna()) == [1, 2, 3, 4, 5]
    # H=2 -> y_t + y_{t+1}; last is NaN (incomplete window)
    out = _forward_sum(s, 2)
    assert out.iloc[0] == 3 and out.iloc[3] == 9
    assert np.isnan(out.iloc[-1])


def test_count_target_alignment():
    df = _series_df()
    out, names = add_count_targets(df, "y", [1, 3])
    assert names == ["target_count_1h", "target_count_3h"]
    # target_count_1h == y exactly
    assert np.allclose(out["target_count_1h"].dropna(), df["y"][: out["target_count_1h"].notna().sum()])


def test_lags_are_past_only():
    df = _series_df()
    out = add_lags(df, ["y"], [1, 2])
    # y_lag_1 at row t equals y at t-1
    assert out["y_lag_1"].iloc[5] == df["y"].iloc[4]
    assert out["y_lag_2"].iloc[5] == df["y"].iloc[3]
    assert np.isnan(out["y_lag_1"].iloc[0])


def test_rolling_excludes_current():
    df = _series_df()
    out = add_rolling(df, ["y"], [3], ["mean"])
    # rolling mean at t uses [t-3, t-1] -> for t=5 that's mean(2,3,4)=3
    assert out["y_roll_mean_3"].iloc[5] == 3.0


def test_proba_target_binary():
    df = _series_df()
    df["flag"] = (df["y"] % 2 == 0).astype(int)
    out, names = add_proba_targets(df, "flag", [3])
    vals = out["target_any_3h"].dropna().unique()
    assert set(vals) <= {0.0, 1.0}


def test_time_since_last():
    ts = pd.date_range("2023-01-01", periods=6, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "flag": [1, 0, 0, 1, 0, 0]})
    out = add_time_since_last(df, "flag")
    # Uses previous flag; counter resets the hour AFTER a flagged hour.
    assert list(out["time_since_last_alert"]) == [1000, 0, 1, 2, 0, 1]


def test_severity_mapping():
    thr = [30.0, 120.0, 360.0]
    labels = ["Low", "Medium", "High", "Critical"]
    assert severity_from_duration(10, thr, labels) == "Low"
    assert severity_from_duration(60, thr, labels) == "Medium"
    assert severity_from_duration(200, thr, labels) == "High"
    assert severity_from_duration(500, thr, labels) == "Critical"
