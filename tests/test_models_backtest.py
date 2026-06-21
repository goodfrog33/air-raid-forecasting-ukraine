"""Tests for forecasters, the registry and the backtesting framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from air_raid_forecasting.evaluation.backtest import backtest_model, make_folds
from air_raid_forecasting.models.base import ModelContext
from air_raid_forecasting.models.baselines import (
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)
from air_raid_forecasting.models.ml import LightGBMForecaster


@pytest.fixture
def national_like():
    n = 24 * 120  # 120 days hourly
    ts = pd.date_range("2023-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    # Signal with daily seasonality + noise, non-negative counts.
    hours = ts.hour.to_numpy()
    y = np.clip(3 + 2 * np.sin(2 * np.pi * hours / 24) + rng.normal(0, 1, n), 0, None)
    df = pd.DataFrame({"timestamp": ts, "alerts_started": y.round()})
    df["y_lag_1"] = df["alerts_started"].shift(1).fillna(0)
    df["y_lag_24"] = df["alerts_started"].shift(24).fillna(0)
    return df


def _ctx(features):
    return ModelContext(value_col="alerts_started", feature_cols=features,
                        seasonal_period=24, timeout_s=60, task="count")


def test_naive_predicts_previous(national_like):
    ctx = _ctx(["y_lag_1"])
    train, test = national_like.iloc[:2000], national_like.iloc[2000:2050]
    m = NaiveForecaster().fit(train, ctx)
    pred = m.predict(train, test, ctx)
    expected = national_like["alerts_started"].iloc[1999:2049].to_numpy()
    assert np.allclose(pred, expected)


def test_seasonal_and_ma_shapes(national_like):
    ctx = _ctx(["y_lag_1"])
    train, test = national_like.iloc[:2000], national_like.iloc[2000:2050]
    for M in (SeasonalNaiveForecaster(24), MovingAverageForecaster(24)):
        pred = M.fit(train, ctx).predict(train, test, ctx)
        assert pred.shape == (50,)
        assert (pred >= 0).all()


def test_lightgbm_beats_nothing(national_like):
    ctx = _ctx(["y_lag_1", "y_lag_24"])
    train, test = national_like.iloc[:2000], national_like.iloc[2000:2200]
    m = LightGBMForecaster(params={"n_estimators": 50}).fit(train, ctx)
    pred = m.predict(train, test, ctx)
    assert pred.shape == (200,)
    assert np.isfinite(pred).all()


def test_make_folds_expanding_no_overlap(national_like):
    folds = make_folds(national_like["timestamp"], scheme="expanding", n_folds=3,
                       test_horizon="10D", min_train="30D", step="10D")
    assert len(folds) >= 1
    for f in folds:
        assert f.train_end <= f.test_start  # no leakage
        assert f.test_start < f.test_end


def test_backtest_model_runs(national_like):
    ctx = _ctx(["y_lag_1"])
    folds = make_folds(national_like["timestamp"], scheme="expanding", n_folds=2,
                       test_horizon="10D", min_train="40D", step="10D")
    res = backtest_model(NaiveForecaster(), national_like, ctx, folds)
    assert not res["per_fold"].empty
    assert "MAE" in res["aggregate"]
