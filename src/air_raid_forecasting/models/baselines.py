"""Baseline forecasters (Phase 6).

Cheap, assumption-light references that any serious model must beat. All three
produce genuine 1-step-ahead predictions using only past actuals, so they are
leakage-free by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from air_raid_forecasting.models.base import Forecaster, ModelContext, clip_nonneg


class NaiveForecaster(Forecaster):
    """ŷ_t = y_{t-1} (last observed value)."""

    name = "naive"
    family = "baseline"

    def fit(self, train: pd.DataFrame, ctx: ModelContext) -> "NaiveForecaster":
        return self

    def predict(self, train: pd.DataFrame, test: pd.DataFrame, ctx: ModelContext) -> np.ndarray:
        full = self._full_values(train, test, ctx.value_col)
        n_test = len(test)
        preds = full[len(full) - n_test - 1 : len(full) - 1]
        return clip_nonneg(preds.astype(float))


class SeasonalNaiveForecaster(Forecaster):
    """ŷ_t = y_{t-m} (value from one seasonal period ago; m defaults to 24h)."""

    name = "seasonal_naive"
    family = "baseline"

    def __init__(self, period: int | None = None) -> None:
        self.period = period

    def fit(self, train: pd.DataFrame, ctx: ModelContext) -> "SeasonalNaiveForecaster":
        self.period = self.period or ctx.seasonal_period
        return self

    def predict(self, train: pd.DataFrame, test: pd.DataFrame, ctx: ModelContext) -> np.ndarray:
        m = self.period or ctx.seasonal_period
        full = pd.Series(self._full_values(train, test, ctx.value_col))
        shifted = full.shift(m)
        preds = shifted.to_numpy()[-len(test):]
        # Fill any leading NaN (when m exceeds history) with the train mean.
        fill = float(np.nanmean(train[ctx.value_col].to_numpy())) if len(train) else 0.0
        preds = np.where(np.isnan(preds), fill, preds)
        return clip_nonneg(preds.astype(float))


class MovingAverageForecaster(Forecaster):
    """ŷ_t = mean(y_{t-w} .. y_{t-1}) — trailing moving average (w defaults to 24h)."""

    name = "moving_average"
    family = "baseline"

    def __init__(self, window: int = 24) -> None:
        self.window = window

    def fit(self, train: pd.DataFrame, ctx: ModelContext) -> "MovingAverageForecaster":
        return self

    def predict(self, train: pd.DataFrame, test: pd.DataFrame, ctx: ModelContext) -> np.ndarray:
        full = pd.Series(self._full_values(train, test, ctx.value_col))
        ma = full.shift(1).rolling(self.window, min_periods=1).mean()
        preds = ma.to_numpy()[-len(test):]
        return clip_nonneg(preds.astype(float))


class PersistenceProbaForecaster(Forecaster):
    """Classification baseline: P(alert) = previous-hour alert flag (persistence)."""

    name = "persistence"
    family = "baseline"
    task = "proba"

    def fit(self, train: pd.DataFrame, ctx: ModelContext) -> "PersistenceProbaForecaster":
        # Slight smoothing toward the base rate so logloss stays finite.
        self.base_rate = float(np.clip(train[ctx.value_col].mean(), 1e-3, 1 - 1e-3))
        return self

    def predict(self, train: pd.DataFrame, test: pd.DataFrame, ctx: ModelContext) -> np.ndarray:
        lag_col = f"{ctx.value_col}_lag_1"
        if lag_col in test.columns:
            # Per-region, leakage-safe previous-hour flag (correct for the long panel).
            prev = test[lag_col].to_numpy().astype(float)
        else:
            full = self._full_values(train, test, ctx.value_col)
            prev = full[len(full) - len(test) - 1 : len(full) - 1].astype(float)
        # Map {0,1} previous flag to a probability shaded toward the base rate.
        return np.where(prev > 0, 0.9, max(self.base_rate, 0.05))


class PriorRateForecaster(Forecaster):
    """Classification baseline: constant P(alert) = training positive rate."""

    name = "prior_rate"
    family = "baseline"
    task = "proba"

    def fit(self, train: pd.DataFrame, ctx: ModelContext) -> "PriorRateForecaster":
        col = ctx.target_col or ctx.value_col
        self.rate = float(np.clip(train[col].mean(), 1e-4, 1 - 1e-4))
        return self

    def predict(self, train: pd.DataFrame, test: pd.DataFrame, ctx: ModelContext) -> np.ndarray:
        return np.full(len(test), self.rate, dtype=float)
