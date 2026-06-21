"""Classical statistical forecasters: ETS and SARIMA (Phase 6).

Both are estimated once on the training window; we then obtain true
**1-step-ahead rolling** forecasts on the test span by rebuilding the model on
the concatenated ``[train_tail, test]`` series and *re-applying the fitted
parameters* without re-estimating:

* SARIMAX -> ``model.filter(params)`` (Kalman one-step-ahead ``fittedvalues``),
* ETS     -> ``model.smooth(params)`` (exp-smoothing one-step-ahead recursion).

For tractability on multi-year hourly data, the models fit on a trailing window
(default 180 days) rather than the whole history — ARIMA/ETS rarely benefit
from years of stale dynamics, and this keeps the seasonal (s=24) estimation
fast. A deliberate, documented modeling choice.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from air_raid_forecasting.logging_utils import get_logger
from air_raid_forecasting.models.base import Forecaster, ModelContext, clip_nonneg

log = get_logger(__name__)


def _tail(y: np.ndarray, max_obs: int) -> np.ndarray:
    return y[-max_obs:] if (max_obs and len(y) > max_obs) else y


class ETSForecaster(Forecaster):
    name = "ets"
    family = "statistical"

    def __init__(self, trend: str | None = "add", damped: bool = True,
                 seasonal: str | None = None, max_train_obs: int = 4320) -> None:
        self.trend = trend
        self.damped = damped
        self.seasonal = seasonal
        self.max_train_obs = max_train_obs
        self._params = None
        self._train_y: np.ndarray | None = None

    def _spec(self, y: np.ndarray, ctx: ModelContext):
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        return ETSModel(
            y, error="add", trend=self.trend, damped_trend=self.damped,
            seasonal=self.seasonal,
            seasonal_periods=ctx.seasonal_period if self.seasonal else None,
        )

    def fit(self, train: pd.DataFrame, ctx: ModelContext) -> "ETSForecaster":
        y = _tail(train[ctx.value_col].astype(float).to_numpy(), self.max_train_obs)
        self._train_y = y
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._params = self._spec(y, ctx).fit(disp=False).params
        return self

    def predict(self, train: pd.DataFrame, test: pd.DataFrame, ctx: ModelContext) -> np.ndarray:
        n = len(test)
        full = np.concatenate([self._train_y, test[ctx.value_col].astype(float).to_numpy()])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._spec(full, ctx).smooth(self._params)
        return clip_nonneg(np.asarray(res.fittedvalues)[-n:])


class SARIMAForecaster(Forecaster):
    name = "sarima"
    family = "statistical"

    def __init__(self, order=(2, 1, 1), seasonal_order=(1, 0, 1, 24),
                 maxiter: int = 50, max_train_obs: int = 4320) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.maxiter = maxiter
        self.max_train_obs = max_train_obs
        self._params = None
        self._train_y: np.ndarray | None = None
        self._seasonal_used = seasonal_order

    def _spec(self, y: np.ndarray, seasonal_order):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        return SARIMAX(
            y, order=self.order, seasonal_order=seasonal_order,
            enforce_stationarity=False, enforce_invertibility=False,
        )

    def fit(self, train: pd.DataFrame, ctx: ModelContext) -> "SARIMAForecaster":
        y = _tail(train[ctx.value_col].astype(float).to_numpy(), self.max_train_obs)
        self._train_y = y
        seasonal_order = self.seasonal_order
        if seasonal_order and seasonal_order[-1] != ctx.seasonal_period:
            seasonal_order = (*seasonal_order[:3], ctx.seasonal_period)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = self._spec(y, seasonal_order).fit(
                    disp=False, maxiter=self.maxiter, method="lbfgs"
                )
                self._seasonal_used = seasonal_order
            except Exception as exc:
                log.warning("SARIMA seasonal fit failed (%s); falling back to ARIMA%s.",
                            exc, self.order)
                self._seasonal_used = (0, 0, 0, 0)
                res = self._spec(y, self._seasonal_used).fit(
                    disp=False, maxiter=self.maxiter, method="lbfgs"
                )
        self._params = res.params
        return self

    def predict(self, train: pd.DataFrame, test: pd.DataFrame, ctx: ModelContext) -> np.ndarray:
        n = len(test)
        full = np.concatenate([self._train_y, test[ctx.value_col].astype(float).to_numpy()])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._spec(full, self._seasonal_used).filter(self._params)
        return clip_nonneg(np.asarray(res.fittedvalues)[-n:])
