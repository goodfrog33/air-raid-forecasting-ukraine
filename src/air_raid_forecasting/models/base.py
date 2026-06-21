"""Unified forecasting interface (Phase 6).

Every model — baseline, statistical, ML or deep — implements the same
:class:`Forecaster` contract so the backtester can treat them interchangeably
and compare them fairly.

Headline task
-------------
All models in the headline comparison answer the *same* question: **predict the
national alert-start count for hour ``t`` using only information available
through hour ``t-1``** (1-step-ahead, rolling origin).

* Series models (baselines, ETS, SARIMA, Prophet, LSTM) forecast the raw
  ``value_col`` series.
* Supervised ML models predict the same target from leakage-safe features
  (lags, shifted rolling stats, recency, calendar).

Because the target window is inclusive of ``t`` (see ``features.targets``), the
ML target ``alerts_started`` and the series target coincide exactly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ModelContext:
    """Everything a model might need to fit/predict, passed by the backtester."""

    value_col: str = "alerts_started"
    feature_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    seasonal_period: int = 24
    timeout_s: int = 240
    seed: int = 42
    task: str = "count"  # "count" (regression) | "proba" (classification)
    target_col: str | None = None  # explicit supervised target; defaults to value_col


class Forecaster(ABC):
    """Abstract base for all forecasters."""

    #: short identifier used in result tables
    name: str = "forecaster"
    #: one of {"baseline", "statistical", "ml", "advanced"}
    family: str = "base"
    #: "count" for regression, "proba" for binary classification
    task: str = "count"

    @abstractmethod
    def fit(self, train: pd.DataFrame, ctx: ModelContext) -> "Forecaster":
        ...

    @abstractmethod
    def predict(self, train: pd.DataFrame, test: pd.DataFrame, ctx: ModelContext) -> np.ndarray:
        """Return predictions aligned 1:1 with the rows of *test*.

        For regression: predicted counts. For classification: P(class=1).
        """

    # -- shared helpers -----------------------------------------------------
    @staticmethod
    def _target(df: pd.DataFrame, ctx: ModelContext) -> np.ndarray:
        col = ctx.target_col or ctx.value_col
        return df[col].to_numpy()

    @staticmethod
    def _full_values(train: pd.DataFrame, test: pd.DataFrame, col: str) -> np.ndarray:
        """Concatenate train+test values of *col* (contiguous in time)."""
        return np.concatenate([train[col].to_numpy(), test[col].to_numpy()])

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__} name={self.name!r} family={self.family!r}>"


def clip_nonneg(arr: np.ndarray) -> np.ndarray:
    """Counts can't be negative; round-free clip used by count models."""
    return np.clip(arr, 0.0, None)
