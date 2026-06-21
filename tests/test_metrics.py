"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np

from air_raid_forecasting.evaluation.metrics import (
    classification_metrics,
    mae,
    regression_metrics,
    rmse,
    smape,
)


def test_basic_regression_metrics():
    y = np.array([0.0, 2.0, 4.0])
    p = np.array([0.0, 2.0, 4.0])
    assert mae(y, p) == 0.0
    assert rmse(y, p) == 0.0
    assert smape(y, p) == 0.0  # perfect, incl. the 0/0 case
    m = regression_metrics(y, p)
    assert m["MAE"] == 0.0 and m["SMAPE"] == 0.0


def test_mape_skips_zeros():
    y = np.array([0.0, 10.0])
    p = np.array([5.0, 11.0])
    m = regression_metrics(y, p)
    assert 0.0 < m["MAPE_coverage"] < 1.0  # one of two actuals is zero


def test_classification_metrics_ranges():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    m = classification_metrics(y, p)
    assert 0.0 <= m["ROC_AUC"] <= 1.0
    assert m["Accuracy"] == 1.0
    assert m["ROC_AUC"] == 1.0


def test_classification_single_class_auc_nan():
    y = np.array([1, 1, 1])
    p = np.array([0.6, 0.7, 0.8])
    m = classification_metrics(y, p)
    assert np.isnan(m["ROC_AUC"])
