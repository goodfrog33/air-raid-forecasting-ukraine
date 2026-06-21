"""Forecasting & classification metrics (Phase 8).

Count forecasts contain many true zeros (most region-hours are quiet), which
makes vanilla MAPE undefined. We therefore report:

* **MAE, RMSE** — primary, always well-defined,
* **MAPE** — computed only over non-zero actuals, with the coverage fraction
  reported alongside so it is never read in isolation,
* **SMAPE** — symmetric, defined as 0 where both actual and prediction are 0.

Classification metrics cover the full brief (accuracy/precision/recall/F1/
ROC-AUC) plus log-loss and Brier score for probabilistic calibration.
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Return (MAPE %, coverage) computed over non-zero actuals only."""
    mask = np.abs(y_true) > 1e-8
    coverage = float(mask.mean())
    if not mask.any():
        return float("nan"), 0.0
    val = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    return float(val), coverage


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    num = 2.0 * np.abs(y_true - y_pred)
    out = np.divide(num, denom, out=np.zeros_like(denom, dtype=float), where=denom != 0)
    return float(np.mean(out) * 100.0)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m, cov = mape(y_true, y_pred)
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": m,
        "MAPE_coverage": cov,
        "SMAPE": smape(y_true, y_pred),
    }


def classification_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        brier_score_loss,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.clip(np.asarray(y_proba, dtype=float), 1e-7, 1 - 1e-7)
    y_hat = (y_proba >= threshold).astype(int)
    single_class = len(np.unique(y_true)) < 2
    return {
        "Accuracy": float(accuracy_score(y_true, y_hat)),
        "Precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "Recall": float(recall_score(y_true, y_hat, zero_division=0)),
        "F1": float(f1_score(y_true, y_hat, zero_division=0)),
        "ROC_AUC": float("nan") if single_class else float(roc_auc_score(y_true, y_proba)),
        "LogLoss": float(log_loss(y_true, y_proba, labels=[0, 1])),
        "Brier": float(brier_score_loss(y_true, y_proba)),
        "PositiveRate": float(np.mean(y_true)),
    }
