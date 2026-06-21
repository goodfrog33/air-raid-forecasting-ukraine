"""Pydantic request/response schemas for the prediction service (Phase 11)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    region: str = Field(..., examples=["Kyiv"], description="Region name (oblast or short name).")
    forecast_horizon_hours: int = Field(6, ge=1, le=72, examples=[6])


class PredictResponse(BaseModel):
    region: str
    forecast_horizon_hours: int
    alert_probability: float = Field(..., ge=0, le=1)
    predicted_alert_count: int
    predicted_duration_minutes: float
    severity: str
    confidence: float = Field(..., ge=0, le=1)
    model_version: str
    # Helpful extras (not in the minimal spec, but useful for clients).
    matched_horizon_hours: int | None = None
    as_of: str | None = None


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest]


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
    n_regions: int
    best_count_model: str | None = None


class MetricsResponse(BaseModel):
    model_version: str
    best_count_model: str | None = None
    count_horizons: list[int]
    proba_windows: list[int]
    severity_metrics: dict | None = None
    production_count_backtest: dict | None = None
    count_comparison_top: list[dict] | None = None
