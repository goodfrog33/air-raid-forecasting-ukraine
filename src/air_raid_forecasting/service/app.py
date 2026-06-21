"""FastAPI prediction service (Phase 11).

Endpoints (per the brief):
    GET  /health          liveness + model status
    GET  /metrics         backtest metrics + best model
    POST /predict         single (region, horizon) forecast
    POST /predict_batch   many forecasts in one call

The best trained model bundle is loaded automatically from ``ARF_MODEL_DIR``
(default ``models/``). Run with::

    uvicorn air_raid_forecasting.service.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException

from air_raid_forecasting.config import load_config
from air_raid_forecasting.logging_utils import get_logger
from air_raid_forecasting.service.predictor import Predictor
from air_raid_forecasting.service.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)

log = get_logger(__name__)

app = FastAPI(
    title="Ukraine Air Raid Alert Forecasting API",
    description="Forecasts alert probability, count, duration and severity by region.",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_predictor() -> Predictor | None:
    cfg = load_config()
    models_dir = os.environ.get("ARF_MODEL_DIR", str(cfg.paths.models_dir))
    try:
        return Predictor.from_dir(models_dir, tz=cfg.project.timezone_local)
    except FileNotFoundError as exc:
        log.warning("Model bundle not loaded: %s", exc)
        return None


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"service": "air-raid-forecasting", "docs": "/docs", "health": "/health"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    cfg = load_config()
    predictor = get_predictor()
    return HealthResponse(
        status="ok" if predictor else "degraded",
        model_version=cfg.service.model_version,
        model_loaded=predictor is not None,
        n_regions=len(predictor.regions) if predictor else 0,
        best_count_model=predictor.b.best_count_model_name if predictor else None,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model bundle not available. Train first.")
    m = predictor.b.metrics or {}
    comp = (m.get("count_comparison") or {}).get("ranking") or []
    return MetricsResponse(
        model_version=predictor.b.version,
        best_count_model=predictor.b.best_count_model_name,
        count_horizons=predictor.b.count_horizons,
        proba_windows=predictor.b.proba_windows,
        severity_metrics=m.get("severity_metrics"),
        production_count_backtest=m.get("production_count_backtest"),
        count_comparison_top=comp[:5],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model bundle not available. Train first.")
    try:
        result = predictor.predict_one(req.region, req.forecast_horizon_hours)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PredictResponse(**result)


@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    predictor = get_predictor()
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model bundle not available. Train first.")
    try:
        items = [(it.region, it.forecast_horizon_hours) for it in req.items]
        results = predictor.predict_batch(items)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return BatchPredictResponse(predictions=[PredictResponse(**r) for r in results])
