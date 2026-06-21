"""Tests for the FastAPI service. The /health endpoint must work even without a
trained model bundle; prediction tests run only if a bundle is present."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from air_raid_forecasting.config import load_config
from air_raid_forecasting.service.app import app, get_predictor

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ok", "degraded"}
    assert "model_version" in body


def test_root():
    r = client.get("/")
    assert r.status_code == 200


def _bundle_exists() -> bool:
    cfg = load_config()
    return (Path(cfg.paths.models_dir) / "model_bundle.joblib").exists()


def test_predict_if_model_available():
    get_predictor.cache_clear()
    predictor = get_predictor()
    if predictor is None:
        return  # not trained yet — health-only environment; skip silently
    region = predictor.regions[0]
    r = client.post("/predict", json={"region": region, "forecast_horizon_hours": 6})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["alert_probability"] <= 1.0
    assert body["predicted_alert_count"] >= 0
    assert body["severity"] in {"Low", "Medium", "High", "Critical"}

    rb = client.post("/predict_batch",
                     json={"items": [{"region": region, "forecast_horizon_hours": 1},
                                     {"region": region, "forecast_horizon_hours": 24}]})
    assert rb.status_code == 200
    assert len(rb.json()["predictions"]) == 2


def test_unknown_region_404_if_model_available():
    get_predictor.cache_clear()
    if get_predictor() is None:
        return
    r = client.post("/predict", json={"region": "Atlantis", "forecast_horizon_hours": 6})
    assert r.status_code == 404


def test_model_selection_and_news_toggle():
    get_predictor.cache_clear()
    predictor = get_predictor()
    if predictor is None:
        return
    region = predictor.regions[0]
    # Explicit model choice is honored (or gracefully falls back to an available one).
    for m in predictor.models:
        r = client.post("/predict", json={"region": region, "forecast_horizon_hours": 6, "model": m})
        assert r.status_code == 200
        assert r.json()["model"] in predictor.models
    # News toggle works when a news variant exists; never errors otherwise.
    r = client.post("/predict", json={"region": region, "forecast_horizon_hours": 6, "use_news": True})
    assert r.status_code == 200
    assert r.json()["news_factor"] == predictor.has_news()
