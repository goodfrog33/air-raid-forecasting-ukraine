"""Trained-model persistence (Phase 6/11).

A :class:`ModelBundle` packages everything the prediction service needs. It
supports a nested, interactive structure:

    variants["base"|"news"] = {
        "count": {horizon_hours: {model_name: Forecaster}},
        "proba": {window_hours: {model_name: Forecaster}},
        "latest_features": DataFrame,        # one row per region (serving state)
        "feature_cols": [...], "categorical_cols": [...],
    }

so the dashboard/API can pick the model (best or manual) and toggle the news
factor on/off. Older single-model bundles (flat ``count_models``/``proba_models``)
still load — the predictor detects which layout it has.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from air_raid_forecasting.logging_utils import get_logger

log = get_logger(__name__)

BUNDLE_NAME = "model_bundle.joblib"


@dataclass
class ModelBundle:
    version: str = "1.0.0"
    created_at: str = ""

    # --- interactive (multi-model, multi-variant) layout ---
    variants: dict = field(default_factory=dict)
    available_models: list[str] = field(default_factory=list)
    available_variants: list[str] = field(default_factory=list)
    # {variant: {"count_1h": {model: MAE}, "proba_6h": {model: ROC_AUC}}}
    per_model_metrics: dict = field(default_factory=dict)
    news_lift: dict = field(default_factory=dict)

    # --- shared auxiliary models / metadata ---
    duration_model: Any = None
    severity_model: Any = None
    severity_thresholds: list[float] = field(default_factory=list)
    severity_labels: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    count_horizons: list[int] = field(default_factory=list)
    proba_windows: list[int] = field(default_factory=list)
    best_count_model_name: str = ""   # headline national comparison winner
    metrics: dict = field(default_factory=dict)

    # --- legacy flat layout (old bundles) ---
    count_models: dict = field(default_factory=dict)
    proba_models: dict = field(default_factory=dict)
    feature_meta: dict = field(default_factory=dict)
    latest_features: Any = None

    def save(self, models_dir: str | Path, name: str = BUNDLE_NAME) -> Path:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / name
        joblib.dump(self, path)
        log.info("Saved model bundle -> %s (%.1f MB)", path, path.stat().st_size / 1e6)
        return path

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        path = Path(path)
        if path.is_dir():
            path = path / BUNDLE_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {path}. Train first: "
                f"python -m air_raid_forecasting.pipeline.run_train"
            )
        bundle = joblib.load(path)
        log.info("Loaded model bundle v%s (%s)", bundle.version, path)
        return bundle
