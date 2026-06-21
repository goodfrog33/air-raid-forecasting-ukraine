"""Trained-model persistence (Phase 6/11).

A :class:`ModelBundle` packages everything the prediction service needs:
the trained per-horizon count regressors and probability classifiers, the
duration & severity models, feature metadata, severity thresholds, backtest
metrics and a model version. Saved/loaded with joblib as a single ``.joblib``
artifact under ``models/``.
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
    # Production estimators (global, region-aware), keyed by horizon hours.
    count_models: dict[int, Any] = field(default_factory=dict)
    proba_models: dict[int, Any] = field(default_factory=dict)
    duration_model: Any = None
    severity_model: Any = None
    # Metadata / context for serving.
    feature_meta: dict = field(default_factory=dict)
    severity_thresholds: list[float] = field(default_factory=list)
    severity_labels: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=list)
    count_horizons: list[int] = field(default_factory=list)
    proba_windows: list[int] = field(default_factory=list)
    best_count_model_name: str = ""
    metrics: dict = field(default_factory=dict)
    # The most recent feature row per region, so the service can predict the
    # "next window" without recomputing the whole feature pipeline at request time.
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
