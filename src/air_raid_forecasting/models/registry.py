"""Model registry / factory (Phase 6).

Maps configuration names to configured :class:`Forecaster` instances and builds
the set of models enabled for a given task. New models are added by registering
one builder here — nothing else in the pipeline needs to change.
"""

from __future__ import annotations

from collections.abc import Callable

from air_raid_forecasting.config import Config
from air_raid_forecasting.logging_utils import get_logger
from air_raid_forecasting.models.advanced import LSTMForecaster, ProphetForecaster, TFTForecaster
from air_raid_forecasting.models.base import Forecaster
from air_raid_forecasting.models.baselines import (
    MovingAverageForecaster,
    NaiveForecaster,
    PersistenceProbaForecaster,
    PriorRateForecaster,
    SeasonalNaiveForecaster,
)
from air_raid_forecasting.models.ml import (
    CatBoostForecaster,
    LightGBMForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)
from air_raid_forecasting.models.statistical import ETSForecaster, SARIMAForecaster

log = get_logger(__name__)

# Builders keyed by model name. Each takes the Config and returns a Forecaster.
BUILDERS: dict[str, Callable[[Config], Forecaster]] = {
    "naive": lambda cfg: NaiveForecaster(),
    "seasonal_naive": lambda cfg: SeasonalNaiveForecaster(cfg.modeling.seasonal_period_hours),
    "moving_average": lambda cfg: MovingAverageForecaster(window=cfg.modeling.seasonal_period_hours),
    "ets": lambda cfg: ETSForecaster(),
    "sarima": lambda cfg: SARIMAForecaster(
        seasonal_order=(1, 0, 1, cfg.modeling.seasonal_period_hours)
    ),
    "random_forest": lambda cfg: RandomForestForecaster(),
    "xgboost": lambda cfg: XGBoostForecaster(),
    "lightgbm": lambda cfg: LightGBMForecaster(),
    "catboost": lambda cfg: CatBoostForecaster(),
    "prophet": lambda cfg: ProphetForecaster(),
    "lstm": lambda cfg: LSTMForecaster(),
    "tft": lambda cfg: TFTForecaster,  # class, instantiated lazily (may raise)
    # Classification baselines.
    "persistence": lambda cfg: PersistenceProbaForecaster(),
    "prior_rate": lambda cfg: PriorRateForecaster(),
}

# Which model names can do classification (probability) tasks.
PROBA_CAPABLE = {
    "persistence", "prior_rate",
    "random_forest", "xgboost", "lightgbm", "catboost",
}

ML_MODELS = {"random_forest", "xgboost", "lightgbm", "catboost"}


def build_model(name: str, cfg: Config) -> Forecaster:
    if name not in BUILDERS:
        raise KeyError(f"Unknown model {name!r}. Known: {sorted(BUILDERS)}")
    built = BUILDERS[name](cfg)
    return built() if isinstance(built, type) else built


def build_count_models(cfg: Config) -> list[Forecaster]:
    """Instantiate the enabled count (regression) models, skipping unavailable ones."""
    names = list(cfg.modeling.enabled_models)
    if cfg.modeling.enable_lstm and "lstm" not in names:
        names.append("lstm")
    if cfg.modeling.enable_tft and "tft" not in names:
        names.append("tft")

    models: list[Forecaster] = []
    for name in names:
        try:
            models.append(build_model(name, cfg))
        except (ImportError, NotImplementedError) as exc:
            log.warning("Skipping model %r (unavailable): %s", name, exc)
    return models


def build_proba_models(cfg: Config) -> list[Forecaster]:
    """Classification models: persistence + prior-rate baselines plus enabled ML."""
    names = ["persistence", "prior_rate"]
    names += [m for m in cfg.modeling.enabled_models if m in ML_MODELS]
    models: list[Forecaster] = []
    for name in names:
        try:
            m = build_model(name, cfg)
            m.task = "proba"
            models.append(m)
        except (ImportError, NotImplementedError) as exc:
            log.warning("Skipping proba model %r: %s", name, exc)
    return models
