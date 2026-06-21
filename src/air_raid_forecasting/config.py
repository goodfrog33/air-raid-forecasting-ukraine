"""Typed configuration loader for the air-raid forecasting project.

The whole pipeline reads a single YAML file (``configs/config.yaml`` by default,
overridable via the ``ARF_CONFIG`` environment variable). The YAML is parsed
into nested :class:`pydantic.BaseModel` objects so downstream code gets
auto-completion and validation instead of raw dict lookups.

Usage
-----
>>> from air_raid_forecasting.config import load_config
>>> cfg = load_config()
>>> cfg.panel.freq
'h'
>>> cfg.paths.processed_dir   # resolved to an absolute Path
PosixPath('.../air_raid_forecasting/data/processed')
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# Project root = three levels up from this file (src/air_raid_forecasting/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class ProjectCfg(BaseModel):
    name: str = "air-raid-forecasting"
    version: str = "1.0.0"
    random_seed: int = 42
    timezone_local: str = "Europe/Kyiv"


class PathsCfg(BaseModel):
    raw_dir: Path = Path("data/raw")
    external_dir: Path = Path("data/external")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    figures_dir: Path = Path("reports/figures")

    def resolve(self, root: Path) -> "PathsCfg":
        """Return a copy with every path made absolute against *root*."""
        return PathsCfg(
            **{
                name: (root / value) if not Path(value).is_absolute() else Path(value)
                for name, value in self.model_dump().items()
            }
        )


class SourceCfg(BaseModel):
    path: str
    filename: str


class DataCfg(BaseModel):
    primary_source: str = "official"
    base_url: str
    sources: dict[str, SourceCfg]
    download_timeout_seconds: int = 120
    exclude_permanent_sirens: list[str] = Field(default_factory=list)


class PreprocessCfg(BaseModel):
    level: str = "oblast"
    min_duration_seconds: int = 30
    max_duration_hours: int = 72
    naive_default_minutes: int = 30
    drop_open_ended: bool = False
    anomaly_zscore_threshold: float = 6.0


class PanelCfg(BaseModel):
    freq: str = "h"
    regions: Any = "auto"
    start: str | None = None
    end: str | None = None


class WeatherCfg(BaseModel):
    enabled: bool = False


class NewsCfg(BaseModel):
    enabled: bool = False
    query: str = 'ukraine (missile OR drone OR "air raid" OR shelling OR airstrike OR rocket)'
    lags_days: list[int] = Field(default_factory=lambda: [1, 2, 7])


class TelegramCfg(BaseModel):
    enabled: bool = False
    channels: list[str] = Field(default_factory=lambda: ["war_monitor", "radar_raketaa"])
    max_pages: int = 45


class FeaturesCfg(BaseModel):
    use_calendar: bool = True
    use_holidays: bool = True
    holiday_country: str = "UA"
    lags_hours: list[int] = Field(default_factory=lambda: [1, 3, 6, 24, 168])
    rolling_windows_hours: list[int] = Field(default_factory=lambda: [3, 6, 24, 168])
    rolling_stats: list[str] = Field(default_factory=lambda: ["mean", "std", "max", "min"])
    add_time_since_last_alert: bool = True
    weather: WeatherCfg = Field(default_factory=WeatherCfg)
    news: NewsCfg = Field(default_factory=NewsCfg)
    telegram: TelegramCfg = Field(default_factory=TelegramCfg)


class DurationTargetCfg(BaseModel):
    enabled: bool = True


class SeverityTargetCfg(BaseModel):
    enabled: bool = True
    quantiles: list[float] = Field(default_factory=lambda: [0.5, 0.8, 0.95])
    labels: list[str] = Field(default_factory=lambda: ["Low", "Medium", "High", "Critical"])


class TargetsCfg(BaseModel):
    count_horizons_hours: list[int] = Field(default_factory=lambda: [1, 6, 24])
    proba_windows_hours: list[int] = Field(default_factory=lambda: [1, 6, 24])
    duration: DurationTargetCfg = Field(default_factory=DurationTargetCfg)
    severity: SeverityTargetCfg = Field(default_factory=SeverityTargetCfg)


class ModelingCfg(BaseModel):
    primary_target: str = "count"
    primary_horizon_hours: int = 1
    enabled_models: list[str] = Field(default_factory=list)
    enable_lstm: bool = True
    enable_tft: bool = False
    per_model_timeout_seconds: int = 240
    seasonal_period_hours: int = 24


class BacktestCfg(BaseModel):
    scheme: str = "expanding"
    n_folds: int = 5
    test_horizon: str = "30D"
    min_train: str = "270D"
    step: str = "30D"
    gap_hours: int = 0


class ProductionCfg(BaseModel):
    model: str = "lightgbm"
    models: list[str] = Field(default_factory=lambda: ["lightgbm", "xgboost", "catboost"])
    train_news_variant: bool = True
    news_variant_models: list[str] = Field(default_factory=lambda: ["lightgbm"])
    train_telegram_variant: bool = True
    telegram_variant_models: list[str] = Field(default_factory=lambda: ["lightgbm"])
    serve_regions: Any = "auto"


class ServiceCfg(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    model_version: str = "1.0.0"


class Config(BaseModel):
    project: ProjectCfg = Field(default_factory=ProjectCfg)
    paths: PathsCfg = Field(default_factory=PathsCfg)
    data: DataCfg
    preprocess: PreprocessCfg = Field(default_factory=PreprocessCfg)
    panel: PanelCfg = Field(default_factory=PanelCfg)
    features: FeaturesCfg = Field(default_factory=FeaturesCfg)
    targets: TargetsCfg = Field(default_factory=TargetsCfg)
    modeling: ModelingCfg = Field(default_factory=ModelingCfg)
    backtest: BacktestCfg = Field(default_factory=BacktestCfg)
    production: ProductionCfg = Field(default_factory=ProductionCfg)
    service: ServiceCfg = Field(default_factory=ServiceCfg)

    # Absolute project root, injected at load time (not part of the YAML).
    project_root: Path = PROJECT_ROOT

    def ensure_dirs(self) -> None:
        """Create all output directories if they do not yet exist."""
        for path in (
            self.paths.raw_dir,
            self.paths.external_dir,
            self.paths.processed_dir,
            self.paths.models_dir,
            self.paths.reports_dir,
            self.paths.figures_dir,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)


def _config_path() -> Path:
    env = os.environ.get("ARF_CONFIG")
    if env:
        p = Path(env)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return DEFAULT_CONFIG_PATH


@lru_cache(maxsize=4)
def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load and validate the YAML config, resolving all paths to absolutes.

    Results are cached by path so repeated calls are cheap. Pass an explicit
    *path* (e.g. in tests) to bypass the ``ARF_CONFIG`` env var.
    """
    cfg_path = Path(path) if path is not None else _config_path()
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    cfg = Config(**raw)
    cfg.paths = cfg.paths.resolve(PROJECT_ROOT)
    cfg.project_root = PROJECT_ROOT
    return cfg
