"""Assemble leakage-safe feature matrices for modeling (Phase 4/5).

Two products mirror the two modeling tasks:

* **national features** — one row per hour; feeds the headline multi-model
  backtest on Target A (count) and Target B (probability of a new alert).
* **region features** — one row per (region, hour); feeds the global
  gradient-boosted production model that serves any region & horizon.

Feature philosophy: the row at time *t* contains only information knowable by
the end of hour ``t-1`` (lags >= 1, shifted rolling windows, recency). The
contemporaneous raw counts are *not* used as features, so the model is honest
about real-time deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from air_raid_forecasting.config import Config, load_config
from air_raid_forecasting.features.calendar import CALENDAR_FEATURES, add_calendar_features
from air_raid_forecasting.features.targets import add_count_targets, add_proba_targets
from air_raid_forecasting.features.timeseries import add_lags, add_rolling, add_time_since_last
from air_raid_forecasting.logging_utils import get_logger

log = get_logger(__name__)

# Columns that are never features (raw signals, identifiers, targets).
_NON_FEATURE = {
    "timestamp", "region", "region_cat", "start_flag",
    "alerts_started", "any_alert", "alert_minutes",
    "regions_under_alert", "alert_minutes_total",
}


@dataclass
class FeatureMeta:
    feature_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    count_targets: list[str] = field(default_factory=list)
    proba_targets: list[str] = field(default_factory=list)
    target_value_col: str = "alerts_started"


def _collect_feature_cols(df: pd.DataFrame, extra: list[str]) -> list[str]:
    cols = [
        c for c in df.columns
        if c not in _NON_FEATURE and not c.startswith("target_")
    ]
    for e in extra:
        if e in df.columns and e not in cols:
            cols.append(e)
    return cols


def _downcast(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].astype("float32")
    return df


def build_national_features(national: pd.DataFrame, cfg: Config | None = None) -> tuple[pd.DataFrame, FeatureMeta]:
    cfg = cfg or load_config()
    df = national.sort_values("timestamp").reset_index(drop=True).copy()
    df["start_flag"] = (df["alerts_started"] > 0).astype(int)

    df = add_calendar_features(
        df, "timestamp", cfg.project.timezone_local,
        cfg.features.holiday_country, cfg.features.use_holidays,
    )
    df = add_lags(df, ["alerts_started", "regions_under_alert", "start_flag"],
                  cfg.features.lags_hours)
    df = add_rolling(df, ["alerts_started", "regions_under_alert", "alert_minutes_total"],
                     cfg.features.rolling_windows_hours, cfg.features.rolling_stats)
    df = add_time_since_last(df, "start_flag")

    df, count_t = add_count_targets(df, "alerts_started", cfg.targets.count_horizons_hours)
    df, proba_t = add_proba_targets(df, "start_flag", cfg.targets.proba_windows_hours)

    feats = _collect_feature_cols(df, CALENDAR_FEATURES + ["time_since_last_alert"])
    df[feats] = df[feats].fillna(0)
    df = _downcast(df, feats)
    meta = FeatureMeta(feature_cols=feats, categorical_cols=[],
                       count_targets=count_t, proba_targets=proba_t)
    log.info("National features: %s rows x %s features", f"{len(df):,}", len(feats))
    return df, meta


def build_region_features(region_panel: pd.DataFrame, cfg: Config | None = None) -> tuple[pd.DataFrame, FeatureMeta]:
    cfg = cfg or load_config()
    df = region_panel.sort_values(["region", "timestamp"]).reset_index(drop=True).copy()

    df = add_calendar_features(
        df, "timestamp", cfg.project.timezone_local,
        cfg.features.holiday_country, cfg.features.use_holidays,
    )
    df = add_lags(df, ["alerts_started", "any_alert", "alert_minutes"],
                  cfg.features.lags_hours, group_col="region")
    df = add_rolling(df, ["alerts_started", "alert_minutes"],
                     cfg.features.rolling_windows_hours, cfg.features.rolling_stats,
                     group_col="region")
    df = add_time_since_last(df, "any_alert", group_col="region")

    df, count_t = add_count_targets(df, "alerts_started", cfg.targets.count_horizons_hours,
                                    group_col="region")
    df, proba_t = add_proba_targets(df, "any_alert", cfg.targets.proba_windows_hours,
                                    group_col="region")

    # Region as an explicit categorical feature for the global model.
    df["region_cat"] = df["region"].astype("category")

    feats = _collect_feature_cols(df, CALENDAR_FEATURES + ["time_since_last_alert", "region_cat"])
    num_feats = [c for c in feats if c != "region_cat"]
    df[num_feats] = df[num_feats].fillna(0)
    df = _downcast(df, num_feats)
    meta = FeatureMeta(feature_cols=feats, categorical_cols=["region_cat"],
                       count_targets=count_t, proba_targets=proba_t)
    log.info("Region features: %s rows x %s features", f"{len(df):,}", len(feats))
    return df, meta
