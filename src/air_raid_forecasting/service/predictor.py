"""Serving logic: turn a model bundle into per-region forecasts (Phase 11).

The :class:`Predictor` loads the trained :class:`ModelBundle` once and answers
requests of the form *(region, horizon)* by:

1. snapping the requested horizon to the nearest trained horizon,
2. feeding the region's latest known feature row to the count regressor and the
   probability classifier for that horizon,
3. predicting expected duration from calendar/region features of "now",
4. deriving a severity class from the predicted duration's distribution band,
5. reporting a confidence score from the classifier's decisiveness.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from air_raid_forecasting.data.regions import normalize_region
from air_raid_forecasting.features.targets import severity_from_duration
from air_raid_forecasting.logging_utils import get_logger
from air_raid_forecasting.models.persistence import ModelBundle

log = get_logger(__name__)
_SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}


def _nearest(value: int, options: list[int]) -> int:
    return min(options, key=lambda o: abs(o - value))


class Predictor:
    def __init__(self, bundle: ModelBundle, tz: str = "Europe/Kyiv") -> None:
        self.b = bundle
        self.tz = tz
        self._latest = bundle.latest_features.set_index("region")
        self._feature_cols = bundle.feature_meta["feature_cols"]

    @classmethod
    def from_dir(cls, models_dir: str | Path, tz: str = "Europe/Kyiv") -> "Predictor":
        return cls(ModelBundle.load(models_dir), tz=tz)

    @property
    def regions(self) -> list[str]:
        return self.b.regions

    def _latest_row(self, region: str) -> pd.DataFrame:
        row = self._latest.loc[[region]].copy()
        return row[self._feature_cols]

    def _duration_frame(self, region: str, now: datetime) -> pd.DataFrame:
        local = now.astimezone(ZoneInfo(self.tz))
        return pd.DataFrame([{
            "region": region,
            "season": _SEASONS[local.month],
            "hour_of_day": local.hour,
            "day_of_week": local.weekday(),
            "month": local.month,
            "weekend_flag": int(local.weekday() >= 5),
            "n_subalerts": 1,
        }])

    def predict_one(self, region: str, horizon_hours: int, now: datetime | None = None) -> dict:
        canonical = normalize_region(region)
        if canonical is None or canonical not in self._latest.index:
            raise KeyError(
                f"Unknown region {region!r}. Known regions: {', '.join(self.regions)}"
            )
        now = now or datetime.now(timezone.utc)

        count_h = _nearest(horizon_hours, self.b.count_horizons)
        proba_h = _nearest(horizon_hours, self.b.proba_windows)
        row = self._latest_row(canonical)

        count = float(self.b.count_models[count_h].predict(row, row, None)[0])
        proba = float(self.b.proba_models[proba_h].predict(row, row, None)[0])
        duration = float(self.b.duration_model.predict(self._duration_frame(canonical, now))[0])

        severity = str(severity_from_duration(
            duration, self.b.severity_thresholds, self.b.severity_labels))
        confidence = round(max(proba, 1.0 - proba), 4)

        return {
            "region": canonical,
            "forecast_horizon_hours": horizon_hours,
            "matched_horizon_hours": count_h,
            "alert_probability": round(proba, 4),
            "predicted_alert_count": int(round(count)),
            "predicted_duration_minutes": round(duration, 1),
            "severity": severity,
            "confidence": confidence,
            "model_version": self.b.version,
            "as_of": now.isoformat(),
        }

    def predict_batch(self, items: list[tuple[str, int]]) -> list[dict]:
        now = datetime.now(timezone.utc)
        return [self.predict_one(r, h, now=now) for r, h in items]
