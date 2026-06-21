"""Serving logic: turn a model bundle into per-region forecasts (Phase 11).

Supports the interactive bundle: the caller can choose the **model** (a specific
algorithm or ``"best"`` = best per-region backtest) and toggle the **news**
factor (switching to the news-augmented model variant). Falls back gracefully
to the legacy single-model bundle layout.
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
        self.v2 = bool(getattr(bundle, "variants", None))
        if self.v2:
            self._latest = {
                v: bundle.variants[v]["latest_features"].set_index("region")
                for v in bundle.variants
            }
            self._feature_cols = {v: bundle.variants[v]["feature_cols"] for v in bundle.variants}
            self.variants = bundle.available_variants or list(bundle.variants.keys())
            self.models = bundle.available_models or ["lightgbm"]
        else:  # legacy flat bundle
            self._latest = {"base": bundle.latest_features.set_index("region")}
            self._feature_cols = {"base": bundle.feature_meta["feature_cols"]}
            self.variants = ["base"]
            self.models = ["lightgbm"]

    @classmethod
    def from_dir(cls, models_dir: str | Path, tz: str = "Europe/Kyiv") -> "Predictor":
        return cls(ModelBundle.load(models_dir), tz=tz)

    # -- introspection used by the dashboard / API ------------------------
    @property
    def regions(self) -> list[str]:
        return self.b.regions

    def has_news(self) -> bool:
        return "news" in self.variants

    def best_model(self, variant: str = "base") -> str:
        per_model = getattr(self.b, "per_model_metrics", {}) or {}
        metrics = {m: v for m, v in per_model.get(variant, {}).get("count_1h", {}).items()
                   if v is not None}
        if metrics:  # lowest MAE wins
            return min(metrics, key=metrics.get)
        best = getattr(self.b, "best_count_model_name", "") or ""
        return best if best in self.models else self.models[0]

    # -- prediction -------------------------------------------------------
    def _duration_frame(self, region: str, now: datetime) -> pd.DataFrame:
        local = now.astimezone(ZoneInfo(self.tz))
        return pd.DataFrame([{
            "region": region, "season": _SEASONS[local.month], "hour_of_day": local.hour,
            "day_of_week": local.weekday(), "month": local.month,
            "weekend_flag": int(local.weekday() >= 5), "n_subalerts": 1,
        }])

    def _models_for(self, variant: str, horizon: int):
        """Return (count_model, proba_model, matched_count_h, matched_proba_h)."""
        count_h = _nearest(horizon, self.b.count_horizons)
        proba_h = _nearest(horizon, self.b.proba_windows)
        if self.v2:
            return None, None, count_h, proba_h  # resolved with model name in predict_one
        return self.b.count_models[count_h], self.b.proba_models[proba_h], count_h, proba_h

    def predict_one(self, region: str, horizon_hours: int, model: str = "best",
                    use_news: bool = False, now: datetime | None = None) -> dict:
        canonical = normalize_region(region)
        variant = "news" if (use_news and self.has_news()) else "base"
        if canonical is None or canonical not in self._latest[variant].index:
            raise KeyError(f"Unknown region {region!r}. Known: {', '.join(self.regions)}")
        now = now or datetime.now(timezone.utc)

        count_h = _nearest(horizon_hours, self.b.count_horizons)
        proba_h = _nearest(horizon_hours, self.b.proba_windows)
        model_name = self.best_model(variant) if model in (None, "best", "Best (auto)") else model

        row = self._latest[variant].loc[[canonical]][self._feature_cols[variant]]
        if self.v2:
            cmodels = self.b.variants[variant]["count"][count_h]
            pmodels = self.b.variants[variant]["proba"][proba_h]
            # The chosen model may not be trained for this variant (e.g. the news
            # variant only has lightgbm) — fall back to the best available one.
            if model_name not in cmodels:
                best = self.best_model(variant)
                model_name = best if best in cmodels else next(iter(cmodels))
            cm = cmodels[model_name]
            pm = pmodels.get(model_name) or next(iter(pmodels.values()))
        else:
            cm = self.b.count_models[count_h]
            pm = self.b.proba_models[proba_h]
            model_name = "lightgbm"

        count = float(cm.predict(row, row, None)[0])
        proba = float(pm.predict(row, row, None)[0])
        duration = float(self.b.duration_model.predict(self._duration_frame(canonical, now))[0])
        severity = str(severity_from_duration(duration, self.b.severity_thresholds,
                                              self.b.severity_labels))
        return {
            "region": canonical,
            "forecast_horizon_hours": horizon_hours,
            "matched_horizon_hours": count_h,
            "alert_probability": round(proba, 4),
            "predicted_alert_count": int(round(count)),
            "predicted_duration_minutes": round(duration, 1),
            "severity": severity,
            "confidence": round(max(proba, 1.0 - proba), 4),
            "model": model_name,
            "news_factor": variant == "news",
            "model_version": self.b.version,
            "as_of": now.isoformat(),
        }

    def predict_batch(self, items: list, model: str = "best", use_news: bool = False) -> list[dict]:
        """Items are (region, horizon) tuples; model/news apply to all."""
        now = datetime.now(timezone.utc)
        return [self.predict_one(r, h, model=model, use_news=use_news, now=now) for r, h in items]
