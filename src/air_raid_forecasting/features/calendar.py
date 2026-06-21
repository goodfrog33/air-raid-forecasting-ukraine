"""Calendar / holiday features (Phase 4).

All calendar fields are derived in **local Europe/Kyiv time** because alert
behaviour follows local day/night and weekday rhythms, not UTC. Cyclical
encodings (sin/cos) let tree-free models use hour/day/month without spurious
ordinal jumps at the wrap-around (23->0, Dec->Jan).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as holidays_lib
except Exception:  # pragma: no cover
    holidays_lib = None


def _cyc(values: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    rad = 2 * np.pi * values / period
    return pd.DataFrame({f"{prefix}_sin": np.sin(rad), f"{prefix}_cos": np.cos(rad)}, index=values.index)


def add_calendar_features(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    tz: str = "Europe/Kyiv",
    holiday_country: str = "UA",
    use_holidays: bool = True,
) -> pd.DataFrame:
    """Append calendar features derived from *ts_col* (assumed tz-aware UTC)."""
    out = df.copy()
    ts = pd.to_datetime(out[ts_col], utc=True)
    local = ts.dt.tz_convert(tz)

    out["hour"] = local.dt.hour
    out["day_of_week"] = local.dt.dayofweek
    out["day_of_month"] = local.dt.day
    out["month"] = local.dt.month
    out["week_of_year"] = local.dt.isocalendar().week.astype(int)
    out["is_weekend"] = (local.dt.dayofweek >= 5).astype(int)
    out["is_night"] = ((local.dt.hour >= 22) | (local.dt.hour < 6)).astype(int)

    out = pd.concat(
        [
            out,
            _cyc(out["hour"], 24, "hour"),
            _cyc(out["day_of_week"], 7, "dow"),
            _cyc(out["month"], 12, "month"),
        ],
        axis=1,
    )

    if use_holidays and holidays_lib is not None:
        years = list(range(local.dt.year.min(), local.dt.year.max() + 1))
        try:
            ua = holidays_lib.country_holidays(holiday_country, years=years)
            dates = local.dt.date
            out["is_holiday"] = dates.map(lambda d: d in ua).astype(int)
        except Exception:
            out["is_holiday"] = 0
    else:
        out["is_holiday"] = 0

    return out


CALENDAR_FEATURES = [
    "hour", "day_of_week", "day_of_month", "month", "week_of_year",
    "is_weekend", "is_night", "is_holiday",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
]
