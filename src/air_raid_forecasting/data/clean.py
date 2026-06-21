"""Phase 2 — Data validation & cleaning.

Turns raw alert rows into a consistent, validated table of **consolidated
oblast-level alert events**, then derives the calendar fields required by the
brief (``duration_minutes``, ``hour_of_day``, ``day_of_week``, ``month``,
``season``, ``weekend_flag``).

Why consolidate to oblast level?
--------------------------------
Since late 2025 alerts are issued at raion/hromada (district/community) level,
so a single oblast-wide threat now produces *many* rows where it used to
produce one. Counting raw rows would manufacture an artificial regime shift in
late 2025. We therefore merge all overlapping alert intervals within an oblast
into unified "the oblast was under alert during [start, end)" events — a
definition that is stable across the entire 2022-2026 history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from air_raid_forecasting.config import Config, load_config
from air_raid_forecasting.data.ingest import load_raw_alerts
from air_raid_forecasting.data.regions import normalize_region
from air_raid_forecasting.logging_utils import get_logger

log = get_logger(__name__)

_SEASONS = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}


@dataclass
class CleaningReport:
    """Audit trail of what the cleaner did — surfaced in logs and notebooks."""

    source: str
    raw_rows: int = 0
    dropped_unmapped_region: int = 0
    dropped_excluded_region: int = 0
    dropped_bad_timestamps: int = 0
    dropped_negative_duration: int = 0
    dropped_too_short: int = 0
    duplicates_removed: int = 0
    raw_intervals: int = 0
    consolidated_events: int = 0
    anomalies_flagged: int = 0
    date_min: Any = None
    date_max: Any = None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["date_min"] = str(self.date_min)
        d["date_max"] = str(self.date_max)
        return d


def _merge_region_intervals(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge overlapping/touching [start, end) intervals (already sorted by start).

    Returns merged ``(starts, ends, n_subalerts)`` where ``n_subalerts`` counts
    how many raw intervals folded into each consolidated event.
    """
    n = len(starts)
    m_start = np.empty(n, dtype=starts.dtype)
    m_end = np.empty(n, dtype=ends.dtype)
    m_count = np.empty(n, dtype=np.int64)

    cur_s, cur_e, cnt = starts[0], ends[0], 1
    k = 0
    for i in range(1, n):
        s, e = starts[i], ends[i]
        if s <= cur_e:  # overlap or touch -> extend current event
            if e > cur_e:
                cur_e = e
            cnt += 1
        else:
            m_start[k], m_end[k], m_count[k] = cur_s, cur_e, cnt
            k += 1
            cur_s, cur_e, cnt = s, e, 1
    m_start[k], m_end[k], m_count[k] = cur_s, cur_e, cnt
    k += 1
    return m_start[:k], m_end[:k], m_count[:k]


def consolidate_to_oblast(raw: pd.DataFrame, cfg: Config, report: CleaningReport) -> pd.DataFrame:
    """Normalize regions, validate timestamps, and merge to oblast events."""
    df = raw.copy()
    report.raw_rows = len(df)

    # Region normalization.
    df["region"] = df["region_raw"].map(normalize_region)
    unmapped = df["region"].isna()
    report.dropped_unmapped_region = int(unmapped.sum())
    df = df[~unmapped]

    excluded = set(cfg.data.exclude_permanent_sirens)
    if excluded:
        is_excluded = df["region"].isin(excluded)
        report.dropped_excluded_region = int(is_excluded.sum())
        df = df[~is_excluded]

    # Timestamp validation.
    bad_ts = df["started_at"].isna() | df["finished_at"].isna()
    report.dropped_bad_timestamps = int(bad_ts.sum())
    df = df[~bad_ts]

    neg = df["finished_at"] <= df["started_at"]
    report.dropped_negative_duration = int(neg.sum())
    df = df[~neg]

    # Drop impossibly short raw alerts before merging.
    dur_s = (df["finished_at"] - df["started_at"]).dt.total_seconds()
    too_short = dur_s < cfg.preprocess.min_duration_seconds
    report.dropped_too_short = int(too_short.sum())
    df = df[~too_short]

    # Exact duplicates.
    before = len(df)
    df = df.drop_duplicates(subset=["region", "started_at", "finished_at"])
    report.duplicates_removed = before - len(df)
    report.raw_intervals = len(df)

    # Merge overlapping intervals within each oblast.
    events: list[pd.DataFrame] = []
    for region, grp in df.sort_values("started_at").groupby("region", sort=False):
        s = grp["started_at"].to_numpy()
        e = grp["finished_at"].to_numpy()
        ms, me, mc = _merge_region_intervals(s, e)
        events.append(
            pd.DataFrame({"region": region, "started_at": ms, "finished_at": me, "n_subalerts": mc})
        )
    out = pd.concat(events, ignore_index=True)
    out["started_at"] = pd.to_datetime(out["started_at"], utc=True)
    out["finished_at"] = pd.to_datetime(out["finished_at"], utc=True)
    report.consolidated_events = len(out)
    return out.sort_values(["started_at", "region"]).reset_index(drop=True)


def add_derived_fields(events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add duration + calendar features (computed in local Europe/Kyiv time)."""
    df = events.copy()
    df["duration_minutes"] = (
        (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60.0
    ).round(2)

    local = df["started_at"].dt.tz_convert(cfg.project.timezone_local)
    df["start_local"] = local
    df["date"] = local.dt.date
    df["year"] = local.dt.year
    df["month"] = local.dt.month
    df["hour_of_day"] = local.dt.hour
    df["day_of_week"] = local.dt.dayofweek  # Monday=0
    df["weekend_flag"] = (df["day_of_week"] >= 5).astype(int)
    df["season"] = df["month"].map(_SEASONS)
    return df


def flag_anomalies(events: pd.DataFrame, cfg: Config, report: CleaningReport) -> pd.DataFrame:
    """Flag (not drop) duration anomalies via robust log-duration z-score + hard cap."""
    df = events.copy()
    dur = df["duration_minutes"].clip(lower=0.1)
    logd = np.log(dur)
    med = logd.median()
    mad = (logd - med).abs().median()
    robust_sigma = 1.4826 * mad if mad > 0 else logd.std(ddof=0)
    z = (logd - med) / (robust_sigma if robust_sigma > 0 else 1.0)
    cap_minutes = cfg.preprocess.max_duration_hours * 60
    df["is_anomaly_duration"] = (
        (z.abs() > cfg.preprocess.anomaly_zscore_threshold)
        | (df["duration_minutes"] > cap_minutes)
    )
    report.anomalies_flagged = int(df["is_anomaly_duration"].sum())
    return df


def clean(
    cfg: Config | None = None,
    source: str | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Full Phase-2 pipeline: load raw -> consolidate -> derive -> flag.

    Returns the cleaned consolidated-events frame and a :class:`CleaningReport`.
    """
    cfg = cfg or load_config()
    source = source or cfg.data.primary_source
    report = CleaningReport(source=source)

    raw = load_raw_alerts(cfg, source=source)
    events = consolidate_to_oblast(raw, cfg, report)
    events = add_derived_fields(events, cfg)
    events = flag_anomalies(events, cfg, report)

    report.date_min = events["started_at"].min()
    report.date_max = events["started_at"].max()
    log.info(
        "Cleaned %s: %s raw rows -> %s consolidated oblast events (%s regions, %s anomalies flagged)",
        source, f"{report.raw_rows:,}", f"{report.consolidated_events:,}",
        events["region"].nunique(), f"{report.anomalies_flagged:,}",
    )
    return events, report
