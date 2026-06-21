"""Build the hourly modeling panels from consolidated alert events.

Two products:

* **region panel** — long format, one row per ``(region, hour)`` with:
    - ``alerts_started``  : number of consolidated alerts that *started* that hour
    - ``alert_minutes``   : minutes the region was under alert that hour (0-60)
    - ``any_alert``       : 1 if the region was under alert at any point that hour
* **national panel** — per-hour aggregate across all regions:
    - ``alerts_started``      : total starts that hour
    - ``regions_under_alert`` : how many regions were under alert that hour
    - ``any_alert``           : 1 if at least one region was under alert
    - ``alert_minutes_total`` : summed region-minutes under alert

The occupancy computation is fully vectorized (no per-event Python loop) so it
scales to the ~1M-row full grid in well under a second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from air_raid_forecasting.config import Config, load_config
from air_raid_forecasting.logging_utils import get_logger

log = get_logger(__name__)

_HOUR_NS = 3_600 * 1_000_000_000
_MIN_NS = 60 * 1_000_000_000


def _within_group_offsets(counts: np.ndarray) -> np.ndarray:
    """For counts [c0, c1, ...] return concatenated [0..c0-1, 0..c1-1, ...]."""
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    inc = np.ones(total, dtype=np.int64)
    inc[0] = 0
    boundaries = np.cumsum(counts)[:-1]
    inc[boundaries] = 1 - counts[:-1]
    return np.cumsum(inc)


def build_region_hourly(events: pd.DataFrame, cfg: Config | None = None) -> pd.DataFrame:
    """Construct the dense long (region x hour) panel from consolidated events."""
    cfg = cfg or load_config()
    if events.empty:
        raise ValueError("No events to build a panel from.")

    regions = sorted(events["region"].unique())
    rcode = {r: i for i, r in enumerate(regions)}
    codes = events["region"].map(rcode).to_numpy()

    # tz-aware -> naive UTC wall clock -> int64 ns since epoch (no tz warning).
    s_ns = events["started_at"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype(np.int64)
    e_ns = events["finished_at"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[ns]").astype(np.int64)

    h0 = s_ns // _HOUR_NS                  # first hour bucket touched
    h_last = (e_ns - 1) // _HOUR_NS        # last hour bucket touched (end exclusive)
    counts = (h_last - h0 + 1).astype(np.int64)

    # Vectorized expansion of every event into its hour buckets.
    ev_idx = np.repeat(np.arange(len(events), dtype=np.int64), counts)
    within = _within_group_offsets(counts)
    hour_idx = h0[ev_idx] + within
    hstart = hour_idx * _HOUR_NS
    hend = hstart + _HOUR_NS
    overlap_ns = np.minimum(e_ns[ev_idx], hend) - np.maximum(s_ns[ev_idx], hstart)
    minutes = np.clip(overlap_ns / _MIN_NS, 0.0, 60.0)
    region_exp = codes[ev_idx]

    occ = (
        pd.DataFrame({"region_code": region_exp, "hour_idx": hour_idx, "alert_minutes": minutes})
        .groupby(["region_code", "hour_idx"], as_index=False)["alert_minutes"].sum()
    )

    starts = (
        pd.DataFrame({"region_code": codes, "hour_idx": h0})
        .groupby(["region_code", "hour_idx"], as_index=False)
        .size()
        .rename(columns={"size": "alerts_started"})
    )

    # Dense grid: every region x every hour in range.
    hmin, hmax = int(h0.min()), int(h_last.max())
    all_hours = np.arange(hmin, hmax + 1, dtype=np.int64)
    full_index = pd.MultiIndex.from_product(
        [range(len(regions)), all_hours], names=["region_code", "hour_idx"]
    )
    panel = pd.DataFrame(index=full_index).reset_index()
    panel = panel.merge(occ, on=["region_code", "hour_idx"], how="left")
    panel = panel.merge(starts, on=["region_code", "hour_idx"], how="left")
    panel["alert_minutes"] = panel["alert_minutes"].fillna(0.0)
    panel["alerts_started"] = panel["alerts_started"].fillna(0).astype(int)
    panel["any_alert"] = (panel["alert_minutes"] > 0).astype(int)

    panel["region"] = panel["region_code"].map({i: r for r, i in rcode.items()})
    panel["timestamp"] = pd.to_datetime(panel["hour_idx"] * 3600, unit="s", utc=True)
    panel = panel.drop(columns=["region_code", "hour_idx"])
    panel = panel[["timestamp", "region", "alerts_started", "alert_minutes", "any_alert"]]
    panel = panel.sort_values(["region", "timestamp"]).reset_index(drop=True)
    log.info(
        "Built region panel: %s rows (%s regions x %s hours)",
        f"{len(panel):,}", len(regions), f"{len(all_hours):,}",
    )
    return panel


def build_national_hourly(region_panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the region panel into a single national hourly series."""
    nat = (
        region_panel.groupby("timestamp", as_index=False)
        .agg(
            alerts_started=("alerts_started", "sum"),
            regions_under_alert=("any_alert", "sum"),
            alert_minutes_total=("alert_minutes", "sum"),
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    nat["any_alert"] = (nat["regions_under_alert"] > 0).astype(int)
    log.info("Built national panel: %s hourly rows", f"{len(nat):,}")
    return nat
