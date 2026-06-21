"""Tests for region normalization, cleaning and panel construction."""

from __future__ import annotations

import pandas as pd

from air_raid_forecasting.data.clean import (
    CleaningReport,
    add_derived_fields,
    consolidate_to_oblast,
)
from air_raid_forecasting.data.panel import build_national_hourly, build_region_hourly
from air_raid_forecasting.data.regions import CANONICAL_OBLASTS, normalize_region


def test_normalize_region_variants():
    assert normalize_region("Kyiv City") == "Kyiv City"
    assert normalize_region("kyiv") == "Kyiv City"
    assert normalize_region("Lvivska oblast") == "Lvivska oblast"
    assert normalize_region("Lviv") == "Lvivska oblast"
    assert normalize_region("Nowhere oblast") is None
    assert normalize_region("") is None
    assert normalize_region(None) is None


def test_canonical_count():
    # 24 oblasts + Kyiv City + Sevastopol + Crimea = 27
    assert len(CANONICAL_OBLASTS) == 27


def test_consolidate_merges_overlaps(cfg, raw_alerts):
    report = CleaningReport(source="official")
    events = consolidate_to_oblast(raw_alerts, cfg, report)
    # Negative-duration row dropped; unmapped region dropped.
    assert report.dropped_negative_duration >= 1
    assert report.dropped_unmapped_region >= 1
    # Every consolidated event has finished_at > started_at.
    assert (events["finished_at"] > events["started_at"]).all()
    # Overlapping raion pair collapses into a single oblast event (n_subalerts >= 2 somewhere).
    assert (events["n_subalerts"] >= 2).any()


def test_derived_fields(cfg, raw_alerts):
    report = CleaningReport(source="official")
    events = consolidate_to_oblast(raw_alerts, cfg, report)
    events = add_derived_fields(events, cfg)
    for col in ["duration_minutes", "hour_of_day", "day_of_week", "month", "season", "weekend_flag"]:
        assert col in events.columns
    assert events["hour_of_day"].between(0, 23).all()
    assert events["day_of_week"].between(0, 6).all()
    assert set(events["weekend_flag"].unique()) <= {0, 1}


def test_panel_shapes_and_occupancy(cfg, raw_alerts):
    report = CleaningReport(source="official")
    events = consolidate_to_oblast(raw_alerts, cfg, report)
    events = add_derived_fields(events, cfg)
    region_panel = build_region_hourly(events, cfg)
    # Dense grid: rows == regions * hours
    n_regions = events["region"].nunique()
    n_hours = region_panel["timestamp"].nunique()
    assert len(region_panel) == n_regions * n_hours
    assert region_panel["alert_minutes"].between(0, 60).all()
    assert set(region_panel["any_alert"].unique()) <= {0, 1}

    national = build_national_hourly(region_panel)
    assert (national["alerts_started"] >= 0).all()
    # National starts equal the sum of region starts each hour.
    assert national["alerts_started"].sum() == region_panel["alerts_started"].sum()
