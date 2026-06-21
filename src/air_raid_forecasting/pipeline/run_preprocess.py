"""Pipeline step 2: clean raw data and build the hourly modeling panels.

    python -m air_raid_forecasting.pipeline.run_preprocess [--source official]

Outputs (under ``data/processed``):
    alerts_events.parquet          consolidated, validated oblast-level events
    cleaning_report.json           audit trail of the cleaning step
    panel_region_hourly.parquet    long (region x hour) panel
    panel_national_hourly.parquet  national hourly series
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from air_raid_forecasting.config import load_config
from air_raid_forecasting.data.clean import clean
from air_raid_forecasting.data.panel import build_national_hourly, build_region_hourly
from air_raid_forecasting.logging_utils import get_logger

log = get_logger(__name__)

EVENTS_FILE = "alerts_events.parquet"
REPORT_FILE = "cleaning_report.json"
REGION_PANEL_FILE = "panel_region_hourly.parquet"
NATIONAL_PANEL_FILE = "panel_national_hourly.parquet"


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Clean data and build hourly panels.")
    parser.add_argument("--source", default=None, help="official | volunteer (default: config).")
    args = parser.parse_args(argv)

    cfg = load_config()
    cfg.ensure_dirs()
    proc = Path(cfg.paths.processed_dir)

    events, report = clean(cfg, source=args.source)
    events_path = proc / EVENTS_FILE
    # Drop the tz-aware python `date` objects that parquet dislikes; keep a string.
    events_to_save = events.copy()
    events_to_save["date"] = events_to_save["date"].astype(str)
    events_to_save["start_local"] = events_to_save["start_local"].astype(str)
    events_to_save.to_parquet(events_path, index=False)
    log.info("Saved %s events -> %s", f"{len(events):,}", events_path)

    with open(proc / REPORT_FILE, "w", encoding="utf-8") as fh:
        json.dump(report.as_dict(), fh, indent=2)

    region_panel = build_region_hourly(events, cfg)
    region_panel.to_parquet(proc / REGION_PANEL_FILE, index=False)
    national_panel = build_national_hourly(region_panel)
    national_panel.to_parquet(proc / NATIONAL_PANEL_FILE, index=False)

    log.info("Preprocess complete.")
    log.info("  events:          %s", events_path)
    log.info("  region panel:    %s rows", f"{len(region_panel):,}")
    log.info("  national panel:  %s rows", f"{len(national_panel):,}")
    return {
        "events": events,
        "report": report,
        "region_panel": region_panel,
        "national_panel": national_panel,
    }


if __name__ == "__main__":
    main()
