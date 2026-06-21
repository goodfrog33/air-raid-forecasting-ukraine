"""Pipeline step 4: build feature matrices + targets + severity labels.

    python -m air_raid_forecasting.pipeline.run_features

Outputs (under ``data/processed``):
    features_national.parquet      national hourly features + targets
    features_region.parquet        long (region x hour) features + targets
    alerts_events_labeled.parquet  events with derived severity class
    features_meta.json             feature/target column lists + severity thresholds
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd

from air_raid_forecasting.config import load_config
from air_raid_forecasting.features.build import build_national_features, build_region_features
from air_raid_forecasting.features.targets import add_severity_label
from air_raid_forecasting.logging_utils import get_logger

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> dict:
    cfg = load_config()
    cfg.ensure_dirs()
    proc = Path(cfg.paths.processed_dir)

    national = pd.read_parquet(proc / "panel_national_hourly.parquet")
    national["timestamp"] = pd.to_datetime(national["timestamp"], utc=True)
    region = pd.read_parquet(proc / "panel_region_hourly.parquet")
    region["timestamp"] = pd.to_datetime(region["timestamp"], utc=True)
    events = pd.read_parquet(proc / "alerts_events.parquet")

    nat_feats, nat_meta = build_national_features(national, cfg)
    nat_feats.to_parquet(proc / "features_national.parquet", index=False)

    reg_feats, reg_meta = build_region_features(region, cfg)
    reg_feats.to_parquet(proc / "features_region.parquet", index=False)

    # Best-effort news features (GDELT) for the optional news model variant.
    if cfg.production.train_news_variant:
        try:
            from air_raid_forecasting.features.news import (
                NEWS_FEATURES_FILE,
                build_news_features,
                fetch_gdelt_daily,
            )
            lo = region["timestamp"].min().tz_convert(None).strftime("%Y-%m-%d")
            hi = region["timestamp"].max().tz_convert(None).strftime("%Y-%m-%d")
            news_daily = fetch_gdelt_daily(lo, hi, cfg.paths.external_dir, query=cfg.features.news.query)
            build_news_features(news_daily).to_parquet(proc / NEWS_FEATURES_FILE, index=False)
            log.info("  news features built (%s rows)", f"{len(news_daily):,}")
        except Exception as exc:  # news is optional; never block the pipeline
            log.warning("News features skipped (GDELT unavailable): %s", exc)

    if cfg.production.train_telegram_variant:
        try:
            from air_raid_forecasting.features.telegram import (
                TELEGRAM_FEATURES_FILE,
                build_telegram_features,
                fetch_telegram_daily,
            )
            tg_daily = fetch_telegram_daily(cfg.paths.external_dir,
                                            channels=cfg.features.telegram.channels,
                                            max_pages=cfg.features.telegram.max_pages)
            build_telegram_features(tg_daily).to_parquet(proc / TELEGRAM_FEATURES_FILE, index=False)
            log.info("  telegram features built (%s rows)", f"{len(tg_daily):,}")
        except Exception as exc:  # telegram is optional; never block the pipeline
            log.warning("Telegram features skipped: %s", exc)

    sev = cfg.targets.severity
    events_lab, thresholds = add_severity_label(events, sev.quantiles, sev.labels)
    events_lab["severity"] = events_lab["severity"].astype(str)
    events_lab.to_parquet(proc / "alerts_events_labeled.parquet", index=False)
    sev_counts = events_lab["severity"].value_counts().to_dict()

    meta = {
        "national": dataclasses.asdict(nat_meta),
        "region": dataclasses.asdict(reg_meta),
        "severity": {
            "labels": sev.labels,
            "quantiles": sev.quantiles,
            "thresholds_minutes": thresholds,
            "class_counts": {k: int(v) for k, v in sev_counts.items()},
        },
    }
    with open(proc / "features_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    log.info("Features complete.")
    log.info("  national: %s rows x %s feats", f"{len(nat_feats):,}", len(nat_meta.feature_cols))
    log.info("  region:   %s rows x %s feats", f"{len(reg_feats):,}", len(reg_meta.feature_cols))
    log.info("  severity thresholds (min): %s | counts: %s", thresholds, sev_counts)
    return meta


if __name__ == "__main__":
    main()
