# Data sources & layout

Raw data is **downloaded at runtime** (live, token-free) and is *not* committed.
Run `python -m air_raid_forecasting.pipeline.run_ingest` to populate `raw/`.

## Layout

| Dir | Contents | Tracked? |
|-----|----------|----------|
| `raw/` | downloaded source CSV/JSON + `_manifest.json` (provenance) | no (regenerated) |
| `external/` | optional enrichment (weather, news) — off by default | no |
| `processed/` | cleaned events, hourly panels, feature matrices, forecasts | no (regenerated) |

## Primary source

[**Vadimkin / ukrainian-air-raid-sirens-dataset**](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
— public, updated daily, all timestamps UTC.

| File | Schema | Coverage |
|------|--------|----------|
| `official_data_en.csv` | `oblast, raion, hromada, level, started_at, finished_at, source` | official records, 2022-03-15 → present |
| `volunteer_data_en.csv` | `region, started_at, finished_at, naive` | eTryvoga volunteer feed, oblast level, 2022-02-25 → present |
| `processors/states.json` | nested oblast → raion → hromada metadata | administrative reference |

The project uses **official** as the canonical source and consolidates all
levels up to oblast intervals. Luhansk oblast and Crimea are excluded
(permanent sirens, per the upstream dataset notes).

## Provenance

`raw/_manifest.json` records, per file: the source URL, byte count, SHA-256
checksum, and download timestamp — so any processed artifact is traceable back
to an exact download.

## Optional external enrichment (Phase 4)

Weather and news/GDELT connectors are scaffolded in `configs/config.yaml`
(`features.weather`, `features.news`), **disabled by default**. They require
network access / API keys and should only be enabled if they demonstrably
improve backtested accuracy.
