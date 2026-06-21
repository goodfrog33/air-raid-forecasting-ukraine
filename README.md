# 🛡️ Time-Series Analysis & Forecasting of Air Raid Alerts in Ukraine

### 🔴 Live demo: **[air-raid-forecasting-ukraine.streamlit.app](https://air-raid-forecasting-ukraine.streamlit.app/)**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://air-raid-forecasting-ukraine.streamlit.app/)

A complete, production-style **defense-analytics platform** that ingests real
historical Ukrainian air raid alert data, analyzes it, and forecasts future
alert activity — backed by rigorous time-series backtesting, a deployable
FastAPI prediction service, and an interactive Streamlit dashboard.

> This is an analytics/forecasting exercise on **open, aggregated, historical**
> data. It is **not** an early-warning system and must not be used as one — see
> [Ethical note & limitations](#-ethical-note--limitations).

---

## Highlights

- **Live data acquisition** from the public, token-free
  [Vadimkin air-raid-sirens dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
  (official + volunteer feeds, UTC, 2022→present). No synthetic data.
- **Honest cleaning**: sub-region (raion/hromada) alerts are merged into
  consolidated **oblast-level** intervals so the series is consistent across the
  whole 2022–2026 history.
- **Four forecasting targets**: alert count (A), alert probability (B), expected
  duration (C), derived severity (D).
- **12+ models** behind one interface: baselines, ETS/SARIMA, Prophet, an LSTM,
  and RF/XGBoost/LightGBM/CatBoost — compared via **expanding-window backtesting**
  (never a random split), with automatic best-model selection.
- **Explainability** (SHAP + permutation + importance), **experiment tracking**,
  a **FastAPI** service (`/health`, `/metrics`, `/predict`, `/predict_batch`),
  and a **Streamlit** dashboard with a **live choropleth map** of Ukraine that
  runs the model for every oblast and colours them by predicted risk.
- **Interactive controls**: a region-scope filter (all Ukraine or one oblast),
  a model selector (best-auto or a specific algorithm — LightGBM/XGBoost/
  CatBoost), and a **news-factor toggle** (GDELT war-news intensity) that
  switches to a news-augmented model variant and reports the measured backtest
  lift. The API exposes the same via optional `model` / `use_news` fields.
- Full **pytest** suite and **Docker** deployment.

---

## 🌐 Live demo / deploy to Streamlit Community Cloud

**Live app:** https://air-raid-forecasting-ukraine.streamlit.app/ — deployed from
this repo on Streamlit Community Cloud.

The repo is deploy-ready: a compact pre-trained model bundle and the small data
files the dashboard reads are committed, so the app serves immediately with no
training step.

**Deploy your own copy** (free, persistent, runs from this public repo):

1. Open **[share.streamlit.io/deploy?repository=goodfrog33/air-raid-forecasting-ukraine&branch=main&mainModule=dashboard/streamlit_app.py](https://share.streamlit.io/deploy?repository=goodfrog33/air-raid-forecasting-ukraine&branch=main&mainModule=dashboard/streamlit_app.py)**
   (or go to share.streamlit.io → **New app → From existing repo**).
2. Repository `goodfrog33/air-raid-forecasting-ukraine`, branch `main`,
   main file `dashboard/streamlit_app.py`.
3. Click **Deploy**. First build ~2-3 min; you get a public URL like
   `https://air-raid-forecasting-ukraine.streamlit.app`.

Cloud picks up `requirements.txt` (lean serving set), `packages.txt` (`libgomp1`)
and `.streamlit/config.toml` automatically. The `src`-layout package is put on
the path by `dashboard/streamlit_app.py`, so no editable install is needed.

> Re-running the full pipeline locally refreshes the committed artifacts; commit
> and push to update the live app.

---

## Project structure

```
air_raid_forecasting/
├── data/{raw,external,processed}/      # raw downloads, enrichment, modeling tables
├── notebooks/                          # 6 runnable walk-through notebooks
├── src/air_raid_forecasting/
│   ├── config.py  logging_utils.py     # typed YAML config + logging
│   ├── data/      {ingest,clean,panel,regions}.py
│   ├── features/  {calendar,timeseries,targets,build}.py
│   ├── models/    {base,baselines,statistical,ml,advanced,auxiliary,registry,persistence}.py
│   ├── evaluation/{metrics,backtest,compare,explain,experiments}.py
│   ├── pipeline/  run_{ingest,preprocess,eda,features,train,all}.py · cli.py
│   ├── service/   {app,predictor,schemas}.py      # FastAPI
│   └── dashboard/ app.py                           # Streamlit
├── service/main.py    dashboard/streamlit_app.py   # entry points
├── models/  reports/  tests/  scripts/
├── configs/config.yaml                 # single source of truth
├── Dockerfile  docker-compose.yml  Makefile  pyproject.toml  requirements.txt
```

---

## Quickstart

### 1. Install (Python 3.10–3.12)

```bash
# with uv (recommended)
uv venv --python 3.12 .venv
uv pip install -e ".[deep,dev]"      # 'deep' adds torch for the LSTM
# or with pip
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[deep,dev]"
```

### 2. Run the full pipeline

```bash
python -m air_raid_forecasting.pipeline.run_all          # ingest→…→train
# or step by step / via the CLI:
arf ingest && arf preprocess && arf eda && arf features && arf train
# quick smoke run (fewer folds):
python -m air_raid_forecasting.pipeline.run_all --fast
```

Artifacts land in `data/processed/`, `reports/` (figures, comparison tables,
explainability), and `models/model_bundle.joblib`.

### 3. Serve & explore

```bash
make serve        # FastAPI at http://localhost:8000  (docs at /docs)
make dashboard    # Streamlit at http://localhost:8501
make test         # run the test suite
```

### 4. Docker

```bash
docker compose run --rm pipeline       # build data + train (one-off)
docker compose up api dashboard        # API :8000 + dashboard :8501
```

---

## The forecasting problem

| Target | Question | Type | Horizons |
|--------|----------|------|----------|
| **A — Count** | How many alerts in the next *H* hours? | regression | 1, 6, 24 h |
| **B — Probability** | Will an alert occur within the next *H* hours? | classification | 1, 6, 24 h |
| **C — Duration** | How long will an alert last? | regression | per event |
| **D — Severity** | Low / Medium / High / Critical | classification | per event |

Severity classes are **derived** from the empirical duration distribution
(quantile cut-points at p50/p80/p95).

### Two modeling tables

1. **National hourly series** — the headline 1-step-ahead model comparison.
2. **Per-region (oblast × hour) panel** — the global, region-aware
   gradient-boosted model that backs the production service (any region, any
   horizon). Binary targets are evaluated *per region* because the national
   "any alert somewhere" rate is ~93% (degenerate).

---

## Modeling decisions (the *why*)

- **Consolidate to oblast level.** District-level alerts only appear from late
  2025; merging overlapping intervals to oblast events avoids a fake regime
  shift and keeps a stable 2022→2026 definition.
- **No leakage, ever.** Every feature at hour *t* uses only data through *t-1*
  (lags ≥ 1, shifted rolling windows, recency). Targets use an inclusive future
  window `[t, t+H-1]`, so H=1 coincides with the contemporaneous value and ML
  targets line up exactly with the 1-step series models.
- **Rolling-origin backtesting**, expanding window (default 3 folds of 30 days,
  configurable) — never a random split. Optional gap guards boundary leakage.
- **Fair comparison.** Baselines, statistical models, Prophet, the LSTM and the
  tree models all answer the *same* 1-step-ahead question on the *same* folds
  and metrics. Statistical models fit on a trailing window for tractability (a
  documented choice — ARIMA/ETS gain little from years of stale dynamics).
- **Counts have many true zeros**, so MAPE is reported only over non-zero
  actuals (with coverage); MAE/RMSE/SMAPE are the primary metrics.

---

## Prediction API

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"region": "Kyiv", "forecast_horizon_hours": 6}'
```

```json
{
  "region": "Kyiv City",
  "forecast_horizon_hours": 6,
  "alert_probability": 0.72,
  "predicted_alert_count": 1,
  "predicted_duration_minutes": 48.0,
  "severity": "High",
  "confidence": 0.81,
  "model_version": "1.0.0",
  "matched_horizon_hours": 6
}
```

Endpoints: `GET /health`, `GET /metrics`, `POST /predict`, `POST /predict_batch`.
The service loads the best trained bundle automatically from `ARF_MODEL_DIR`.

---

## Results

See [`reports/training_report.md`](reports/training_report.md),
[`reports/count_national_ranking.md`](reports/count_national_ranking.md) and
[`reports/final_report.md`](reports/final_report.md) for the full, regenerated
comparison tables, backtest metrics and explainability. The headline table below
is regenerated by `scripts/build_final_report.py` after each training run.

**Best national count model: `catboost`** (expanding-window backtest, primary metric MAE).

| rank | model | family | MAE | RMSE | SMAPE |
| --- | --- | --- | --- | --- | --- |
| 1 | catboost | ml | 1.6629 | 2.5406 | 71.9892 |
| 2 | random_forest | ml | 1.6866 | 2.5648 | 72.5050 |
| 3 | prophet | advanced | 1.6995 | 2.5773 | 73.0551 |
| 4 | ets | statistical | 1.7089 | 2.5730 | 72.7813 |
| 5 | moving_average | baseline | 1.7373 | 2.6049 | 73.7734 |
| 6 | lightgbm | ml | 1.7546 | 2.5904 | 73.7844 |
| 7 | xgboost | ml | 1.8287 | 2.6537 | 74.3037 |
| 8 | naive | baseline | 2.2743 | 3.5250 | 93.4422 |
| 9 | seasonal_naive | baseline | 2.3028 | 3.5728 | 92.2853 |

_All models beat the naive/seasonal baselines. This table is from the `--fast`
profile (2 folds; LSTM & SARIMA skipped for speed); the full `arf train` run uses
3 folds and the complete 11-model set._


---

## Data source & provenance

| File | Source | Notes |
|------|--------|-------|
| `official_data_en.csv` | Vadimkin dataset | authoritative, oblast/raion/hromada, 2022-03-15→ |
| `volunteer_data_en.csv` | eTryvoga via Vadimkin | oblast level, 2022-02-25→ |
| `states.json` | Vadimkin dataset | administrative metadata |

Every download is recorded with URL, byte count, SHA-256 and timestamp in
`data/raw/_manifest.json`. Luhansk oblast and Crimea are excluded (permanent
sirens, per the upstream dataset notes).

**External enrichment** (weather, news/GDELT) is scaffolded as optional,
disabled-by-default connectors in `configs/config.yaml`; enable to evaluate
their lift. They are off by default because they require network/API keys and
did not justify the added complexity for the core forecasting task.

---

## ⚠️ Ethical note & limitations

This project analyzes **open, aggregated, historical** alert records for
educational and analytical purposes. It is **not** an early-warning system, has
no real-time feed, and its forecasts are statistical expectations — **never**
rely on it for safety decisions; follow official channels. Forecasting models
capture historical regularities and cannot anticipate the intent behind
attacks. Use responsibly.

## License

MIT — see headers. Data © its respective sources under their terms.
```
