# Final Report — Air Raid Alert Forecasting in Ukraine

_A miniature defense-analytics platform. Auto-generated from pipeline artifacts._

## 1. Data

- **Events:** 74,675 consolidated oblast-level alerts across **24** regions.
- **Span:** 2022-03-15 16:10:34+00:00 → 2026-06-19 23:47:31+00:00.
- **Average alerts/day:** 47.96.
- **Duration (min):** mean 124.99, median 44.22, p90 225.03, p99 804.17.
- **Most-affected regions:** Zaporizhzhia (7,920), Donetsk (7,150), Sumy (5,702), Poltava (5,411), Kharkiv (5,114), Mykolaiv (4,819).

Source: Vadimkin *ukrainian-air-raid-sirens-dataset* (live download, UTC). Provenance recorded in `data/raw/_manifest.json`.

## 2. EDA highlights

- **Stationarity:** ADF p=0.0, KPSS p=0.01 (classic trend-stationary signature).
- **STL strengths:** seasonal 0.149, trend 0.281.
- Figures in `reports/figures/` (daily series, hour×weekday heatmap, regional intensity, duration distribution, STL decomposition, ACF/PACF).

## 3. Forecasting targets

A: alert count (next 1/6/24h) · B: alert probability (next 1/6/24h) · C: expected duration · D: severity (Low/Medium/High/Critical, derived from duration quantiles).

## 4. Model comparison — national hourly count (1-step-ahead)

**Best model: `catboost`** (by MAE, expanding-window backtest).

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

## 5. Model comparison — per-region P(alert within 6h)

_Not available — run training._

## 6. Iterative optimization

- Tracked LightGBM search → best CV MAE **1.6660** (naive baseline 2.2743055555555554).
- Best params: `{'n_estimators': 300, 'num_leaves': 31, 'learning_rate': 0.05}`
- Full history in `reports/experiments.jsonl`.

## 7. Production model

A global, region-aware gradient-boosted model serves any region & horizon for Targets A & B; auxiliary models cover duration (C) and severity (D).

**Production region count (H=1) backtest:**
```
{
  "MAE": 0.17024041816199934,
  "RMSE": 0.3099420405368962,
  "MAPE": 81.48459871962243,
  "MAPE_coverage": 0.1025462962962963,
  "SMAPE": 190.66388234804978,
  "fit_seconds": 11.685
}
```

**Severity classifier:** {"accuracy": 0.46113157013726147, "f1_macro": 0.4314503693670476}

## 8. Explainability

Top drivers (mean |SHAP| / importance):
- `alerts_started_roll_mean_168`: 0.02946
- `region_cat`: 0.00933
- `hour`: 0.00885
- `alert_minutes_lag_1`: 0.00724
- `alerts_started_roll_std_168`: 0.00643
- `alerts_started_roll_sum_24`: 0.00476
- `alert_minutes_roll_sum_24`: 0.00379
- `alert_minutes_roll_std_3`: 0.00364
- `alerts_started_roll_mean_72`: 0.0033
- `alerts_started_roll_sum_168`: 0.00294

See `reports/figures/production_count_shap_summary.png`.

## 9. Deliverables

Source package, 6 notebooks, multi-model pipeline, backtesting & comparison frameworks, FastAPI service, Streamlit dashboard, pytest suite, Dockerized deployment, and this report.

> **Ethical note:** analytical use on open historical data only — not an early-warning system.
