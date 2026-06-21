# Training Report

_Generated: 2026-06-21T15:52:27.510745+00:00_

**Best national count model:** `catboost` (primary metric: MAE)

## Optimization
- Best tuned LightGBM MAE: **1.666005935778315**
- Naive baseline MAE: 2.2743055555555554
- Best params: `{'n_estimators': 300, 'num_leaves': 31, 'learning_rate': 0.05}`

## Production region model (count, H=1) — backtest
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

## Severity classifier
```
{
  "accuracy": 0.46113157013726147,
  "f1_macro": 0.4314503693670476
}
```