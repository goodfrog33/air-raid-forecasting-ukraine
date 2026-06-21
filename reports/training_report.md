# Training Report

_Generated: 2026-06-21T19:43:10.130457+00:00_

**Best national count model:** `catboost` (primary metric: MAE)

## Optimization
- Best tuned LightGBM MAE: **1.666005935778315**
- Naive baseline MAE: 2.2743055555555554
- Best params: `{'n_estimators': 300, 'num_leaves': 31, 'learning_rate': 0.05}`

## Production region model (count, H=1) — backtest
```
{
  "per_model_MAE": {
    "lightgbm": 0.16983451337152658,
    "xgboost": 0.17280643062369883,
    "catboost": 0.17170355240655324
  }
}
```

## Severity classifier
```
{
  "accuracy": 0.46113157013726147,
  "f1_macro": 0.4314503693670476
}
```