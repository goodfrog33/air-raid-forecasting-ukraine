### National hourly alert-count — 1-step-ahead model comparison

| rank | model | family | MAE | RMSE | SMAPE | MAPE |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | catboost | ml | 1.6629 | 2.5406 | 71.9892 | 63.4837 |
| 2 | random_forest | ml | 1.6866 | 2.5648 | 72.5050 | 64.2790 |
| 3 | prophet | advanced | 1.6995 | 2.5773 | 73.0551 | 65.1538 |
| 4 | ets | statistical | 1.7089 | 2.5730 | 72.7813 | 67.1918 |
| 5 | moving_average | baseline | 1.7373 | 2.6049 | 73.7734 | 68.8264 |
| 6 | lightgbm | ml | 1.7546 | 2.5904 | 73.7844 | 70.8538 |
| 7 | xgboost | ml | 1.8287 | 2.6537 | 74.3037 | 75.6325 |
| 8 | naive | baseline | 2.2743 | 3.5250 | 93.4422 | 94.7047 |
| 9 | seasonal_naive | baseline | 2.3028 | 3.5728 | 92.2853 | 94.4372 |