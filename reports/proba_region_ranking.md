### Per-region P(alert within 6h) — classification model comparison

| rank | model | family | ROC_AUC | F1 | Accuracy | Precision | Recall | LogLoss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | xgboost | ml | 0.9145 | 0.8426 | 0.8340 | 0.8435 | 0.8417 | 0.3641 |
| 2 | lightgbm | ml | 0.9109 | 0.8409 | 0.8323 | 0.8431 | 0.8387 | 0.3704 |
| 3 | persistence | baseline | 0.7822 | 0.7455 | 0.7735 | 0.9171 | 0.6281 | 0.5052 |
| 4 | prior_rate | baseline | 0.5000 | 0.0000 | 0.4717 | 0.0000 | 0.0000 | 0.7041 |