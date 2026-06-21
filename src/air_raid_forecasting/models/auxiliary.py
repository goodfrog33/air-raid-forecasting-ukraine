"""Auxiliary production models: alert duration (Target C) and severity (Target D).

These operate on the event table (one row per consolidated alert) rather than
the hourly panel. Kept as small, module-level classes so they serialize cleanly
inside the joblib model bundle. Categorical inputs (region, season) are ordinal-
encoded with maps learned at fit time (unseen -> -1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NUM_FEATURES = ["hour_of_day", "day_of_week", "month", "weekend_flag", "n_subalerts"]
CAT_FEATURES = ["region", "season"]


def _encode(df: pd.DataFrame, cat_maps: dict[str, dict]) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in NUM_FEATURES:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in CAT_FEATURES:
        X[c] = df[c].astype(str).map(cat_maps.get(c, {})).fillna(-1).astype(int)
    return X


class DurationModel:
    """LightGBM regressor predicting alert duration (minutes), trained on log1p."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.model = None
        self.cat_maps: dict[str, dict] = {}

    def fit(self, events: pd.DataFrame) -> "DurationModel":
        import lightgbm as lgb

        for c in CAT_FEATURES:
            cats = pd.Index(events[c].astype(str).unique())
            self.cat_maps[c] = {v: i for i, v in enumerate(cats)}
        X = _encode(events, self.cat_maps)
        y = np.log1p(events["duration_minutes"].clip(lower=0).to_numpy())
        self.model = lgb.LGBMRegressor(
            n_estimators=400, num_leaves=63, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=self.seed, verbosity=-1,
        )
        self.model.fit(X, y)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        X = _encode(frame, self.cat_maps)
        return np.expm1(self.model.predict(X)).clip(min=0)


class SeverityModel:
    """LightGBM multiclass classifier over the derived severity labels."""

    def __init__(self, labels: list[str], seed: int = 42) -> None:
        self.labels = labels
        self.seed = seed
        self.model = None
        self.cat_maps: dict[str, dict] = {}
        self.label_to_idx = {lab: i for i, lab in enumerate(labels)}
        self.idx_to_label = {i: lab for i, lab in enumerate(labels)}

    def fit(self, events: pd.DataFrame) -> "SeverityModel":
        import lightgbm as lgb

        for c in CAT_FEATURES:
            cats = pd.Index(events[c].astype(str).unique())
            self.cat_maps[c] = {v: i for i, v in enumerate(cats)}
        X = _encode(events, self.cat_maps)
        y = events["severity"].astype(str).map(self.label_to_idx)
        keep = y.notna()
        self.model = lgb.LGBMClassifier(
            n_estimators=400, num_leaves=63, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=self.seed, verbosity=-1,
        )
        self.model.fit(X[keep], y[keep].astype(int))
        return self

    def predict(self, frame: pd.DataFrame) -> list[str]:
        X = _encode(frame, self.cat_maps)
        idx = self.model.predict(X)
        return [self.idx_to_label.get(int(i), self.labels[0]) for i in np.atleast_1d(idx)]

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        X = _encode(frame, self.cat_maps)
        return self.model.predict_proba(X)
