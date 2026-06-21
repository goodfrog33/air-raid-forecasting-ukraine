"""Gradient-boosted & random-forest forecasters (Phase 6).

A thin, uniform wrapper turns scikit-learn-style estimators into
:class:`Forecaster` objects that the backtester can drive. The same class
serves regression (count) and classification (probability) by switching the
underlying estimator on ``ctx.task``.

Categorical handling
--------------------
The only categorical feature in this project is ``region_cat`` (used by the
global per-region production model). LightGBM and CatBoost consume it natively;
RandomForest and XGBoost get an ordinal encoding learned on the training fold
(unseen categories -> -1). For the national headline series there are no
categoricals, so this path is inert there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from air_raid_forecasting.models.base import Forecaster, ModelContext, clip_nonneg


class _BaseTreeForecaster(Forecaster):
    family = "ml"
    cat_handling = "ordinal"  # "ordinal" | "native" | "catboost"

    def __init__(self, task: str = "count", params: dict | None = None) -> None:
        self.task = task
        self.params = params or {}
        self.model = None
        self.feature_cols: list[str] = []
        self.cat_cols: list[str] = []
        self._cat_maps: dict[str, dict] = {}

    # subclasses build the concrete estimator
    def _make_estimator(self, task: str, seed: int):  # pragma: no cover - overridden
        raise NotImplementedError

    def _prepare_X(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X = df[self.feature_cols].copy()
        if not self.cat_cols:
            return X
        if self.cat_handling == "native":
            for c in self.cat_cols:
                X[c] = X[c].astype("category")
        elif self.cat_handling == "catboost":
            for c in self.cat_cols:
                X[c] = X[c].astype(str)
        else:  # ordinal
            for c in self.cat_cols:
                if fit:
                    cats = pd.Index(df[c].astype(str).unique())
                    self._cat_maps[c] = {v: i for i, v in enumerate(cats)}
                m = self._cat_maps.get(c, {})
                X[c] = df[c].astype(str).map(m).fillna(-1).astype(int)
        return X

    def fit(self, train: pd.DataFrame, ctx: ModelContext) -> "_BaseTreeForecaster":
        self.task = ctx.task
        self.feature_cols = ctx.feature_cols
        self.cat_cols = [c for c in ctx.categorical_cols if c in ctx.feature_cols]
        X = self._prepare_X(train, fit=True)
        y = self._target(train, ctx)
        if ctx.task == "proba":
            y = (y > 0).astype(int)
        self.model = self._make_estimator(ctx.task, ctx.seed)
        self._fit_estimator(X, y)
        return self

    def _fit_estimator(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict(self, train: pd.DataFrame, test: pd.DataFrame, ctx: ModelContext) -> np.ndarray:
        X = self._prepare_X(test, fit=False)
        if self.task == "proba":
            proba = self.model.predict_proba(X)
            return proba[:, 1]
        return clip_nonneg(np.asarray(self.model.predict(X), dtype=float))

    def feature_importance(self) -> pd.Series | None:
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return None
        return pd.Series(self.model.feature_importances_, index=self.feature_cols).sort_values(
            ascending=False
        )


class RandomForestForecaster(_BaseTreeForecaster):
    name = "random_forest"
    cat_handling = "ordinal"

    def _make_estimator(self, task: str, seed: int):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        # Depth is capped: unbounded trees on tens of thousands of rows are very
        # slow and the SIGALRM timeout can't interrupt scikit-learn's C code.
        kw = dict(
            n_estimators=self.params.get("n_estimators", 120),
            max_depth=self.params.get("max_depth", 12),
            min_samples_leaf=self.params.get("min_samples_leaf", 20),
            n_jobs=-1,
            random_state=seed,
        )
        return RandomForestClassifier(**kw) if task == "proba" else RandomForestRegressor(**kw)


class XGBoostForecaster(_BaseTreeForecaster):
    name = "xgboost"
    cat_handling = "ordinal"

    def _make_estimator(self, task: str, seed: int):
        import xgboost as xgb

        kw = dict(
            n_estimators=self.params.get("n_estimators", 400),
            max_depth=self.params.get("max_depth", 6),
            learning_rate=self.params.get("learning_rate", 0.05),
            subsample=self.params.get("subsample", 0.8),
            colsample_bytree=self.params.get("colsample_bytree", 0.8),
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
        if task == "proba":
            return xgb.XGBClassifier(eval_metric="logloss", **kw)
        return xgb.XGBRegressor(objective="reg:squarederror", **kw)


class LightGBMForecaster(_BaseTreeForecaster):
    name = "lightgbm"
    cat_handling = "native"

    def _make_estimator(self, task: str, seed: int):
        import lightgbm as lgb

        kw = dict(
            n_estimators=self.params.get("n_estimators", 500),
            num_leaves=self.params.get("num_leaves", 63),
            learning_rate=self.params.get("learning_rate", 0.05),
            subsample=self.params.get("subsample", 0.8),
            colsample_bytree=self.params.get("colsample_bytree", 0.8),
            min_child_samples=self.params.get("min_child_samples", 30),
            n_jobs=-1,
            random_state=seed,
            verbosity=-1,
        )
        return lgb.LGBMClassifier(**kw) if task == "proba" else lgb.LGBMRegressor(**kw)


class CatBoostForecaster(_BaseTreeForecaster):
    name = "catboost"
    cat_handling = "catboost"

    def _make_estimator(self, task: str, seed: int):
        from catboost import CatBoostClassifier, CatBoostRegressor

        kw = dict(
            iterations=self.params.get("iterations", 500),
            depth=self.params.get("depth", 8),
            learning_rate=self.params.get("learning_rate", 0.05),
            random_seed=seed,
            verbose=0,
            allow_writing_files=False,
        )
        if task == "proba":
            return CatBoostClassifier(loss_function="Logloss", **kw)
        return CatBoostRegressor(loss_function="RMSE", **kw)

    def _fit_estimator(self, X: pd.DataFrame, y: np.ndarray) -> None:
        cat_idx = [X.columns.get_loc(c) for c in self.cat_cols] if self.cat_cols else None
        self.model.fit(X, y, cat_features=cat_idx)
