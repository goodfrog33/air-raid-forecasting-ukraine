"""Advanced forecasters: Prophet, LSTM, and an optional TFT (Phase 6).

* **Prophet** — additive decomposition (trend + daily/weekly seasonality). Its
  forecast for hour *t* is its fitted seasonal+trend curve, so unlike the other
  models it does not exploit recent autocorrelation — an honest representation
  of the decomposition family on a short horizon. Also exposes
  :meth:`forecast_future` (with intervals) for the dashboard.
* **LSTM** — a small recurrent net over the recent value window, giving genuine
  teacher-forced 1-step-ahead predictions.
* **TFT** — Temporal Fusion Transformer, only available if the optional
  ``pytorch-forecasting`` extra is installed; otherwise instantiation raises and
  the registry skips it.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from air_raid_forecasting.logging_utils import get_logger
from air_raid_forecasting.models.base import Forecaster, ModelContext, clip_nonneg

log = get_logger(__name__)


class ProphetForecaster(Forecaster):
    name = "prophet"
    family = "advanced"

    def __init__(self, max_train_obs: int = 0) -> None:
        self.max_train_obs = max_train_obs  # 0 = use full history (trend needs it)
        self.m = None

    def _frame(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        return pd.DataFrame({
            "ds": pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(None),
            "y": df[value_col].astype(float).to_numpy(),
        })

    def fit(self, train: pd.DataFrame, ctx: ModelContext) -> "ProphetForecaster":
        from prophet import Prophet

        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        frame = self._frame(train, ctx.value_col)
        if self.max_train_obs and len(frame) > self.max_train_obs:
            frame = frame.iloc[-self.max_train_obs:]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = Prophet(daily_seasonality=True, weekly_seasonality=True,
                        yearly_seasonality=False, interval_width=0.8)
            m.fit(frame)
        self.m = m
        return self

    def predict(self, train: pd.DataFrame, test: pd.DataFrame, ctx: ModelContext) -> np.ndarray:
        future = pd.DataFrame({
            "ds": pd.to_datetime(test["timestamp"], utc=True).dt.tz_convert(None)
        })
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self.m.predict(future)
        return clip_nonneg(fc["yhat"].to_numpy())

    def forecast_future(self, periods: int, freq: str = "h") -> pd.DataFrame:
        """Forecast *periods* steps beyond training end, with 80% intervals."""
        future = self.m.make_future_dataframe(periods=periods, freq=freq, include_history=False)
        fc = self.m.predict(future)
        out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        for c in ("yhat", "yhat_lower", "yhat_upper"):
            out[c] = out[c].clip(lower=0)
        return out


class LSTMForecaster(Forecaster):
    name = "lstm"
    family = "advanced"

    def __init__(self, seq_len: int = 48, hidden: int = 32, epochs: int = 8,
                 lr: float = 1e-3, batch: int = 256, max_train_obs: int = 8760) -> None:
        self.seq_len = seq_len
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.batch = batch
        self.max_train_obs = max_train_obs  # trailing window cap (bounds CPU/memory)
        self.net = None
        self.mu = 0.0
        self.sigma = 1.0

    def _build_net(self):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, hidden):
                super().__init__()
                self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return _Net(self.hidden)

    def fit(self, train: pd.DataFrame, ctx: ModelContext) -> "LSTMForecaster":
        import torch

        torch.manual_seed(ctx.seed)
        y = train[ctx.value_col].astype(float).to_numpy()
        if self.max_train_obs and len(y) > self.max_train_obs:
            y = y[-self.max_train_obs:]
        self.mu, self.sigma = float(y.mean()), float(y.std() + 1e-6)
        z = (y - self.mu) / self.sigma
        n = len(z) - self.seq_len
        if n <= 10:  # too little data; degenerate to a constant predictor
            self.net = None
            return self
        X = np.lib.stride_tricks.sliding_window_view(z[:-1], self.seq_len)[:n]
        target = z[self.seq_len:]
        Xt = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        yt = torch.tensor(target, dtype=torch.float32)

        self.net = self._build_net()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        self.net.train()
        idx = np.arange(len(Xt))
        rng = np.random.default_rng(ctx.seed)
        for _ in range(self.epochs):
            rng.shuffle(idx)
            for s in range(0, len(idx), self.batch):
                b = idx[s:s + self.batch]
                opt.zero_grad()
                pred = self.net(Xt[b])
                loss = loss_fn(pred, yt[b])
                loss.backward()
                opt.step()
        self.net.eval()
        return self

    def predict(self, train: pd.DataFrame, test: pd.DataFrame, ctx: ModelContext) -> np.ndarray:
        import torch

        n_test = len(test)
        if self.net is None:
            return clip_nonneg(np.full(n_test, self.mu, dtype=float))
        full = self._full_values(train, test, ctx.value_col)
        z = (full - self.mu) / self.sigma
        start = len(full) - n_test
        windows = [z[i - self.seq_len:i] for i in range(start, len(full))]
        Xt = torch.tensor(np.array(windows), dtype=torch.float32).unsqueeze(-1)
        with torch.no_grad():
            preds = self.net(Xt).numpy()
        return clip_nonneg(preds * self.sigma + self.mu)


class TFTForecaster(Forecaster):
    """Temporal Fusion Transformer (optional — requires pytorch-forecasting)."""

    name = "tft"
    family = "advanced"

    def __init__(self, **kwargs) -> None:
        try:
            import pytorch_forecasting  # noqa: F401
            import lightning  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "TFTForecaster requires the optional 'tft' extra "
                "(pip install -e '.[tft]'). Original error: %s" % exc
            )
        self.kwargs = kwargs
        raise NotImplementedError(
            "TFT is provided as an optional extension point; enable via "
            "modeling.enable_tft and implement training with pytorch-forecasting."
        )

    def fit(self, train, ctx):  # pragma: no cover
        raise NotImplementedError

    def predict(self, train, test, ctx):  # pragma: no cover
        raise NotImplementedError
