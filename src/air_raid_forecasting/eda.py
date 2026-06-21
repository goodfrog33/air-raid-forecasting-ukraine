"""Phase 3 — Exploratory Data Analysis.

Generates publication-quality figures (``reports/figures``) and a machine-
readable statistics summary (``reports/eda_summary.json``) covering temporal,
geographic and statistical patterns of Ukrainian air raid alerts.

Run with::

    python -m air_raid_forecasting.pipeline.run_eda
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / reproducible figure generation
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from air_raid_forecasting.config import Config, load_config  # noqa: E402
from air_raid_forecasting.data.regions import short_name  # noqa: E402
from air_raid_forecasting.logging_utils import get_logger  # noqa: E402

log = get_logger(__name__)

_DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 120, "figure.autolayout": True})


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("  figure -> %s", path.name)


# --------------------------------------------------------------------------- #
# Temporal patterns
# --------------------------------------------------------------------------- #
def plot_daily_timeseries(events: pd.DataFrame, figdir: Path) -> None:
    daily = events.set_index("started_at").resample("1D").size().rename("alerts")
    daily.index = daily.index.tz_convert(None)
    roll = daily.rolling(7, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(daily.index, daily.values, lw=0.6, alpha=0.4, color="#4C72B0", label="Daily")
    ax.plot(roll.index, roll.values, lw=2.2, color="#C44E52", label="7-day mean")
    ax.set(title="Daily air raid alerts (national, oblast-level events)",
           xlabel="Date", ylabel="Alerts per day")
    ax.legend()
    _save(fig, figdir / "01_daily_timeseries.png")


def plot_hourly_profile(events: pd.DataFrame, figdir: Path) -> None:
    prof = events.groupby("hour_of_day").size()
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x=prof.index, y=prof.values, color="#4C72B0", ax=ax)
    ax.set(title="Alerts by hour of day (local, Europe/Kyiv)",
           xlabel="Hour of day", ylabel="Total alerts")
    _save(fig, figdir / "02_hourly_profile.png")


def plot_dow_profile(events: pd.DataFrame, figdir: Path) -> None:
    prof = events.groupby("day_of_week").size().reindex(range(7), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=_DOW, y=prof.values, color="#55A868", ax=ax)
    ax.set(title="Alerts by day of week", xlabel="", ylabel="Total alerts")
    _save(fig, figdir / "03_day_of_week.png")


def plot_monthly_trend(events: pd.DataFrame, figdir: Path) -> None:
    monthly = events.set_index("started_at").resample("1MS").size()
    monthly.index = monthly.index.tz_convert(None)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(monthly.index, monthly.values, width=20, color="#8172B3")
    ax.set(title="Monthly alert totals", xlabel="Month", ylabel="Alerts per month")
    _save(fig, figdir / "04_monthly_trend.png")


def plot_seasonal(events: pd.DataFrame, figdir: Path) -> None:
    prof = events.groupby("season").size().reindex(_SEASON_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=prof.index, y=prof.values, palette="mako", hue=prof.index, legend=False, ax=ax)
    ax.set(title="Alerts by season", xlabel="", ylabel="Total alerts")
    _save(fig, figdir / "05_seasonal.png")


def plot_hour_dow_heatmap(events: pd.DataFrame, figdir: Path) -> None:
    pivot = (
        events.groupby(["day_of_week", "hour_of_day"]).size()
        .unstack(fill_value=0).reindex(range(7))
    )
    pivot.index = _DOW
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot, cmap="rocket_r", ax=ax, cbar_kws={"label": "Alerts"})
    ax.set(title="Alert intensity: hour of day x day of week",
           xlabel="Hour of day", ylabel="")
    _save(fig, figdir / "06_hour_dow_heatmap.png")


# --------------------------------------------------------------------------- #
# Geographic patterns
# --------------------------------------------------------------------------- #
def plot_region_totals(events: pd.DataFrame, figdir: Path) -> None:
    totals = events.groupby("region").size().sort_values(ascending=False)
    labels = [short_name(r) for r in totals.index]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(y=labels, x=totals.values, palette="flare", hue=labels, legend=False, ax=ax)
    ax.set(title="Total alerts by region", xlabel="Total alerts", ylabel="")
    _save(fig, figdir / "07_region_totals.png")


def plot_region_duration(events: pd.DataFrame, figdir: Path) -> None:
    dur = (
        events.groupby("region")["duration_minutes"].median()
        .sort_values(ascending=False)
    )
    labels = [short_name(r) for r in dur.index]
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(y=labels, x=dur.values, palette="crest", hue=labels, legend=False, ax=ax)
    ax.set(title="Median alert duration by region", xlabel="Median duration (min)", ylabel="")
    _save(fig, figdir / "08_region_duration.png")


def plot_region_month_heatmap(events: pd.DataFrame, figdir: Path) -> None:
    tmp = events.copy()
    tmp["ym"] = tmp["started_at"].dt.tz_convert(None).dt.to_period("M").astype(str)
    pivot = tmp.groupby(["region", "ym"]).size().unstack(fill_value=0)
    pivot.index = [short_name(r) for r in pivot.index]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(16, 9))
    sns.heatmap(pivot, cmap="magma_r", ax=ax, cbar_kws={"label": "Alerts/month"})
    ax.set(title="Regional alert intensity over time", xlabel="Month", ylabel="")
    ax.set_xticks(ax.get_xticks()[::3])
    _save(fig, figdir / "09_region_month_heatmap.png")


# --------------------------------------------------------------------------- #
# Statistical patterns
# --------------------------------------------------------------------------- #
def plot_duration_distribution(events: pd.DataFrame, figdir: Path) -> None:
    dur = events["duration_minutes"].clip(lower=0.1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(dur, bins=80, color="#4C72B0", ax=axes[0])
    axes[0].set(title="Duration distribution", xlabel="Minutes", ylabel="Count")
    axes[0].set_xlim(0, dur.quantile(0.99))
    sns.histplot(np.log10(dur), bins=60, color="#C44E52", ax=axes[1])
    axes[1].set(title="Duration distribution (log10)", xlabel="log10(minutes)", ylabel="Count")
    _save(fig, figdir / "10_duration_distribution.png")


def plot_decomposition(national: pd.DataFrame, figdir: Path) -> dict:
    """STL decomposition of the daily national series (weekly seasonality)."""
    from statsmodels.tsa.seasonal import STL

    daily = (
        national.set_index("timestamp")["alerts_started"].resample("1D").sum()
    )
    daily.index = daily.index.tz_convert(None)
    stl = STL(daily, period=7, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    axes[0].plot(daily.index, daily.values, color="#333"); axes[0].set_ylabel("Observed")
    axes[1].plot(daily.index, stl.trend, color="#C44E52"); axes[1].set_ylabel("Trend")
    axes[2].plot(daily.index, stl.seasonal, color="#55A868"); axes[2].set_ylabel("Seasonal(7)")
    axes[3].plot(daily.index, stl.resid, color="#8172B3", lw=0.6); axes[3].set_ylabel("Resid")
    axes[0].set_title("STL decomposition of daily national alerts")
    _save(fig, figdir / "11_decomposition.png")
    strength_seasonal = max(0.0, 1 - stl.resid.var() / (stl.seasonal + stl.resid).var())
    strength_trend = max(0.0, 1 - stl.resid.var() / (stl.trend + stl.resid).var())
    return {"seasonal_strength": float(strength_seasonal), "trend_strength": float(strength_trend)}


def plot_acf(national: pd.DataFrame, figdir: Path) -> None:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    series = national.set_index("timestamp")["alerts_started"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(series, lags=72, ax=axes[0])
    axes[0].set_title("ACF (hourly national alerts, 72 lags)")
    plot_pacf(series, lags=72, ax=axes[1], method="ywm")
    axes[1].set_title("PACF (hourly national alerts, 72 lags)")
    _save(fig, figdir / "12_acf_pacf.png")


def stationarity_tests(national: pd.DataFrame) -> dict:
    """ADF (H0: unit root) and KPSS (H0: stationary) on the hourly series."""
    from statsmodels.tsa.stattools import adfuller, kpss

    series = national["alerts_started"].astype(float)
    out: dict = {}
    try:
        adf = adfuller(series, autolag="AIC")
        out["adf"] = {"statistic": float(adf[0]), "pvalue": float(adf[1]),
                      "stationary_at_5pct": bool(adf[1] < 0.05)}
    except Exception as exc:  # pragma: no cover
        out["adf"] = {"error": str(exc)}
    try:
        kp = kpss(series, regression="c", nlags="auto")
        out["kpss"] = {"statistic": float(kp[0]), "pvalue": float(kp[1]),
                       "stationary_at_5pct": bool(kp[1] > 0.05)}
    except Exception as exc:  # pragma: no cover
        out["kpss"] = {"error": str(exc)}
    return out


def summarize(events: pd.DataFrame, national: pd.DataFrame, extra: dict) -> dict:
    dur = events["duration_minutes"]
    region_counts = events.groupby("region").size().sort_values(ascending=False)
    n_days = (events["started_at"].max() - events["started_at"].min()).days or 1
    return {
        "n_events": int(len(events)),
        "n_regions": int(events["region"].nunique()),
        "date_min": str(events["started_at"].min()),
        "date_max": str(events["started_at"].max()),
        "total_alerts": int(len(events)),
        "avg_alerts_per_day": round(len(events) / n_days, 2),
        "duration_minutes": {
            "mean": round(float(dur.mean()), 2),
            "median": round(float(dur.median()), 2),
            "p90": round(float(dur.quantile(0.90)), 2),
            "p99": round(float(dur.quantile(0.99)), 2),
            "max": round(float(dur.max()), 2),
        },
        "top_regions": {short_name(r): int(c) for r, c in region_counts.head(10).items()},
        "hourly_alert_rate": round(float((national["any_alert"]).mean()), 4),
        **extra,
    }


def run_eda(cfg: Config | None = None) -> dict:
    """Generate all EDA figures + the summary JSON. Returns the summary dict."""
    cfg = cfg or load_config()
    _setup_style()
    proc = Path(cfg.paths.processed_dir)
    figdir = Path(cfg.paths.figures_dir)
    figdir.mkdir(parents=True, exist_ok=True)

    events = pd.read_parquet(proc / "alerts_events.parquet")
    events["started_at"] = pd.to_datetime(events["started_at"], utc=True)
    national = pd.read_parquet(proc / "panel_national_hourly.parquet")
    national["timestamp"] = pd.to_datetime(national["timestamp"], utc=True)

    log.info("Generating EDA figures ...")
    plot_daily_timeseries(events, figdir)
    plot_hourly_profile(events, figdir)
    plot_dow_profile(events, figdir)
    plot_monthly_trend(events, figdir)
    plot_seasonal(events, figdir)
    plot_hour_dow_heatmap(events, figdir)
    plot_region_totals(events, figdir)
    plot_region_duration(events, figdir)
    plot_region_month_heatmap(events, figdir)
    plot_duration_distribution(events, figdir)
    decomp = plot_decomposition(national, figdir)
    plot_acf(national, figdir)

    extra = {"decomposition": decomp, "stationarity": stationarity_tests(national)}
    summary = summarize(events, national, extra)

    out_path = Path(cfg.paths.reports_dir) / "eda_summary.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log.info("EDA summary -> %s", out_path)
    return summary
