"""Interactive Streamlit dashboard (Phase 12).

Sections:
    * Executive Summary  — headline KPIs + top regions
    * Analytics          — trends, seasonality, regional & duration analysis
    * Forecasting        — future forecast w/ intervals + model comparison
    * Prediction Tool    — live region/horizon forecast from the trained bundle
    * Explainability     — SHAP + feature importance for the production model

Run with::

    streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from air_raid_forecasting.config import load_config
from air_raid_forecasting.data.regions import short_name

CFG = load_config()
PROC = Path(CFG.paths.processed_dir)
REPORTS = Path(CFG.paths.reports_dir)
FIGURES = Path(CFG.paths.figures_dir)
MODELS = Path(CFG.paths.models_dir)
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# --------------------------------------------------------------------------- #
# Cached loaders
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_events() -> pd.DataFrame:
    df = pd.read_parquet(PROC / "alerts_events_labeled.parquet")
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    return df


@st.cache_data(show_spinner=False)
def load_national() -> pd.DataFrame:
    df = pd.read_parquet(PROC / "panel_national_hourly.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data(show_spinner=False)
def load_forecast() -> pd.DataFrame | None:
    p = PROC / "forecast_national.parquet"
    return pd.read_parquet(p) if p.exists() else None


@st.cache_resource(show_spinner=False)
def load_predictor():
    from air_raid_forecasting.service.predictor import Predictor
    try:
        return Predictor.from_dir(MODELS, tz=CFG.project.timezone_local)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def section_summary(events: pd.DataFrame, national: pd.DataFrame) -> None:
    st.header("Executive Summary")
    summary = load_json(REPORTS / "eda_summary.json") or {}
    n_days = max((events["started_at"].max() - events["started_at"].min()).days, 1)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total alerts", f"{len(events):,}")
    c2.metric("Avg alerts / day", f"{len(events)/n_days:.1f}")
    c3.metric("Regions", events["region"].nunique())
    c4.metric("Hours w/ any alert", f"{national['any_alert'].mean()*100:.0f}%")

    st.caption(f"Data span: {events['started_at'].min().date()} → {events['started_at'].max().date()}")

    totals = (events.groupby("region").size().sort_values(ascending=False)
              .rename("alerts").reset_index())
    totals["region"] = totals["region"].map(short_name)
    fig = px.bar(totals.head(12), x="alerts", y="region", orientation="h",
                 title="Top affected regions", color="alerts", color_continuous_scale="reds")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=450)
    st.plotly_chart(fig, use_container_width=True)


def section_analytics(events: pd.DataFrame, national: pd.DataFrame) -> None:
    st.header("Analytics")
    tab_trend, tab_season, tab_region, tab_dur = st.tabs(
        ["Trends", "Seasonality", "Regional", "Duration"])

    with tab_trend:
        daily = national.set_index("timestamp")["alerts_started"].resample("1D").sum()
        roll = daily.rolling(7, min_periods=1).mean()
        fig = go.Figure()
        fig.add_scatter(x=daily.index, y=daily.values, name="Daily", line=dict(width=1, color="#9ecae1"))
        fig.add_scatter(x=roll.index, y=roll.values, name="7-day mean", line=dict(width=2.5, color="#c0392b"))
        fig.update_layout(title="Daily national alerts", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab_season:
        pivot = (events.groupby(["day_of_week", "hour_of_day"]).size()
                 .unstack(fill_value=0).reindex(range(7)))
        fig = px.imshow(pivot.values, labels=dict(x="Hour", y="Day", color="Alerts"),
                        x=list(range(24)), y=DOW, color_continuous_scale="rocket_r",
                        title="Alert intensity: hour × day of week", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        hourly = events.groupby("hour_of_day").size()
        col1.plotly_chart(px.bar(x=hourly.index, y=hourly.values,
                          labels={"x": "Hour", "y": "Alerts"}, title="By hour of day"),
                          use_container_width=True)
        seasonal = events.groupby("season").size().reindex(
            ["Winter", "Spring", "Summer", "Autumn"]).fillna(0)
        col2.plotly_chart(px.bar(x=seasonal.index, y=seasonal.values,
                          labels={"x": "", "y": "Alerts"}, title="By season"),
                          use_container_width=True)

    with tab_region:
        agg = (events.groupby("region")
               .agg(alerts=("region", "size"), median_duration=("duration_minutes", "median"))
               .sort_values("alerts", ascending=False).reset_index())
        agg["region"] = agg["region"].map(short_name)
        fig = px.scatter(agg, x="alerts", y="median_duration", text="region", size="alerts",
                         title="Region: frequency vs typical duration",
                         labels={"alerts": "Total alerts", "median_duration": "Median duration (min)"})
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

    with tab_dur:
        dur = events["duration_minutes"].clip(upper=events["duration_minutes"].quantile(0.99))
        fig = px.histogram(dur, nbins=80, title="Alert duration distribution (clipped at p99)",
                           labels={"value": "Duration (min)"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(events.groupby("severity")["duration_minutes"]
                     .describe()[["count", "mean", "50%", "max"]].round(1))


def section_forecasting() -> None:
    st.header("Forecasting")
    fc = load_forecast()
    if fc is not None:
        fc["ds"] = pd.to_datetime(fc["ds"])
        fig = go.Figure()
        fig.add_scatter(x=fc["ds"], y=fc["yhat_upper"], line=dict(width=0), showlegend=False)
        fig.add_scatter(x=fc["ds"], y=fc["yhat_lower"], fill="tonexty", line=dict(width=0),
                        name="80% interval", fillcolor="rgba(192,57,43,0.2)")
        fig.add_scatter(x=fc["ds"], y=fc["yhat"], line=dict(color="#c0392b", width=2.5), name="Forecast")
        fig.update_layout(title="72-hour national alert-count forecast (Prophet)", height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No forecast yet — run the training pipeline to generate it.")

    st.subheader("Model comparison (rigorous backtesting)")
    cnt = load_json(REPORTS / "count_national_ranking.json")
    if cnt:
        st.markdown(f"**Count (national, 1-step):** best = `{cnt['best_model']}` by {cnt['primary_metric']}")
        st.dataframe(pd.DataFrame(cnt["ranking"]).round(4))
    prob = load_json(REPORTS / "proba_region_ranking.json")
    if prob:
        st.markdown(f"**Classification (per-region, 6h):** best = `{prob['best_model']}` by {prob['primary_metric']}")
        st.dataframe(pd.DataFrame(prob["ranking"]).round(4))
    if not cnt and not prob:
        st.info("No comparison tables yet — run the training pipeline.")


def section_prediction() -> None:
    st.header("Prediction Tool")
    predictor = load_predictor()
    if predictor is None:
        st.warning("Model bundle not found. Train the models first "
                   "(`python -m air_raid_forecasting.pipeline.run_train`).")
        return
    col1, col2 = st.columns(2)
    region = col1.selectbox("Region", predictor.regions,
                            format_func=short_name)
    horizon = col2.select_slider("Forecast horizon (hours)", options=[1, 3, 6, 12, 24], value=6)
    if st.button("Forecast", type="primary"):
        res = predictor.predict_one(region, int(horizon))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Alert probability", f"{res['alert_probability']*100:.0f}%")
        m2.metric("Predicted count", res["predicted_alert_count"])
        m3.metric("Expected duration", f"{res['predicted_duration_minutes']:.0f} min")
        m4.metric("Severity", res["severity"])
        st.progress(res["confidence"], text=f"Model confidence: {res['confidence']*100:.0f}%")
        st.json(res)


def section_explain() -> None:
    st.header("Explainability")
    shap_fig = FIGURES / "production_count_shap_summary.png"
    payload = load_json(REPORTS / "production_count_explainability.json")
    if shap_fig.exists():
        st.subheader("SHAP — mean feature contribution")
        st.image(str(shap_fig), use_container_width=True)
    if payload:
        fi = payload.get("shap_top") or payload.get("feature_importance_top") or {}
        if fi:
            s = pd.Series(fi).sort_values(ascending=True).tail(15)
            st.plotly_chart(px.bar(x=s.values, y=s.index, orientation="h",
                            title="Top features", labels={"x": "Importance", "y": ""}),
                            use_container_width=True)
        perm = payload.get("permutation_importance_top")
        if perm:
            st.subheader("Permutation importance")
            st.dataframe(pd.Series(perm).sort_values(ascending=False).round(5))
    if not shap_fig.exists() and not payload:
        st.info("No explainability artifacts yet — run the training pipeline.")


def render() -> None:
    st.set_page_config(page_title="Ukraine Air Raid Forecasting", page_icon="🛡️", layout="wide")
    st.title("🛡️ Ukraine Air Raid Alert — Analytics & Forecasting")
    st.caption("A miniature defense-analytics platform · data: Vadimkin air-raid-sirens dataset (UTC)")

    try:
        events, national = load_events(), load_national()
    except FileNotFoundError:
        st.error("Processed data not found. Run the pipeline: "
                 "`python -m air_raid_forecasting.pipeline.run_all`")
        return

    page = st.sidebar.radio(
        "Section",
        ["Executive Summary", "Analytics", "Forecasting", "Prediction Tool", "Explainability"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Events: {len(events):,}  ·  Regions: {events['region'].nunique()}")

    if page == "Executive Summary":
        section_summary(events, national)
    elif page == "Analytics":
        section_analytics(events, national)
    elif page == "Forecasting":
        section_forecasting()
    elif page == "Prediction Tool":
        section_prediction()
    elif page == "Explainability":
        section_explain()
