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
from air_raid_forecasting.data.geo import load_geojson, region_centroids
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
def load_region_panel() -> pd.DataFrame:
    df = pd.read_parquet(PROC / "panel_region_hourly.parquet")
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


@st.cache_data(show_spinner=False)
def load_geo():
    """Return (geojson, centroids); (None, None) if unavailable."""
    try:
        geo = load_geojson(CFG.paths.external_dir, download=True)
        return geo, region_centroids(geo)
    except Exception:
        return None, None


@st.cache_data(show_spinner="Running the model for every region…")
def live_predictions(horizon: int, model: str = "best", factor: str = "base") -> pd.DataFrame:
    predictor = load_predictor()
    rows = predictor.predict_batch([(r, horizon) for r in predictor.regions],
                                   model=model, factor=factor)
    df = pd.DataFrame(rows)
    df["short"] = df["region"].map(short_name)
    return df


def _model_label(name: str) -> str:
    return {"best": "Best (auto)"}.get(name, name)


_FACTOR_LABELS = {
    "base": "Off (no event factor)",
    "news": "📰 GDELT war-news",
    "telegram": "📡 Telegram channels (war_monitor, radar_raketaa)",
}


def _factor_label(v: str) -> str:
    return _FACTOR_LABELS.get(v, v)


def _factor_picker(predictor, key: str) -> str:
    """Radio for the event-signal factor (Off / GDELT / Telegram); shows its lift."""
    opts = [v for v in ("base", "news", "telegram") if v in predictor.factors]
    if len(opts) <= 1:
        return "base"
    choice = st.radio("Event-signal factor", opts, format_func=_factor_label,
                      horizontal=True, key=key)
    lift = predictor.factor_lift(choice)
    if lift and lift.get("pct_improvement") is not None:
        st.caption(f"{_factor_label(choice)} → backtest count MAE {lift['variant_mae']:.3f} "
                   f"vs base {lift['base_mae']:.3f} ({lift['pct_improvement']:+.1f}%).")
    return choice


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _fmt(x, nd=2, dash="—"):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return dash


def section_summary(events: pd.DataFrame, series: pd.DataFrame, scope: str,
                    events_all: pd.DataFrame) -> None:
    eda = load_json(REPORTS / "eda_summary.json") or {}
    count = load_json(REPORTS / "count_national_ranking.json") or {}
    proba = load_json(REPORTS / "proba_region_ranking.json") or {}
    metrics = load_json(REPORTS / "metrics_summary.json") or {}
    explain = load_json(REPORTS / "production_count_explainability.json") or {}

    crank = {r["model"]: r for r in count.get("ranking", [])}
    prank = {r["model"]: r for r in proba.get("ranking", [])}
    best_count = count.get("best_model")
    best_proba = proba.get("best_model")
    n_days = max((events_all["started_at"].max() - events_all["started_at"].min()).days, 1)

    st.header("Executive Summary")
    st.markdown(
        "An end-to-end analytics platform that forecasts **Ukrainian air raid alerts** "
        "from real historical data. It cleans and consolidates alerts to oblast level, "
        "explores their temporal and geographic structure, and benchmarks a dozen "
        "forecasting models with leakage-free **expanding-window backtesting** to power a "
        "live per-region prediction service.")

    # --- headline KPIs (project-wide) ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total alerts", f"{eda.get('total_alerts', len(events_all)):,}")
    c2.metric("Avg / day", _fmt(eda.get("avg_alerts_per_day", len(events_all) / n_days), 1))
    c3.metric("Regions", eda.get("n_regions", events_all["region"].nunique()))
    c4.metric("Hours w/ alert", f"{float(eda.get('hourly_alert_rate', 0))*100:.0f}%")
    c5.metric("Median duration", f"{_fmt(eda.get('duration_minutes', {}).get('median'), 0)} min")
    st.caption(f"Data: Vadimkin air-raid-sirens dataset (live, UTC) · "
               f"{str(eda.get('date_min', ''))[:10]} → {str(eda.get('date_max', ''))[:10]} · "
               "_region scope (sidebar) applies to the Analytics tab._")

    st.divider()
    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("What the data shows")
        dur = eda.get("duration_minutes", {})
        decomp = eda.get("decomposition", {})
        adf = eda.get("stationarity", {}).get("adf", {})
        top = list(eda.get("top_regions", {}).items())
        st.markdown(
            f"- **Where:** most-affected regions are "
            f"{', '.join(k for k, _ in top[:5]) if top else 'frontline oblasts'} — the eastern/"
            "southern frontline dominates.\n"
            f"- **How often:** at least one region is under alert **{float(eda.get('hourly_alert_rate', 0))*100:.0f}%** "
            "of all hours; binary risk is therefore modelled **per region**, not nationally.\n"
            f"- **How long:** a typical alert lasts **{_fmt(dur.get('median'), 0)} min** "
            f"(90th pct ≈ {_fmt(dur.get('p90'), 0)} min, tail to {_fmt(dur.get('max'), 0)} min).\n"
            f"- **Rhythm:** clear daily + weekly seasonality (STL seasonal strength "
            f"{_fmt(decomp.get('seasonal_strength'), 2)}, trend {_fmt(decomp.get('trend_strength'), 2)}).\n"
            f"- **Stationarity:** ADF p={_fmt(adf.get('pvalue'), 3)} → the hourly series is "
            "trend-stationary, justifying lag/rolling features over raw levels.")
    with right:
        totals = (events_all.groupby("region").size().sort_values(ascending=False)
                  .rename("alerts").reset_index())
        totals["region"] = totals["region"].map(short_name)
        fig = px.bar(totals.head(10), x="alerts", y="region", orientation="h",
                     title="Top affected regions", color="alerts", color_continuous_scale="reds")
        fig.update_layout(yaxis=dict(autorange="reversed"), height=360,
                          margin=dict(l=0, r=0, t=40, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Forecasting models & accuracy")
    st.markdown(
        "Twelve models (baselines · ETS/SARIMA · Prophet · LSTM · RF/XGBoost/LightGBM/"
        "CatBoost) were compared on identical rolling-origin folds. Headline results:")
    rows = []
    if best_count:
        rows.append({"Task": "Alert count — national, next hour", "Type": "regression",
                     "Best model": best_count, "Score": f"MAE {_fmt(crank.get(best_count, {}).get('MAE'))}",
                     "vs baseline": f"naive MAE {_fmt(crank.get('naive', {}).get('MAE'))}"})
    if best_proba:
        rows.append({"Task": "Alert probability — per region, next 6 h", "Type": "classification",
                     "Best model": best_proba, "Score": f"ROC-AUC {_fmt(prank.get(best_proba, {}).get('ROC_AUC'))}",
                     "vs baseline": f"persistence {_fmt(prank.get('persistence', {}).get('ROC_AUC'))}"})
    pm = (metrics.get("production_count_backtest") or {}).get("per_model_MAE") or {}
    if pm:
        bm = min(pm, key=pm.get)
        rows.append({"Task": "Production count — per region, next hour", "Type": "regression",
                     "Best model": bm, "Score": f"MAE {_fmt(pm[bm], 3)}", "vs baseline": "—"})
    sev = metrics.get("severity_metrics") or {}
    if sev:
        rows.append({"Task": "Severity — Low/Med/High/Critical", "Type": "classification",
                     "Best model": "lightgbm", "Score": f"acc {_fmt(sev.get('accuracy'))} · F1 {_fmt(sev.get('f1_macro'))}",
                     "vs baseline": "—"})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    m1, m2, m3 = st.columns(3)
    if best_count:
        nv = crank.get("naive", {}).get("MAE")
        bv = crank.get(best_count, {}).get("MAE")
        delta = f"-{_fmt(nv - bv)} vs naive" if (nv and bv) else None
        m1.metric("Best count MAE (national)", _fmt(bv), delta, delta_color="inverse")
    if best_proba:
        m2.metric("Best alert-probability AUC", _fmt(prank.get(best_proba, {}).get("ROC_AUC")))
    lift = metrics.get("news_lift") or {}
    if lift.get("pct_improvement") is not None:
        m3.metric("News factor (count MAE)", f"{lift['pct_improvement']:+.1f}%",
                  help="GDELT war-news intensity; negative = it slightly hurt this target.")

    drivers = explain.get("shap_top") or explain.get("feature_importance_top") or {}
    if drivers:
        st.subheader("What drives the forecasts")
        top_d = list(drivers.items())[:6]
        st.markdown("Top features of the production model: "
                    + ", ".join(f"`{k}`" for k, _ in top_d) + ".")
    st.caption("⚠️ Analytical use on open historical data only — not an early-warning system.")


def section_analytics(events: pd.DataFrame, series: pd.DataFrame, events_all: pd.DataFrame) -> None:
    st.header("Analytics")
    tab_trend, tab_season, tab_region, tab_dur = st.tabs(
        ["Trends", "Seasonality", "Regional", "Duration"])

    with tab_trend:
        daily = series.set_index("timestamp")["alerts_started"].resample("1D").sum()
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
                        x=list(range(24)), y=DOW, color_continuous_scale="Inferno",
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
        st.caption("Cross-region comparison (always all regions).")
        agg = (events_all.groupby("region")
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
    col1, col2, col3 = st.columns(3)
    region = col1.selectbox("Region", predictor.regions, format_func=short_name)
    horizon = col2.select_slider("Forecast horizon (hours)", options=[1, 3, 6, 12, 24], value=6)
    model = col3.selectbox("Prediction model", ["best", *predictor.models], format_func=_model_label,
                           help="'Best (auto)' = lowest backtest MAE for the chosen factor set.")
    factor = _factor_picker(predictor, key="pred_factor")
    if st.button("Forecast", type="primary"):
        res = predictor.predict_one(region, int(horizon), model=model, factor=factor)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Alert probability", f"{res['alert_probability']*100:.0f}%")
        m2.metric("Predicted count", res["predicted_alert_count"])
        m3.metric("Expected duration", f"{res['predicted_duration_minutes']:.0f} min")
        m4.metric("Severity", res["severity"])
        st.progress(res["confidence"], text=f"Model confidence: {res['confidence']*100:.0f}%")
        st.caption(f"Model: **{res.get('model')}** · factor: **{_factor_label(res.get('factor'))}** · "
                   f"matched horizon: {res.get('matched_horizon_hours')}h")
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


_SEVERITY_COLORS = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e67e22", "Critical": "#e74c3c"}
_SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]


def section_map() -> None:
    st.header("🗺️ Live Prediction Map")
    predictor = load_predictor()
    if predictor is None:
        st.warning("Model bundle not found. Train the models first "
                   "(`python -m air_raid_forecasting.pipeline.run_train`).")
        return

    c1, c2, c3 = st.columns([1, 1, 1])
    horizon = c1.select_slider("Forecast horizon (hours)", options=[1, 3, 6, 12, 24], value=6)
    metric = c2.selectbox("Colour regions by",
                          ["Alert probability", "Predicted count", "Severity"])
    model = c3.selectbox("Prediction model", ["best", *predictor.models],
                         format_func=_model_label,
                         help="'Best (auto)' picks the model with the lowest backtest MAE.")
    factor = _factor_picker(predictor, key="map_factor")
    df = live_predictions(int(horizon), model, factor)
    geojson, centroids = load_geo()

    hover = {
        "region": False, "alert_probability": ":.2f", "predicted_alert_count": True,
        "predicted_duration_minutes": ":.0f", "severity": True,
    }
    if geojson is not None:
        # maplibre renderer (choropleth_map) fills polygons in screen space, so it
        # is immune to the d3-geo winding bug that made one oblast flood the frame.
        common = dict(geojson=geojson, locations="region",
                      featureidkey="properties.region_canonical",
                      hover_name="short", hover_data=hover,
                      map_style="carto-positron", center={"lat": 48.4, "lon": 31.4},
                      zoom=4.3, opacity=0.72)
        if metric == "Severity":
            fig = px.choropleth_map(df, color="severity", color_discrete_map=_SEVERITY_COLORS,
                                    category_orders={"severity": _SEVERITY_ORDER}, **common)
        elif metric == "Predicted count":
            fig = px.choropleth_map(df, color="predicted_alert_count",
                                    color_continuous_scale="Oranges", **common)
        else:
            fig = px.choropleth_map(df, color="alert_probability", range_color=(0, 1),
                                    color_continuous_scale="Reds", **common)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=600)
    else:
        # Bubble-map fallback (no GeoJSON): centroids + colour/size by prediction.
        d = df.copy()
        d["lon"] = d["region"].map(lambda r: (centroids or {}).get(r, (31.0, 49.0))[0])
        d["lat"] = d["region"].map(lambda r: (centroids or {}).get(r, (31.0, 49.0))[1])
        fig = px.scatter_geo(d, lat="lat", lon="lon", color="alert_probability",
                             size="predicted_alert_count", hover_name="short",
                             color_continuous_scale="Reds", range_color=(0, 1))
        fig.update_geos(scope="europe", center=dict(lat=48.5, lon=31.5),
                        projection_scale=4, visible=True)
        fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))

    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Live model output for all {len(df)} regions · horizon = next {horizon}h · "
               f"as of {df['as_of'].iloc[0][:16].replace('T', ' ')} UTC")

    sort_col = {"Alert probability": "alert_probability", "Predicted count": "predicted_alert_count",
                "Severity": "alert_probability"}[metric]
    table = (df[["short", "alert_probability", "predicted_alert_count",
                 "predicted_duration_minutes", "severity", "confidence"]]
             .sort_values(sort_col, ascending=False)
             .rename(columns={"short": "Region", "alert_probability": "P(alert)",
                              "predicted_alert_count": "Count", "severity": "Severity",
                              "predicted_duration_minutes": "Duration (min)",
                              "confidence": "Confidence"}))
    st.dataframe(table.reset_index(drop=True), use_container_width=True, height=380)


def _artifacts_signature() -> tuple:
    """Mtimes of the key artifacts; changes when a new build/train is deployed."""
    files = [MODELS / "model_bundle.joblib", PROC / "alerts_events_labeled.parquet",
             PROC / "forecast_national.parquet", REPORTS / "metrics_summary.json",
             REPORTS / "count_national_ranking.json"]
    return tuple((f.name, f.stat().st_mtime) for f in files if f.exists())


def render() -> None:
    st.set_page_config(page_title="Ukraine Air Raid Forecasting", page_icon="🛡️", layout="wide")
    # Auto-clear caches when the underlying artifacts change (e.g. after a redeploy),
    # so a new model bundle / refreshed reports show up without a manual reboot.
    sig = _artifacts_signature()
    if st.session_state.get("_artifact_sig") != sig:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state["_artifact_sig"] = sig
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
        ["Executive Summary", "Live Map", "Analytics", "Forecasting",
         "Prediction Tool", "Explainability"],
    )
    st.sidebar.markdown("---")

    # Global region-scope filter (drives Summary & Analytics).
    regions = sorted(events["region"].unique())
    label_to_region = {short_name(r): r for r in regions}
    scope = st.sidebar.selectbox(
        "Region scope", ["All Ukraine", *sorted(label_to_region)],
        help="Filter the Summary & Analytics views to one region, or view all of Ukraine.",
    )
    region_sel = label_to_region.get(scope)
    if region_sel:
        events_scoped = events[events["region"] == region_sel]
        rp = load_region_panel()
        series = rp[rp["region"] == region_sel][["timestamp", "alerts_started", "any_alert"]].copy()
    else:
        events_scoped = events
        series = national[["timestamp", "alerts_started", "any_alert"]].copy()
    st.sidebar.caption(f"Scope: {scope} · Events: {len(events_scoped):,}")

    if page == "Executive Summary":
        section_summary(events_scoped, series, scope, events)
    elif page == "Live Map":
        section_map()
    elif page == "Analytics":
        section_analytics(events_scoped, series, events)
    elif page == "Forecasting":
        section_forecasting()
    elif page == "Prediction Tool":
        section_prediction()
    elif page == "Explainability":
        section_explain()
