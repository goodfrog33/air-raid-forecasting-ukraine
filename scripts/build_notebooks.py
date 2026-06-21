"""Generate the project's Jupyter notebooks programmatically with nbformat.

Each notebook is a thin, runnable walk-through that calls the same library code
the pipeline uses — so notebooks never drift from the production modules.

    python scripts/build_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def notebook(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
    return nb


NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "01_data_acquisition.ipynb": [
        ("md", "# 01 · Data Acquisition\n\n"
               "Live-download the historical Ukrainian air raid alert data "
               "(Vadimkin *ukrainian-air-raid-sirens-dataset*, public & token-free, UTC). "
               "No synthetic fallback — the lineage stays honest."),
        ("code", "from air_raid_forecasting.config import load_config\n"
                 "from air_raid_forecasting.data.ingest import ingest, load_raw_alerts, available_raw\n"
                 "cfg = load_config()\n"
                 "ingest(cfg)  # downloads official + volunteer + states.json into data/raw"),
        ("code", "available_raw(cfg)"),
        ("code", "raw = load_raw_alerts(cfg, source='official')\n"
                 "print(raw.shape)\nraw.head()"),
        ("code", "print('Date range:', raw.started_at.min(), '->', raw.started_at.max())\n"
                 "raw.level.value_counts(dropna=False)"),
        ("md", "The raw provenance (URL, bytes, SHA-256, timestamp) is recorded in "
               "`data/raw/_manifest.json`."),
    ],
    "02_cleaning_validation.ipynb": [
        ("md", "# 02 · Cleaning & Validation\n\n"
               "Normalize regions, validate timestamps/durations, and **merge sub-region "
               "alerts up to consolidated oblast intervals** so the series is consistent "
               "across the whole 2022–2026 history. Then derive the required calendar fields."),
        ("code", "from air_raid_forecasting.config import load_config\n"
                 "from air_raid_forecasting.data.clean import clean\n"
                 "cfg = load_config()\n"
                 "events, report = clean(cfg, source='official')\n"
                 "report.as_dict()"),
        ("code", "events[['region','started_at','finished_at','duration_minutes',"
                 "'hour_of_day','day_of_week','season','weekend_flag','is_anomaly_duration']].head()"),
        ("code", "from air_raid_forecasting.data.panel import build_region_hourly, build_national_hourly\n"
                 "region_panel = build_region_hourly(events, cfg)\n"
                 "national = build_national_hourly(region_panel)\n"
                 "national.tail()"),
        ("md", "`alerts_started` (count) and `any_alert` (occupancy) are the modeling signals."),
    ],
    "03_eda.ipynb": [
        ("md", "# 03 · Exploratory Data Analysis\n\n"
               "Temporal, geographic and statistical patterns. Figures are written to "
               "`reports/figures/` and a summary to `reports/eda_summary.json`."),
        ("code", "from air_raid_forecasting.eda import run_eda\n"
                 "summary = run_eda()\n"
                 "summary"),
        ("code", "from IPython.display import Image\nImage('../reports/figures/01_daily_timeseries.png')"),
        ("code", "Image('../reports/figures/06_hour_dow_heatmap.png')"),
        ("code", "Image('../reports/figures/11_decomposition.png')"),
        ("md", "ADF vs KPSS give the classic trend-stationary signature; the daily series "
               "shows strong weekly structure (see STL seasonal component)."),
    ],
    "04_feature_engineering.ipynb": [
        ("md", "# 04 · Feature Engineering\n\n"
               "Calendar/holiday features, lag features, leakage-safe rolling statistics, "
               "`time_since_last_alert`, and the forecasting targets (A: count, B: probability, "
               "C: duration, D: severity)."),
        ("code", "from air_raid_forecasting.pipeline.run_features import main as build_features\n"
                 "meta = build_features([])\n"
                 "meta['severity']"),
        ("code", "import pandas as pd, json\n"
                 "nat = pd.read_parquet('../data/processed/features_national.parquet')\n"
                 "print(nat.shape)\n"
                 "[c for c in nat.columns if 'lag' in c or 'roll' in c][:15]"),
        ("md", "Leakage safety: every feature at time *t* uses only information through *t-1*."),
    ],
    "05_modeling_backtesting.ipynb": [
        ("md", "# 05 · Modeling, Backtesting & Comparison\n\n"
               "Rolling-origin (expanding-window) backtest of the full model zoo on the national "
               "hourly count series, plus per-region classification. Trains the production models "
               "and persists the bundle."),
        ("code", "from air_raid_forecasting.pipeline.run_train import main as train\n"
                 "summary = train([])  # use ['--fast'] for a quick run\n"
                 "summary['count_comparison']['best_model']"),
        ("code", "import pandas as pd\n"
                 "pd.read_csv('../reports/count_national_ranking.csv').round(4)"),
        ("code", "pd.read_csv('../reports/proba_region_ranking.csv').round(4)"),
        ("md", "The best model is selected automatically by mean MAE across folds. "
               "Statistical/Prophet families are honest references; the gradient-boosted "
               "global model backs the production service."),
    ],
    "06_explainability.ipynb": [
        ("md", "# 06 · Explainability\n\n"
               "SHAP, permutation importance and model feature importance for the production "
               "count model."),
        ("code", "import json\n"
                 "json.load(open('../reports/production_count_explainability.json'))"),
        ("code", "from IPython.display import Image\nImage('../reports/figures/production_count_shap_summary.png')"),
        ("code", "import pandas as pd\n"
                 "pd.read_csv('../reports/production_count_permutation_importance.csv').head(15)"),
        ("md", "Recent-activity lags and recency (`time_since_last_alert`) dominate, with "
               "calendar features adding the diurnal/weekly structure."),
    ],
}


def main() -> None:
    for name, cells in NOTEBOOKS.items():
        nb = notebook(cells)
        path = NB_DIR / name
        nbf.write(nb, path)
        print("wrote", path)


if __name__ == "__main__":
    main()
