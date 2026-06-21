"""Air Raid Alert Forecasting — a miniature defense-analytics platform.

End-to-end time-series analysis and forecasting of Ukrainian air raid alerts:
data acquisition, cleaning, EDA, feature engineering, a multi-model forecasting
pipeline with rigorous backtesting, a FastAPI prediction service, and a
Streamlit dashboard.
"""

from __future__ import annotations

__version__ = "1.0.0"

from air_raid_forecasting.config import Config, load_config  # noqa: E402
from air_raid_forecasting.logging_utils import get_logger  # noqa: E402

__all__ = ["__version__", "Config", "load_config", "get_logger"]
