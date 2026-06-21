"""Streamlit entry point. Run locally: ``streamlit run dashboard/streamlit_app.py``.

On hosts that only install ``requirements.txt`` (e.g. Streamlit Community Cloud)
the ``src``-layout package isn't pip-installed, so we put ``src`` on the path
before importing it.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from air_raid_forecasting.dashboard.app import render  # noqa: E402

render()
