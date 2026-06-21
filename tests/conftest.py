"""Shared pytest fixtures.

Tests run on small synthetic data so they are fast and do not require the live
download or a trained model bundle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from air_raid_forecasting.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def raw_alerts() -> pd.DataFrame:
    """A tiny raw alert frame in the unified RAW_COLUMNS schema."""
    base = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")
    rows = []
    rng = np.random.default_rng(0)
    for day in range(30):
        for region in ["Kyivska oblast", "Lvivska oblast", "Kharkivska oblast"]:
            n = rng.integers(0, 3)
            for _ in range(int(n)):
                start = base + pd.Timedelta(days=day, hours=int(rng.integers(0, 24)))
                dur = int(rng.integers(20, 180))
                rows.append({
                    "region_raw": region, "raion": None, "hromada": None, "level": "oblast",
                    "started_at": start, "finished_at": start + pd.Timedelta(minutes=dur),
                    "naive": False, "source_dataset": "official",
                })
    # Add an overlapping pair (to exercise interval merging) and a bad row.
    s = base + pd.Timedelta(days=1, hours=5)
    rows.append({"region_raw": "Kyivska oblast", "raion": "X", "hromada": None, "level": "raion",
                 "started_at": s, "finished_at": s + pd.Timedelta(minutes=60),
                 "naive": False, "source_dataset": "official"})
    rows.append({"region_raw": "Kyivska oblast", "raion": "Y", "hromada": None, "level": "raion",
                 "started_at": s + pd.Timedelta(minutes=30), "finished_at": s + pd.Timedelta(minutes=120),
                 "naive": False, "source_dataset": "official"})
    # Negative-duration row with a VALID region (exercises the duration filter).
    rows.append({"region_raw": "Lvivska oblast", "raion": None, "hromada": None, "level": "oblast",
                 "started_at": s, "finished_at": s - pd.Timedelta(minutes=5),
                 "naive": False, "source_dataset": "official"})
    # Unmapped region with a valid duration (exercises the region filter).
    rows.append({"region_raw": "Nowhere oblast", "raion": None, "hromada": None, "level": "oblast",
                 "started_at": s, "finished_at": s + pd.Timedelta(minutes=20),
                 "naive": False, "source_dataset": "official"})
    return pd.DataFrame(rows)
