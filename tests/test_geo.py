"""Tests for the map/geo helpers (region<->polygon join + centroids)."""

from __future__ import annotations

from pathlib import Path

from air_raid_forecasting.config import load_config
from air_raid_forecasting.data.geo import (
    SHAPENAME_TO_CANONICAL,
    load_geojson,
    region_centroids,
)
from air_raid_forecasting.data.regions import CANONICAL_OBLASTS


def test_shapename_map_covers_all_canonical_regions():
    # Every canonical region must have a polygon name to key the choropleth on.
    assert set(CANONICAL_OBLASTS) <= set(SHAPENAME_TO_CANONICAL.values())


def test_geojson_join_and_centroids():
    cfg = load_config()
    path = Path(cfg.paths.external_dir) / "ukraine_oblasts.geojson"
    if not path.exists():
        return  # geojson not downloaded in this environment — skip silently
    geo = load_geojson(cfg.paths.external_dir, download=False)
    feats = geo["features"]
    mapped = [f["properties"].get("region_canonical") for f in feats]
    assert all(m is not None for m in mapped)  # every feature joined

    cents = region_centroids(geo)
    assert len(cents) >= 24
    for lon, lat in cents.values():  # all centroids inside Ukraine's bounding box
        assert 21.0 < lon < 41.0
        assert 43.0 < lat < 53.0
