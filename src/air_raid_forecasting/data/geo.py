"""Geographic helpers for the map dashboard.

Loads (and caches) a simplified Ukraine oblast GeoJSON from geoBoundaries and
maps its English ``shapeName`` field to this project's canonical region names,
so a choropleth can be keyed directly on our predictions. Also computes polygon
centroids for a bubble-map fallback when the GeoJSON is unavailable.
"""

from __future__ import annotations

import json
from pathlib import Path

from air_raid_forecasting.logging_utils import get_logger

log = get_logger(__name__)

GEOJSON_URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/"
    "gbOpen/UKR/ADM1/geoBoundaries-UKR-ADM1_simplified.geojson"
)
GEOJSON_FILENAME = "ukraine_oblasts.geojson"

# geoBoundaries shapeName  ->  our canonical region name.
SHAPENAME_TO_CANONICAL: dict[str, str] = {
    "Vinnytsia Oblast": "Vinnytska oblast",
    "Volyn Oblast": "Volynska oblast",
    "Dnipropetrovsk Oblast": "Dnipropetrovska oblast",
    "Donetsk Oblast": "Donetska oblast",
    "Zhytomyr Oblast": "Zhytomyrska oblast",
    "Zakarpattia Oblast": "Zakarpatska oblast",
    "Zaporizhia Oblast": "Zaporizka oblast",
    "Ivano-Frankivsk Oblast": "Ivano-Frankivska oblast",
    "Kyiv Oblast": "Kyivska oblast",
    "Kirovohrad Oblast": "Kirovohradska oblast",
    "Luhansk Oblast": "Luhanska oblast",
    "Lviv Oblast": "Lvivska oblast",
    "Mykolaiv Oblast": "Mykolaivska oblast",
    "Odessa Oblast": "Odeska oblast",
    "Poltava Oblast": "Poltavska oblast",
    "Rivne Oblast": "Rivnenska oblast",
    "Sumy Oblast": "Sumska oblast",
    "Ternopil Oblast": "Ternopilska oblast",
    "Kharkiv Oblast": "Kharkivska oblast",
    "Kherson Oblast": "Khersonska oblast",
    "Khmelnytskyi Oblast": "Khmelnytska oblast",
    "Cherkasy Oblast": "Cherkaska oblast",
    "Chernivtsi Oblast": "Chernivetska oblast",
    "Chernihiv Oblast": "Chernihivska oblast",
    "Kyiv": "Kyiv City",
    "Sevastopol": "Sevastopol",
    "Autonomous Republic of Crimea": "Avtonomna Respublika Krym",
}


def _external_path(external_dir: str | Path) -> Path:
    return Path(external_dir) / GEOJSON_FILENAME


def download_geojson(external_dir: str | Path, timeout: int = 60) -> Path:
    """Download the oblast GeoJSON into *external_dir* (idempotent)."""
    import requests

    dest = _external_path(external_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    log.info("Downloading Ukraine oblast GeoJSON -> %s", dest)
    resp = requests.get(GEOJSON_URL, timeout=timeout)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load_geojson(external_dir: str | Path, download: bool = True) -> dict:
    """Load the GeoJSON and annotate each feature with ``region_canonical``.

    The choropleth keys on ``properties.region_canonical``. Features that map to
    excluded/permanent-siren regions keep a canonical name too (so they can be
    rendered greyed-out as 'no data').
    """
    path = _external_path(external_dir)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"GeoJSON not found at {path}")
        path = download_geojson(external_dir)
    geo = json.loads(path.read_text(encoding="utf-8"))
    for feat in geo.get("features", []):
        shape = feat.get("properties", {}).get("shapeName")
        feat.setdefault("properties", {})["region_canonical"] = SHAPENAME_TO_CANONICAL.get(shape)
    return geo


def _polygon_centroid(geometry: dict) -> tuple[float, float] | None:
    """Approximate centroid (lon, lat) as the mean of all coordinate pairs."""
    coords: list[list[float]] = []

    def _walk(node):
        if isinstance(node, (list, tuple)):
            if len(node) == 2 and all(isinstance(c, (int, float)) for c in node):
                coords.append(node)
            else:
                for child in node:
                    _walk(child)

    _walk(geometry.get("coordinates", []))
    if not coords:
        return None
    lon = sum(c[0] for c in coords) / len(coords)
    lat = sum(c[1] for c in coords) / len(coords)
    return lon, lat


def region_centroids(geojson: dict) -> dict[str, tuple[float, float]]:
    """Map canonical region name -> (lon, lat) centroid for the bubble fallback."""
    out: dict[str, tuple[float, float]] = {}
    for feat in geojson.get("features", []):
        region = feat.get("properties", {}).get("region_canonical")
        if not region:
            continue
        c = _polygon_centroid(feat.get("geometry", {}))
        if c:
            out[region] = c
    return out
