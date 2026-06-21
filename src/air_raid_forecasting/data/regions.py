"""Region normalization and administrative metadata.

The source data labels alerts by ``oblast`` (region) — and, since late 2025,
also by ``raion`` (district) and ``hromada`` (community). For consistent
forecasting across the whole 2022+ history we normalize everything up to the
**oblast** level, since district-level data only exists for the tail of the
series.

This module provides:
* the canonical list of Ukraine's 27 top-level regions (24 oblasts + Kyiv City
  + Sevastopol + Crimea),
* an alias map that folds known spelling variants onto the canonical name,
* :func:`normalize_region` used throughout the cleaning pipeline,
* :func:`load_states` to parse the upstream ``states.json`` into a tidy frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# Canonical English transliterations as they appear in the upstream dataset.
CANONICAL_OBLASTS: list[str] = [
    "Vinnytska oblast",
    "Volynska oblast",
    "Dnipropetrovska oblast",
    "Donetska oblast",
    "Zhytomyrska oblast",
    "Zakarpatska oblast",
    "Zaporizka oblast",
    "Ivano-Frankivska oblast",
    "Kyivska oblast",
    "Kirovohradska oblast",
    "Luhanska oblast",
    "Lvivska oblast",
    "Mykolaivska oblast",
    "Odeska oblast",
    "Poltavska oblast",
    "Rivnenska oblast",
    "Sumska oblast",
    "Ternopilska oblast",
    "Kharkivska oblast",
    "Khersonska oblast",
    "Khmelnytska oblast",
    "Cherkaska oblast",
    "Chernivetska oblast",
    "Chernihivska oblast",
    "Kyiv City",
    "Sevastopol",
    "Avtonomna Respublika Krym",
]

# Map of known spelling / formatting variants -> canonical region name.
ALIASES: dict[str, str] = {
    "m. kyiv": "Kyiv City",
    "kyiv": "Kyiv City",
    "kyiv city": "Kyiv City",
    "misto kyiv": "Kyiv City",
    "kyivska mts": "Kyiv City",
    "m. sevastopol": "Sevastopol",
    "sevastopilska mizka rada": "Sevastopol",
    "crimea": "Avtonomna Respublika Krym",
    "ar krym": "Avtonomna Respublika Krym",
    "republic of crimea": "Avtonomna Respublika Krym",
}

# Short, human-friendly display names (used by the dashboard / API examples).
SHORT_NAMES: dict[str, str] = {
    "Vinnytska oblast": "Vinnytsia",
    "Volynska oblast": "Volyn",
    "Dnipropetrovska oblast": "Dnipropetrovsk",
    "Donetska oblast": "Donetsk",
    "Zhytomyrska oblast": "Zhytomyr",
    "Zakarpatska oblast": "Zakarpattia",
    "Zaporizka oblast": "Zaporizhzhia",
    "Ivano-Frankivska oblast": "Ivano-Frankivsk",
    "Kyivska oblast": "Kyiv Oblast",
    "Kirovohradska oblast": "Kirovohrad",
    "Luhanska oblast": "Luhansk",
    "Lvivska oblast": "Lviv",
    "Mykolaivska oblast": "Mykolaiv",
    "Odeska oblast": "Odesa",
    "Poltavska oblast": "Poltava",
    "Rivnenska oblast": "Rivne",
    "Sumska oblast": "Sumy",
    "Ternopilska oblast": "Ternopil",
    "Kharkivska oblast": "Kharkiv",
    "Khersonska oblast": "Kherson",
    "Khmelnytska oblast": "Khmelnytskyi",
    "Cherkaska oblast": "Cherkasy",
    "Chernivetska oblast": "Chernivtsi",
    "Chernihivska oblast": "Chernihiv",
    "Kyiv City": "Kyiv",
    "Sevastopol": "Sevastopol",
    "Avtonomna Respublika Krym": "Crimea",
}

_CANONICAL_LOWER = {name.lower(): name for name in CANONICAL_OBLASTS}
# Allow lookups by short name too (e.g. user passes "Kyiv" or "Lviv").
_SHORT_LOWER = {short.lower(): full for full, short in SHORT_NAMES.items()}


def normalize_region(name: str | None) -> str | None:
    """Fold a raw region label onto its canonical oblast name.

    Returns ``None`` for empty / unrecognized input so callers can decide how
    to handle it (the cleaner drops un-mappable rows and logs the count).
    """
    if name is None:
        return None
    key = str(name).strip()
    if not key:
        return None
    low = key.lower()
    if low in _CANONICAL_LOWER:
        return _CANONICAL_LOWER[low]
    if low in ALIASES:
        return ALIASES[low]
    if low in _SHORT_LOWER:
        return _SHORT_LOWER[low]
    # Heuristic: tolerate trailing/inner whitespace and "oblast'" apostrophes.
    squashed = " ".join(low.replace("'", "").split())
    if squashed in _CANONICAL_LOWER:
        return _CANONICAL_LOWER[squashed]
    return None


def short_name(region: str) -> str:
    """Human-friendly short label for a canonical region."""
    return SHORT_NAMES.get(region, region)


def load_states(path: str | Path) -> pd.DataFrame:
    """Parse the upstream ``states.json`` into a tidy oblast/raion/hromada frame.

    Returns columns: ``oblast``, ``oblast_id``, ``raion``, ``raion_id``,
    ``hromada``, ``hromada_id`` (Ukrainian names as published upstream). Used
    for reference / metadata; the modeling pipeline keys off oblast names.
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    states = payload.get("states", payload) if isinstance(payload, dict) else payload
    rows: list[dict] = []
    for state in states:
        oblast = state.get("stateName")
        oblast_id = state.get("stateId")
        districts = state.get("districts", []) or []
        if not districts:
            rows.append(
                {"oblast": oblast, "oblast_id": oblast_id, "raion": None,
                 "raion_id": None, "hromada": None, "hromada_id": None}
            )
        for district in districts:
            raion = district.get("districtName")
            raion_id = district.get("districtId")
            communities = district.get("communities", []) or []
            if not communities:
                rows.append(
                    {"oblast": oblast, "oblast_id": oblast_id, "raion": raion,
                     "raion_id": raion_id, "hromada": None, "hromada_id": None}
                )
            for community in communities:
                rows.append(
                    {
                        "oblast": oblast,
                        "oblast_id": oblast_id,
                        "raion": raion,
                        "raion_id": raion_id,
                        "hromada": community.get("communityName"),
                        "hromada_id": community.get("communityId"),
                    }
                )
    return pd.DataFrame(rows)
