"""News / event-signal features from GDELT (Phase 4, optional factor).

Fetches a daily "war-news intensity" signal from the token-free GDELT DOC 2.0
API (``timelinevol`` = share of global news coverage matching the query) for
Ukraine strike-related coverage, and turns it into leakage-safe features that
can be toggled on/off as a prediction factor.

The signal is a *national* daily tempo broadcast to every region — a proxy for
how intense the air-war news cycle is on a given day. Features use only
**lagged** news (yesterday and earlier), so they are safe for forecasting.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pandas as pd

from air_raid_forecasting.logging_utils import get_logger

log = get_logger(__name__)

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_QUERY = 'ukraine (missile OR drone OR "air raid" OR shelling OR airstrike OR rocket)'
NEWS_RAW_FILE = "news_gdelt.parquet"
NEWS_FEATURES_FILE = "news_features.parquet"

# Feature columns this module produces (all lagged / leakage-safe).
NEWS_FEATURE_COLS = ["news_vol_lag1", "news_vol_lag2", "news_vol_lag7", "news_vol_roll7"]


def _chunk_ranges(start: pd.Timestamp, end: pd.Timestamp, days: int = 90):
    cur = start
    step = pd.Timedelta(days=days)
    while cur < end:
        nxt = min(cur + step, end)
        yield cur, nxt
        cur = nxt


def fetch_gdelt_daily(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    external_dir: str | Path,
    query: str = DEFAULT_QUERY,
    timeout: int = 60,
    force: bool = False,
    delay: float = 6.0,
    retries: int = 3,
) -> pd.DataFrame:
    """Fetch the daily GDELT volume series (chunked, rate-limited), cache to parquet.

    GDELT enforces ~1 request / 5 s, so we fetch in **yearly** chunks (≈5
    requests for the full history; each returns daily resolution) and pause
    ``delay`` (>5 s) between them. Returns columns ``date`` (UTC) and
    ``news_volume``.
    """
    import time

    import requests

    external_dir = Path(external_dir)
    external_dir.mkdir(parents=True, exist_ok=True)
    cache = external_dir / NEWS_RAW_FILE

    start, end = pd.Timestamp(start), pd.Timestamp(end)
    # Incremental: keep what we already have, fetch only uncovered yearly chunks.
    existing = pd.read_parquet(cache) if (cache.exists() and not force) else pd.DataFrame(columns=["date", "news_volume"])
    covered = set(existing["date"]) if len(existing) else set()
    rows: list[dict] = existing.to_dict("records") if len(existing) else []
    q = quote(query)
    chunks = list(_chunk_ranges(start, end, days=365))
    for i, (c0, c1) in enumerate(chunks):
        want = pd.date_range(c0.normalize(), (c1 - pd.Timedelta(days=1)).normalize(), freq="D", tz="UTC")
        if len(want) and sum(d in covered for d in want) / len(want) > 0.9:
            continue  # this year is already covered — skip the request
        url = (f"{GDELT_URL}?query={q}&mode=timelinevol&format=json"
               f"&startdatetime={c0.strftime('%Y%m%d%H%M%S')}"
               f"&enddatetime={c1.strftime('%Y%m%d%H%M%S')}")
        for attempt in range(retries):
            try:
                resp = requests.get(url, timeout=timeout)
                if resp.status_code == 429:
                    raise requests.HTTPError("429 rate limited")
                resp.raise_for_status()
                tl = resp.json().get("timeline", [])
                if tl:
                    for pt in tl[0].get("data", []):
                        rows.append({"date": pt["date"], "news_volume": float(pt["value"])})
                break
            except Exception as exc:  # backoff and retry; skip the chunk if it keeps failing
                if attempt == retries - 1:
                    log.warning("GDELT chunk %s..%s failed after %d tries: %s",
                                c0.date(), c1.date(), retries, exc)
                else:
                    time.sleep(delay * (attempt + 2))
        if i < len(chunks) - 1:
            time.sleep(delay)

    if not rows:
        raise RuntimeError("GDELT returned no data; news factor unavailable.")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    df.to_parquet(cache, index=False)
    log.info("Fetched GDELT news: %s daily points (%s..%s) -> %s",
             f"{len(df):,}", df["date"].min().date(), df["date"].max().date(), cache.name)
    return df


def build_news_features(news_daily: pd.DataFrame) -> pd.DataFrame:
    """Daily leakage-safe news features keyed by date (uses lagged values only)."""
    df = news_daily.sort_values("date").reset_index(drop=True).copy()
    vol = df["news_volume"]
    df["news_vol_lag1"] = vol.shift(1)
    df["news_vol_lag2"] = vol.shift(2)
    df["news_vol_lag7"] = vol.shift(7)
    df["news_vol_roll7"] = vol.shift(1).rolling(7, min_periods=1).mean()
    out = df[["date"] + NEWS_FEATURE_COLS].copy()
    out[NEWS_FEATURE_COLS] = out[NEWS_FEATURE_COLS].fillna(0.0)
    return out


def attach_news_features(panel: pd.DataFrame, news_features: pd.DataFrame,
                         ts_col: str = "timestamp") -> pd.DataFrame:
    """Broadcast daily news features onto an hourly panel by calendar date."""
    df = panel.copy()
    df["_date"] = pd.to_datetime(df[ts_col], utc=True).dt.normalize()
    merged = df.merge(news_features.rename(columns={"date": "_date"}), on="_date", how="left")
    merged[NEWS_FEATURE_COLS] = merged[NEWS_FEATURE_COLS].fillna(0.0)
    return merged.drop(columns=["_date"])
