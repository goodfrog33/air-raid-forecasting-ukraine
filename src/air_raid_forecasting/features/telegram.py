"""Telegram channel signal features (Phase 4, optional factor — separate from GDELT).

Scrapes the public web preview (``https://t.me/s/<channel>``) of war-monitoring
channels for a daily "threat-message intensity" signal, and turns it into
leakage-safe lagged features that can be toggled on as a *separate* prediction
factor alongside the GDELT news factor.

Default channels: ``war_monitor`` and ``radar_raketaa`` (real-time Ukrainian
missile/drone alert channels). The web preview only exposes recent posts (with
bounded ``?before=`` pagination), so historical depth is limited — coverage is
recent-heavy. Features use only **lagged** values, so they stay forecasting-safe.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd

from air_raid_forecasting.logging_utils import get_logger

log = get_logger(__name__)

DEFAULT_CHANNELS = ["war_monitor", "radar_raketaa"]
TELEGRAM_RAW_FILE = "telegram_posts.parquet"
TELEGRAM_FEATURES_FILE = "telegram_features.parquet"
TELEGRAM_FEATURE_COLS = ["tg_vol_lag1", "tg_vol_lag2", "tg_vol_roll7", "tg_threat_lag1"]

# Threat keywords (Ukrainian + transliteration) used to flag a "threat" post.
_THREAT = re.compile(
    r"ракет|кинджал|шахед|geran|герань|дрон|бпла|балісти|калібр|пуск|зліт|вильот|"
    r"іскандер|missile|drone|shahed|ballistic|launch|kinzhal|kalibr|mig|міг",
    re.IGNORECASE,
)
_TAG = re.compile(r"<[^>]+>")
_DT = re.compile(r'datetime="([^"]+)"')
_POST_ID = re.compile(r'data-post="[^"/]+/(\d+)"')
_TEXT = re.compile(r'tgme_widget_message_text[^>]*>(.*?)</div>', re.S)


def _parse_page(page: str) -> tuple[list[dict], int | None]:
    """Return (records, min_post_id) from one t.me/s page."""
    records: list[dict] = []
    # Split into per-message chunks; the first chunk is the page header.
    for block in page.split("tgme_widget_message_wrap")[1:]:
        dt = _DT.search(block)
        if not dt:
            continue
        txt_m = _TEXT.search(block)
        text = html.unescape(_TAG.sub(" ", txt_m.group(1))) if txt_m else ""
        records.append({"ts": dt.group(1), "threat": bool(_THREAT.search(text))})
    ids = [int(x) for x in _POST_ID.findall(page)]
    return records, (min(ids) if ids else None)


def fetch_telegram_daily(
    external_dir: str | Path,
    channels: list[str] | None = None,
    max_pages: int = 40,
    delay: float = 1.2,
    timeout: int = 25,
    force: bool = False,
) -> pd.DataFrame:
    """Scrape recent posts (bounded) and aggregate to a daily intensity series.

    Returns columns ``date``, ``tg_volume`` (posts/day, summed over channels) and
    ``tg_threat`` (threat-flagged posts/day). Cached to parquet.
    """
    import time

    import requests

    external_dir = Path(external_dir)
    external_dir.mkdir(parents=True, exist_ok=True)
    cache = external_dir / TELEGRAM_RAW_FILE
    if cache.exists() and not force:
        df = pd.read_parquet(cache)
        log.info("Loaded cached Telegram posts (%s rows) from %s", f"{len(df):,}", cache.name)
        return _aggregate(df)

    channels = channels or DEFAULT_CHANNELS
    headers = {"User-Agent": "Mozilla/5.0 (compatible; air-raid-forecasting/1.0)"}
    rows: list[dict] = []
    for ch in channels:
        before: int | None = None
        seen_pages = 0
        while seen_pages < max_pages:
            url = f"https://t.me/s/{ch}" + (f"?before={before}" if before else "")
            try:
                resp = requests.get(url, headers=headers, timeout=timeout)
                resp.raise_for_status()
            except Exception as exc:
                log.warning("Telegram fetch %s (before=%s) failed: %s", ch, before, exc)
                break
            recs, min_id = _parse_page(resp.text)
            if not recs:
                break
            for r in recs:
                r["channel"] = ch
            rows.extend(recs)
            seen_pages += 1
            if min_id is None or (before is not None and min_id >= before):
                break  # no further pagination possible
            before = min_id
            time.sleep(delay)
        log.info("  scraped %s: %d pages", ch, seen_pages)

    if not rows:
        raise RuntimeError("Telegram scrape returned no posts; factor unavailable.")
    posts = pd.DataFrame(rows)
    posts["ts"] = pd.to_datetime(posts["ts"], utc=True)
    posts = posts.drop_duplicates(subset=["channel", "ts", "threat"]).reset_index(drop=True)
    posts.to_parquet(cache, index=False)
    log.info("Scraped %s Telegram posts (%s..%s) from %d channels -> %s",
             f"{len(posts):,}", posts["ts"].min().date(), posts["ts"].max().date(),
             len(channels), cache.name)
    return _aggregate(posts)


def _aggregate(posts: pd.DataFrame) -> pd.DataFrame:
    posts = posts.copy()
    posts["date"] = pd.to_datetime(posts["ts"], utc=True).dt.normalize()
    daily = posts.groupby("date").agg(
        tg_volume=("ts", "size"),
        tg_threat=("threat", "sum"),
    ).reset_index()
    return daily.sort_values("date").reset_index(drop=True)


def build_telegram_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.sort_values("date").reset_index(drop=True).copy()
    vol, threat = df["tg_volume"].astype(float), df["tg_threat"].astype(float)
    df["tg_vol_lag1"] = vol.shift(1)
    df["tg_vol_lag2"] = vol.shift(2)
    df["tg_vol_roll7"] = vol.shift(1).rolling(7, min_periods=1).mean()
    df["tg_threat_lag1"] = threat.shift(1)
    out = df[["date"] + TELEGRAM_FEATURE_COLS].copy()
    out[TELEGRAM_FEATURE_COLS] = out[TELEGRAM_FEATURE_COLS].fillna(0.0)
    return out


def attach_telegram_features(panel: pd.DataFrame, tg_features: pd.DataFrame,
                             ts_col: str = "timestamp") -> pd.DataFrame:
    df = panel.copy()
    df["_date"] = pd.to_datetime(df[ts_col], utc=True).dt.normalize()
    merged = df.merge(tg_features.rename(columns={"date": "_date"}), on="_date", how="left")
    merged[TELEGRAM_FEATURE_COLS] = merged[TELEGRAM_FEATURE_COLS].fillna(0.0)
    return merged.drop(columns=["_date"])
