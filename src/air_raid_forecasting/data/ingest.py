"""Phase 1 — Data acquisition (LIVE DOWNLOAD ONLY).

Downloads historical Ukrainian air raid alert data from the public, token-free
`Vadimkin/ukrainian-air-raid-sirens-dataset
<https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset>`_ repository.

Two datasets are fetched:

* **official** (``official_data_en.csv``) — authoritative records from
  2022-03-15 onward, labelled by oblast / raion / hromada. All times UTC.
* **volunteer** (``volunteer_data_en.csv``) — community-collected records from
  2022-02-25 (oblast level only), extending the early history.

Plus ``states.json`` (administrative metadata). Raw downloads land in
``data/raw`` together with a ``_manifest.json`` recording the source URL, byte
count, SHA-256 and download timestamp for full provenance.

Per project requirements there is **no synthetic fallback**: if a source is
unreachable the ingest fails loudly so the data lineage stays honest.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from air_raid_forecasting.config import Config, load_config
from air_raid_forecasting.logging_utils import get_logger

log = get_logger(__name__)

# Unified raw alert schema produced by :func:`load_raw_alerts`.
RAW_COLUMNS = [
    "region_raw",
    "raion",
    "hromada",
    "level",
    "started_at",
    "finished_at",
    "naive",
    "source_dataset",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, timeout: int = 120) -> int:
    """Stream *url* to *dest*. Returns bytes written. Raises on any failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s -> %s", url, dest)
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            written = 0
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    if chunk:
                        fh.write(chunk)
                        written += len(chunk)
            tmp.replace(dest)
    except requests.RequestException as exc:  # network / HTTP errors
        raise RuntimeError(
            f"LIVE DOWNLOAD FAILED for {url!r}. This project is configured for "
            f"live-download-only data acquisition (no synthetic fallback). "
            f"Check connectivity and the source URL. Original error: {exc}"
        ) from exc
    if written == 0:
        raise RuntimeError(f"Downloaded 0 bytes from {url!r}; aborting.")
    log.info("  -> %s (%.1f MB)", dest.name, written / 1e6)
    return written


def ingest(
    cfg: Config | None = None,
    which: list[str] | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Download configured sources into ``data/raw`` and write a manifest.

    Parameters
    ----------
    which:
        Subset of source keys to fetch (``official``/``volunteer``/``states``).
        Defaults to all configured sources.
    force:
        Re-download even if the file already exists locally.
    """
    cfg = cfg or load_config()
    cfg.ensure_dirs()
    raw_dir = Path(cfg.paths.raw_dir)
    keys = which or list(cfg.data.sources.keys())

    manifest: dict[str, dict] = {}
    paths: dict[str, Path] = {}
    for key in keys:
        src = cfg.data.sources[key]
        url = f"{cfg.data.base_url}/{src.path}"
        dest = raw_dir / src.filename
        if dest.exists() and not force:
            log.info("Source %s already present (%s); skipping (use force=True to refresh).",
                     key, dest.name)
        else:
            download_file(url, dest, timeout=cfg.data.download_timeout_seconds)
        paths[key] = dest
        manifest[key] = {
            "url": url,
            "filename": src.filename,
            "bytes": dest.stat().st_size,
            "sha256": _sha256(dest),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        }

    manifest_path = raw_dir / "_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    log.info("Wrote provenance manifest -> %s", manifest_path)
    return paths


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_raw_alerts(cfg: Config | None = None, source: str | None = None) -> pd.DataFrame:
    """Load a raw alert CSV into the unified :data:`RAW_COLUMNS` schema.

    Timestamps are parsed as timezone-aware UTC. ``source`` defaults to the
    configured ``primary_source``.
    """
    cfg = cfg or load_config()
    source = source or cfg.data.primary_source
    raw_dir = Path(cfg.paths.raw_dir)
    fname = cfg.data.sources[source].filename
    path = raw_dir / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Raw file {path} not found. Run the ingest step first "
            f"(python -m air_raid_forecasting.pipeline.run_ingest)."
        )

    df = _read_csv(path)
    out = pd.DataFrame()
    if source == "official":
        out["region_raw"] = df["oblast"]
        out["raion"] = df.get("raion")
        out["hromada"] = df.get("hromada")
        out["level"] = df.get("level")
        out["naive"] = False
    elif source == "volunteer":
        out["region_raw"] = df["region"]
        out["raion"] = None
        out["hromada"] = None
        out["level"] = "oblast"
        out["naive"] = df.get("naive", False)
    else:
        raise ValueError(f"Unknown alert source {source!r}")

    out["started_at"] = pd.to_datetime(df["started_at"], utc=True, errors="coerce")
    out["finished_at"] = pd.to_datetime(df["finished_at"], utc=True, errors="coerce")
    out["source_dataset"] = source
    # Normalize the 'naive' column to real booleans.
    out["naive"] = (
        out["naive"].map({"True": True, "False": False, True: True, False: False})
        .fillna(False)
        .astype(bool)
    )
    log.info("Loaded %s raw rows from %s (%s)", f"{len(out):,}", fname, source)
    return out[RAW_COLUMNS]


def available_raw(cfg: Config | None = None) -> dict[str, bool]:
    """Report which configured raw files are present on disk."""
    cfg = cfg or load_config()
    raw_dir = Path(cfg.paths.raw_dir)
    return {
        key: (raw_dir / src.filename).exists()
        for key, src in cfg.data.sources.items()
    }
