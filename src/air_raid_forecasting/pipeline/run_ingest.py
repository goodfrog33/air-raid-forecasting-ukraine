"""Pipeline step 1: live-download the raw air raid alert data.

    python -m air_raid_forecasting.pipeline.run_ingest [--force] [--which official volunteer states]
"""

from __future__ import annotations

import argparse

from air_raid_forecasting.config import load_config
from air_raid_forecasting.data.ingest import ingest
from air_raid_forecasting.logging_utils import get_logger

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Download raw air raid alert datasets.")
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    parser.add_argument(
        "--which", nargs="*", default=None,
        help="Subset of sources (official volunteer states). Default: all.",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    paths = ingest(cfg, which=args.which, force=args.force)
    log.info("Ingest complete. Files:")
    for key, path in paths.items():
        log.info("  %-10s %s", key, path)
    return paths


if __name__ == "__main__":
    main()
