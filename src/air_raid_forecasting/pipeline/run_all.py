"""Run the full pipeline end to end: ingest -> preprocess -> eda -> features -> train.

    python -m air_raid_forecasting.pipeline.run_all [--fast] [--skip-ingest]
"""

from __future__ import annotations

import argparse

from air_raid_forecasting.logging_utils import get_logger
from air_raid_forecasting.pipeline import (
    run_eda,
    run_features,
    run_ingest,
    run_preprocess,
    run_train,
)

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the entire forecasting pipeline.")
    parser.add_argument("--fast", action="store_true", help="Fewer folds / quicker training.")
    parser.add_argument("--skip-ingest", action="store_true", help="Reuse already-downloaded data.")
    parser.add_argument("--skip-eda", action="store_true", help="Skip figure generation.")
    args = parser.parse_args(argv)

    if not args.skip_ingest:
        log.info("STEP 1/5 — ingest"); run_ingest.main([])
    log.info("STEP 2/5 — preprocess"); run_preprocess.main([])
    if not args.skip_eda:
        log.info("STEP 3/5 — eda"); run_eda.main([])
    log.info("STEP 4/5 — features"); run_features.main([])
    log.info("STEP 5/5 — train")
    run_train.main(["--fast"] if args.fast else [])
    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
